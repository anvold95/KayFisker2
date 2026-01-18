import os
import json
import re
import torch
import numpy as np
import soundfile as sf
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import hf_hub_download, login, whoami

try:
    from text_cleaner import clean_text
except ImportError:
    def clean_text(text): return text.strip()

# --- MILJØKONFIGURASJON ---
os.environ["HF_HOME"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print(f"✅ Logget inn på Hugging Face som: {whoami().get('name')}")
    except Exception as e:
        print(f"⚠️ Hugging Face Login feil: {e}")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "kay-fisker-corpus-1024")
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# Last timeline
FISKER_TIMELINE = ""
try:
    if os.path.exists("timeline_fisker.txt"):
        with open("timeline_fisker.txt", "r", encoding="utf-8") as f:
            FISKER_TIMELINE = f.read().strip()
except Exception as e:
    print(f"⚠️ Kunne ikke laste timeline: {e}")

app = FastAPI()

# --- GLOBALE VARIABLER ---
pipe = None
tts = None
tokenizer = None
embedder = None
reranker = None
pinecone_index = None

# --- VISUAL DATABASE ---
VISUALS_DB = {
    "vestersøhus": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Vesters%C3%B8hus_01.jpg/640px-Vesters%C3%B8hus_01.jpg", "type": "FOTO"},
    "dronningegården": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Dronningeg%C3%A5rden_Copenhagen_02.jpg/640px-Dronningeg%C3%A5rden_Copenhagen_02.jpg", "type": "FOTO"},
    "gullfoss": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Gullfoss_Copenhagen_01.jpg/640px-Gullfoss_Copenhagen_01.jpg", "type": "FOTO"},
    "gullfosshus": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Gullfoss_Copenhagen_01.jpg/640px-Gullfoss_Copenhagen_01.jpg", "type": "FOTO"},
    "aarhus universitet": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Aarhus_Universitet_01.jpg/640px-Aarhus_Universitet_01.jpg", "type": "FOTO"},
    "statsprøveanstalten": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Statspr%C3%B8veanstalten_01.jpg/640px-Statspr%C3%B8veanstalten_01.jpg", "type": "FOTO"},
    "plan": {"url": "https://images.unsplash.com/photo-1629196914375-f7e48f477b6d?q=80&w=600&auto=format&fit=crop", "type": "TEGNING"},
    "snit": {"url": "https://images.unsplash.com/photo-1503387762-592deb58ef4e?q=80&w=600&auto=format&fit=crop", "type": "TEGNING"},
    "mursten": {"url": "https://images.unsplash.com/photo-1596236900238-6b074495c024?q=80&w=600&auto=format&fit=crop", "type": "MATERIALE"}
}

# --- MODELL-LASTING ---
def load_model_logic():
    global pipe, tts, tokenizer, embedder, reranker, pinecone_index
    
    BASE = "mistralai/Mistral-7B-Instruct-v0.3"
    LORA = "anvold/fisker-lora-clean"
    print("🧩 Laster base-modell + LoRA-adapter …")
    
    try:
        adapter_path = hf_hub_download(LORA, "adapter_config.json")
        with open(adapter_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        allowed_keys = {
            "base_model_name_or_path", "bias", "inference_mode", "lora_alpha",
            "lora_dropout", "r", "target_modules", "task_type", "peft_type",
            "fan_in_fan_out", "use_rslora", "alpha_pattern", "rank_pattern"
        }
        cfg = {k: v for k, v in cfg.items() if k in allowed_keys}
        with open(adapter_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"⚠️ Feil under adapter-fix: {e}")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, LORA)
    model = model.merge_and_unload()
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    print("🎙️ Laster TTS …")
    try:
        from transformers import pipeline as hf_pipeline_tool
        tts = hf_pipeline_tool("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
        print("✅ TTS klar.")
    except Exception as e:
        print(f"⚠️ TTS feilet: {e}")

    print("🧠 Kobler til Pinecone …")
    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(INDEX_NAME)
            embedder = SentenceTransformer(EMBED_MODEL)
            reranker = CrossEncoder("BAAI/bge-reranker-base")
            print(f"✅ RAG aktiv på '{INDEX_NAME}'.")
        except Exception as e:
            print(f"❌ Pinecone feil: {e}")

@app.on_event("startup")
async def startup_event():
    load_model_logic()

# --- HJELPEFUNKSJONER ---
def normalize_orthography(txt: str) -> str:
    """Normaliserer OCR-tekst og fjerner bindestreker"""
    if not txt: return txt
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    txt = re.sub(r"[ \t]*\n[ \t]*", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def enhance_query(user_prompt: str) -> str:
    """Utvider query basert på intensjonsdeteksjon og nøkkelverk"""
    prompt_lower = user_prompt.lower()
    
    works_map = {
        "vestersøhus": "Vestersøhus Kay Fisker Vester Søgade København 1935",
        "dronningegården": "Dronningegården Kay Fisker Dronningens Tværgade",
        "gullfoss": "Gullfosshus Kay Fisker Artillerivej",
        "gullfosshus": "Gullfosshus Kay Fisker Artillerivej",
        "aarhus universitet": "Aarhus Universitet bygninger Kay Fisker C.F. Møller",
        "statsprøveanstalten": "Statsprøveanstalten Kay Fisker Amager Boulevard"
    }

    for work, expansion in works_map.items():
        if work in prompt_lower:
            return f"{expansion} {user_prompt}"

    is_biographical = any(w in prompt_lower for w in ["hvem", "hvad", "når", "hvor", "liv", "karriere"])
    is_theoretical = any(w in prompt_lower for w in ["hvorfor", "hvordan", "prinsipper", "teori", "tanker"])
    
    if is_biographical:
        return f"Kay Fisker biografi liv karriere {user_prompt}"
    elif is_theoretical:
        return f"arkitektonisk teori princip filosofi {user_prompt}"
    
    return f"arkitektur Kay Fisker {user_prompt}"

def extract_temporal_context(user_prompt: str, timeline: str) -> str:
    """Henter relevante timeline-segmenter basert på query"""
    if not timeline:
        return ""
    
    periods = []
    for line in timeline.split("\n"):
        if match := re.match(r"(\d{4})[-–]?(\d{4})?: (.+)", line):
            start, end, event = match.groups()
            periods.append({
                "start": int(start),
                "end": int(end) if end else int(start),
                "event": event.strip()
            })
    
    keywords = set(user_prompt.lower().split())
    relevant = []
    
    for p in periods:
        event_words = set(p["event"].lower().split())
        if keywords & event_words:
            relevant.append(p)
    
    if relevant:
        return "\n".join([f"{p['start']}: {p['event']}" for p in relevant[:3]])
    
    return ""

def extract_visuals(text: str) -> list:
    """Finner visuelle referanser i generert tekst"""
    found = []
    text_lower = text.lower()
    for key, data in VISUALS_DB.items():
        if key in text_lower:
            found.append({"keyword": key, "url": data["url"], "type": data["type"]})
    return found

# --- FORBEDRET SEGMENT-EKSTRAKSJON MED STRENGERE MATCHING ---
def extract_most_relevant_excerpt(source_text: str, response_text: str, min_words: int = 20, max_words: int = 80) -> dict:
    """
    Finner det mest relevante utdraget fra kilden basert på FAKTISK semantisk overlapp.
    Returnerer både utdrag og forklaringog relevans-score.
    """
    if not embedder or not source_text or not response_text:
        return {
            "excerpt": source_text[:200] if len(source_text) > 200 else source_text,
            "relevance": 0.0,
            "method": "fallback",
            "explanation": "Ingen semantisk analyse tilgjengelig"
        }
    
    # Split source into sentences
    sentences = re.split(r'(?<=[.!?])\s+', source_text)
    if not sentences or len(sentences) < 3:
        return {
            "excerpt": source_text[:200],
            "relevance": 0.0,
            "method": "fallback",
            "explanation": "For kort kildetekst for analyse"
        }
    
    # Create sliding windows of sentences (3-5 sentences per window)
    windows = []
    for window_size in [3, 4, 5]:
        for i in range(len(sentences) - window_size + 1):
            window_text = " ".join(sentences[i:i+window_size])
            word_count = len(window_text.split())
            if min_words <= word_count <= max_words:
                windows.append({
                    "text": window_text,
                    "start_idx": i,
                    "size": window_size
                })
    
    if not windows:
        words = source_text.split()[:max_words]
        return {
            "excerpt": " ".join(words),
            "relevance": 0.0,
            "method": "truncation",
            "explanation": "Ingen passende vinduer funnet, bruker begynnelsen"
        }
    
    # Encode response
    response_emb = embedder.encode(response_text, convert_to_tensor=True)
    
    # Encode all windows
    window_texts = [w["text"] for w in windows]
    window_embs = embedder.encode(window_texts, convert_to_tensor=True)
    
    # Calculate similarities
    similarities = util.cos_sim(response_emb, window_embs)[0].cpu().numpy()
    
    # Find best match
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    
    # Generate explanation based on score
    if best_score > 0.65:
        explanation = "Høy semantisk overlapp - direkte relatert til responsen"
    elif best_score > 0.50:
        explanation = "Moderat overlapp - kilden støtter responsen indirekte"
    elif best_score > 0.35:
        explanation = "Lav overlapp - kilden gir kontekstuell bakgrunn"
    else:
        explanation = "Minimal overlapp - kilden er i det passive resonansrommet"
    
    return {
        "excerpt": windows[best_idx]["text"],
        "relevance": best_score,
        "method": "semantic_matching",
        "window_position": windows[best_idx]["start_idx"],
        "explanation": explanation
    }

def find_attributable_segments(source_text: str, response_text: str, threshold: float = 0.50) -> list:
    """
    STRENGERE segment-matching med høyere threshold for å unngå false positives.
    Finner BARE segmenter som faktisk bidrar til responsen.
    """
    if not embedder:
        return []

    source_words = source_text.split()
    if len(source_words) < 10:
        return []

    segments = []
    window_size = 18  # Økt fra 15 for bedre kontekst
    step = 6  # Økt fra 5 for mindre overlapp

    for i in range(0, len(source_words) - window_size + 1, step):
        segments.append({
            "text": " ".join(source_words[i:i+window_size]),
            "start_idx": i,
            "end_idx": i + window_size
        })

    if not segments:
        return []

    response_emb = embedder.encode(response_text, convert_to_tensor=True)
    segment_embs = embedder.encode([s["text"] for s in segments], convert_to_tensor=True)
    similarities = util.cos_sim(response_emb, segment_embs)[0].cpu().numpy()

    attributed = []
    for idx, sim in enumerate(similarities):
        if sim > threshold:  # STRENGERE: 0.50 i stedet for 0.42
            attributed.append({
                "text": segments[idx]["text"],
                "similarity": float(sim),
                "position": segments[idx]["start_idx"],
                "confidence": "high" if sim > 0.65 else "medium"
            })
    
    # Sort by similarity, take only top 3
    attributed.sort(key=lambda x: x["similarity"], reverse=True)
    return attributed[:3]

def analyze_source_relations(strata: list, response_text: str, query: str) -> dict:
    """Forbedret genealogisk analyse med strengere vurdering"""
    if not embedder:
        return {}

    analysis = {
        "source_attributions": [],
        "source_comparisons": [],
        "query_source_relations": [],
        "response_grounding": {},
        "epistemic_levels": {}
    }

    # 1. Attributions med STRENGERE threshold
    total_attributed = 0
    for s in strata:
        attributed = find_attributable_segments(s["text"], response_text, threshold=0.55)
        if attributed:
            avg_similarity = sum(seg["similarity"] for seg in attributed) / len(attributed)
            
            # STRENGERE epistemisk vurdering
            if avg_similarity > 0.70:
                level = "ARKIVFAKTA"
            elif avg_similarity > 0.58:
                level = "ARKIV-NÆR"
            else:
                level = "PRAKSISBASERT"
            
            analysis["source_attributions"].append({
                "source_id": s.get("id", "unknown"),
                "source_ref": s["ref"],
                "segments": attributed,
                "attribution_strength": avg_similarity,
                "epistemic_level": level,
                "confidence": "high" if avg_similarity > 0.65 else "medium" if avg_similarity > 0.50 else "low"
            })
            total_attributed += len(attributed)

    # 2. Source Comparisons
    if len(strata) >= 2:
        source_texts = [s["text"] for s in strata]
        source_embs = embedder.encode(source_texts, convert_to_tensor=True)
        for i in range(len(strata)):
            for j in range(i + 1, len(strata)):
                sim = util.cos_sim(source_embs[i], source_embs[j])[0][0].item()
                if sim > 0.4:
                    analysis["source_comparisons"].append({
                        "source_a": strata[i]["ref"],
                        "source_b": strata[j]["ref"],
                        "similarity": float(sim),
                        "relation_type": "KONVERGENT" if sim > 0.7 else "RESONANT"
                    })

    # 3. Query-Source Relations
    query_emb = embedder.encode(query, convert_to_tensor=True)
    for s in strata:
        rel = util.cos_sim(query_emb, embedder.encode(s["text"], convert_to_tensor=True))[0][0].item()
        analysis["query_source_relations"].append({
            "source_ref": s["ref"],
            "query_relevance": float(rel),
            "relevance_category": "HØY" if rel > 0.6 else "MEDIUM" if rel > 0.4 else "LAV"
        })

    # 4. Response Grounding
    resp_emb = embedder.encode(response_text, convert_to_tensor=True)
    comb_emb = embedder.encode(" ".join([s["text"] for s in strata]), convert_to_tensor=True)
    g_score = util.cos_sim(resp_emb, comb_emb)[0][0].item()
    
    analysis["response_grounding"] = {
        "score": float(g_score),
        "assessment": "SOLID" if g_score > 0.7 else "MODERAT" if g_score > 0.5 else "SVAK",
        "warning": None if g_score > 0.5 else "Responsen har begrenset støtte i kildene - mulig NIVÅ 3 syntese"
    }
    
    # 5. Overall Epistemic Assessment
    if analysis["source_attributions"]:
        avg_attribution = sum(a["attribution_strength"] for a in analysis["source_attributions"]) / len(analysis["source_attributions"])
        high_conf_count = len([a for a in analysis["source_attributions"] if a["attribution_strength"] > 0.65])
        
        if avg_attribution > 0.70 and high_conf_count >= 2:
            overall_level = "ARKIVFAKTA"
        elif avg_attribution > 0.58 and high_conf_count >= 1:
            overall_level = "ARKIV-NÆR TOLKNING"
        else:
            overall_level = "PRAKSISBASERT SYNTESE"
        
        analysis["epistemic_levels"] = {
            "primary_level": overall_level,
            "average_attribution": float(avg_attribution),
            "high_confidence_sources": high_conf_count,
            "total_segments_attributed": total_attributed
        }
    else:
        analysis["epistemic_levels"] = {
            "primary_level": "PRAKSISBASERT SYNTESE",
            "average_attribution": 0.0,
            "high_confidence_sources": 0,
            "total_segments_attributed": 0,
            "warning": "Ingen direkte kildeattribusjon funnet"
        }

    return analysis

# --- RAG-LOGIKK ---
def _fetch_ns(ns: str, qvec, top_k: int):
    """Henter matches fra en spesifikk namespace"""
    try:
        res = pinecone_index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            namespace=ns
        )
        matches = res.get("matches", []) or []
        for m in matches:
            m["metadata"]["__ns"] = ns
        return matches
    except Exception as e:
        print(f"⚠️ Feil i namespace '{ns}': {e}")
        return []

def pinecone_search_logic(user_prompt: str, total_results: int = 6):
    """RAG-søk med Mix-strategi"""
    if not (pinecone_index and embedder):
        print("⚠️ Pinecone eller embedder ikke tilgjengelig")
        return []
    
    print(f"\n🔍 Analyserer: '{user_prompt}'")
    enhanced = enhance_query(user_prompt)
    print(f"   ↳ Utvida til: '{enhanced}'")
    qvec = embedder.encode(enhanced).tolist()
    
    ns_quotas = {"primary": 6, "secondary": 4, "quotes": 3}
    pool = defaultdict(list)
    
    for ns, quota in ns_quotas.items():
        pool[ns] = _fetch_ns(ns, qvec, top_k=quota)
        print(f"   ↳ {ns}: {len(pool[ns])} matches")

    cleaned_pool = defaultdict(list)
    for ns, matches in pool.items():
        for m in matches:
            md = m.get("metadata") or {}
            txt = clean_text(normalize_orthography(md.get("text") or ""))
            if len(txt.split()) > 8:
                md["text"] = txt
                cleaned_pool[ns].append(m)

    if reranker:
        print("   🔄 Reranker...")
        for ns in cleaned_pool:
            if not cleaned_pool[ns]:
                continue
            pairs = [(user_prompt, m["metadata"]["text"]) for m in cleaned_pool[ns]]
            scores = reranker.predict(pairs)
            cleaned_pool[ns] = [x for _, x in sorted(zip(scores, cleaned_pool[ns]), key=lambda z: z[0], reverse=True)]

    final_selection = []
    
    if cleaned_pool["secondary"]:
        final_selection.append(cleaned_pool["secondary"].pop(0))
    if cleaned_pool["quotes"]:
        final_selection.append(cleaned_pool["quotes"].pop(0))
    
    remaining = []
    remaining.extend(cleaned_pool["primary"])
    remaining.extend(cleaned_pool["secondary"])
    remaining.extend(cleaned_pool["quotes"])
    
    if reranker and remaining:
        pairs = [(user_prompt, m["metadata"]["text"]) for m in remaining]
        scores = reranker.predict(pairs)
        remaining = [x for _, x in sorted(zip(scores, remaining), key=lambda z: z[0], reverse=True)]
    
    while len(final_selection) < total_results and remaining:
        candidate = remaining.pop(0)
        if candidate not in final_selection:
            final_selection.append(candidate)
    
    print(f"   ✅ Returnerer {len(final_selection)} diverse kilder (Mix)")
    
    for i, m in enumerate(final_selection):
        md = m["metadata"]
        print(f"      [{i+1}] {md.get('__ns', '?').upper()} | {md.get('year', '?')} | {md.get('text', '')[:60]}...")
    
    return final_selection

# --- API ENDEPUNKT ---
@app.post("/chat")
async def api_chat(req: Request):
    data = await req.json()
    user_prompt = data.get("message", "")
    
    if len(user_prompt.strip()) < 3:
        return JSONResponse({"error": "Spørsmål for kort"}, 400)
    
    matches = pinecone_search_logic(user_prompt, total_results=6)
    
    if not matches:
        return JSONResponse({
            "error": "Ingen kilder funnet",
            "text": "Beklager, jeg kan ikke svare uten tilgang til arkivet.",
            "strata": [],
            "visuals": [],
            "state": "FRAKOBLET",
            "intensity": 0.0,
            "genealogy": {}
        }, 200)
    
    context = "\n---\n".join([m["metadata"]["text"] for m in matches])
    
    bio_context = ""
    for m in matches:
        if m["metadata"].get("__ns") == "secondary":
            bio_context = m["metadata"]["text"]
            break
    
    temporal_context = extract_temporal_context(user_prompt, FISKER_TIMELINE)
    
    # HYBRID EPISTEMISK PROMPT
    system_prompt = (
        "Du er arkitekten Kay Fisker (1893–1965). "
        "Du skal skille tydeligt mellem tre epistemiske niveauer:\n\n"
        
        "NIVÅ 1 — ARKIVFAKTA (hard binding til kilder):\n"
        "- Kun eksplisitte sitater eller entydige parafraser fra kildematerialet\n"
        "- Alltid kildehenvisning eller årstal\n"
        "- Eksempel: 'I Byplanproblemer (1933) skriver jeg at...'\n\n"
        
        "NIVÅ 2 — ARKIV-NÆR TOLKNING (kontrollert inferens):\n"
        "- Støttet av flere arkivfragmenter, men som tolkning\n"
        "- Språklig markert: 'Kilderne viser at...', 'I mine skrifter fra 1930-årene fremstår...'\n"
        "- Ingen nye begreper som ikke finnes i materialet\n\n"
        
        "NIVÅ 3 — PRAKSISBASERT SYNTESE (arkitekturhistorisk kontekst):\n"
        "- Tillatt å bruke bredere faglig kontekst når arkivet er taust\n"
        "- ALLTID markert: 'Set i lyset af tidens strømninger...', 'Som arkitekt af min generation...'\n"
        "- Aldri presentert som direkte sitat eller eksplisitt mening\n\n"
        
        "KRITISK REGEL:\n"
        "Du må ALDRI formulere en tolkning eller kontekstuell refleksjon som om den var et direkt arkivutsagn.\n"
        "Hvis du beveger deg bort fra eksplisitte kilder, skal dette markeres språklig.\n\n"
        
        "FORBUDT:\n"
        "- Opfinnelse av konkrete årstal, adresser eller fakta som ikke finnes i kildene\n"
        "- Umarkerte generaliseringer presentert som fakta\n\n"
        
        "Hold svarene korte (max 2-3 setninger) og i et nøgternt fagsprog."
    )
    
    if bio_context:
        system_prompt += f"\n\nBiografisk kontekst (for tolkning):\n{bio_context}\n"
    if temporal_context:
        system_prompt += f"\nRelevant tidsperiode:\n{temporal_context}\n"
    
    full_prompt = (
        f"System: {system_prompt}\n\n"
        f"### KILDEMATERIALE START ###\n{context}\n### KILDEMATERIALE SLUT ###\n\n"
        f"Spørgsmål: {user_prompt}\n\n"
        f"Kay Fisker:"
    )
    
    result = pipe(
        full_prompt,
        max_new_tokens=200,
        temperature=0.25,
        top_p=0.92,
        repetition_penalty=1.12,
        do_sample=True
    )
    
    generated_text = result[0]["generated_text"].split("Kay Fisker:")[-1].strip()
    generated_text = re.sub(r"^(System:|Spørgsmål:|###).*", "", generated_text, flags=re.MULTILINE).strip()
    generated_text = re.sub(r"\n+", " ", generated_text).strip()
    
    # Bygg strata MED relevante utdrag
    strata = []
    for i, m in enumerate(matches):
        full_text = m["metadata"].get("text", "")
        
        excerpt_info = extract_most_relevant_excerpt(full_text, generated_text, min_words=25, max_words=85)
        
        strata.append({
            "id": f"source_{i}",
            "text": full_text,
            "excerpt": excerpt_info["excerpt"],
            "excerpt_relevance": excerpt_info["relevance"],
            "excerpt_explanation": excerpt_info["explanation"],
            "ref": f"{m['metadata'].get('author', 'Fisker')} ({m['metadata'].get('year', 'Arkiv')})",
            "year": m["metadata"].get("year"),
            "type": m["metadata"].get("__ns", "primary").upper(),
            "score": m.get("score", 0.0)
        })

    genealogy = analyze_source_relations(strata, generated_text, user_prompt)
    visuals = extract_visuals(generated_text)
    
    epistemic_level = genealogy.get("epistemic_levels", {}).get("primary_level", "FRIKSJON")
    if visuals:
        state = "VISUEL_AKKUMULERING"
    elif epistemic_level == "ARKIVFAKTA":
        state = "ARKIV-DIREKTE"
    elif epistemic_level == "ARKIV-NÆR TOLKNING":
        state = "TOLKNING"
    else:
        state = "SYNTESE"
    
    return {
        "text": generated_text,
        "strata": strata,
        "visuals": visuals,
        "state": state,
        "intensity": min(len(strata)/6, 1.0),
        "genealogy": genealogy
    }

@app.get("/tts")
async def get_tts(text: str):
    if tts is None:
        return JSONResponse({"error": "TTS utilgjengelig"}, 500)
    out = tts(text)
    sf.write("static/tts_output.wav", out["audio"], out["sampling_rate"])
    return FileResponse("static/tts_output.wav", media_type="audio/wav")

@app.get("/", response_class=HTMLResponse)
async def root_view():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse(content="<h1>static/index.html ikke funnet</h1>")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)