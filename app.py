import os
import json
import re
import torch
import time
import numpy as np
import soundfile as sf
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import hf_hub_download, login, whoami

# Forsøk å importere din egne tekstrenser
try:
    from text_cleaner import clean_text
except ImportError:
    def clean_text(text): return text.strip()

# --- MILJØKONFIGURASJON ---
os.environ["HF_HOME"] = "/app/cache"
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print(f"✅ Logget inn på Hugging Face som: {whoami().get('name')}")
    except Exception as e:
        print(f"⚠️ Login feil: {e}")

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
    if not txt: return txt
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    txt = re.sub(r"[ \t]*\n[ \t]*", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def enhance_query(user_prompt: str) -> str:
    prompt_lower = user_prompt.lower()
    
    # Nøkkelverk-mapping for bedre treffsikkerhet
    works_map = {
        "vestersøhus": "Vestersøhus Kay Fisker Vester Søgade København 1935",
        "dronningegården": "Dronningegården Kay Fisker Dronningens Tværgade",
        "gullfoss": "Gullfosshus Kay Fisker Artillerivej",
        "aarhus universitet": "Aarhus Universitet bygninger Kay Fisker C.F. Møller",
        "statsprøveanstalten": "Statsprøveanstalten Kay Fisker Amager Boulevard"
    }

    for work, expansion in works_map.items():
        if work in prompt_lower:
            return f"{expansion} {user_prompt}"

    is_biographical = any(w in prompt_lower for w in ["hvem", "hvad", "når", "hvor", "liv", "karriere"])
    is_theoretical = any(w in prompt_lower for w in ["hvorfor", "hvordan", "prinsipper", "teori", "tanker"])
    
    if is_biographical: return f"Kay Fisker biografi liv karriere {user_prompt}"
    elif is_theoretical: return f"arkitektonisk teori princip filosofi {user_prompt}"
    
    return f"arkitektur Kay Fisker {user_prompt}"

def extract_temporal_context(user_prompt: str, timeline: str) -> str:
    if not timeline: return ""
    periods = []
    for line in timeline.split("\n"):
        if match := re.match(r"(\d{4})[-–]?(\d{4})?: (.+)", line):
            start, end, event = match.groups()
            periods.append({"start": int(start), "end": int(end) if end else int(start), "event": event.strip()})
    
    keywords = set(user_prompt.lower().split())
    relevant = [p for p in periods if keywords & set(p["event"].lower().split())]
    return "\n".join([f"{p['start']}: {p['event']}" for p in relevant[:3]]) if relevant else ""

def diversify_sources(matches: list, max_per_year: int = 2, max_per_type: int = 3) -> list:
    year_count = defaultdict(int)
    type_count = defaultdict(int)
    diversified = []
    for m in matches:
        year = m["metadata"].get("year", "unknown")
        doc_type = m["metadata"].get("__ns", "unknown")
        if year_count[year] < max_per_year and type_count[doc_type] < max_per_type:
            diversified.append(m)
            year_count[year] += 1
            type_count[doc_type] += 1
        if len(diversified) >= 6: break
    return diversified

def extract_visuals(text: str) -> list:
    found = []
    text_lower = text.lower()
    for key, data in VISUALS_DB.items():
        if key in text_lower:
            found.append({"keyword": key, "url": data["url"], "type": data["type"]})
    return found

# --- GENEALOGISK ANALYSE ---
def find_attributable_segments(source_text: str, response_text: str) -> list:
    if not embedder: return []
    source_words = source_text.split()
    if len(source_words) < 8: return []
    
    segments = []
    window_size, step = 12, 6
    for i in range(0, len(source_words) - window_size + 1, step):
        segments.append({
            "text": " ".join(source_words[i:i+window_size]),
            "start_idx": i, "end_idx": i + window_size
        })
    if not segments: return []

    response_emb = embedder.encode(response_text, convert_to_tensor=True)
    segment_embs = embedder.encode([s["text"] for s in segments], convert_to_tensor=True)
    similarities = util.cos_sim(response_emb, segment_embs)[0].cpu().numpy()

    attributed = []
    for idx, sim in enumerate(similarities):
        if sim > 0.45:
            attributed.append({"text": segments[idx]["text"], "similarity": float(sim)})
    
    attributed.sort(key=lambda x: x["similarity"], reverse=True)
    return attributed[:3]

def analyze_source_relations(strata: list, response_text: str, query: str) -> dict:
    if not embedder: return {}
    analysis = {"source_attributions": [], "source_comparisons": [], "query_source_relations": [], "response_grounding": []}

    for s in strata:
        attributed = find_attributable_segments(s["text"], response_text)
        if attributed:
            analysis["source_attributions"].append({
                "source_id": s.get("id", "unknown"), "source_ref": s["ref"],
                "segments": attributed, "attribution_strength": sum(seg["similarity"] for seg in attributed) / len(attributed)
            })

    if len(strata) >= 2:
        source_texts = [s["text"] for s in strata]
        source_embs = embedder.encode(source_texts, convert_to_tensor=True)
        for i in range(len(strata)):
            for j in range(i + 1, len(strata)):
                sim = util.cos_sim(source_embs[i], source_embs[j])[0][0].item()
                if sim > 0.4:
                    analysis["source_comparisons"].append({
                        "source_a": strata[i]["ref"], "source_b": strata[j]["ref"],
                        "similarity": float(sim), "relation_type": "KONVERGENT" if sim > 0.7 else "RESONANT"
                    })

    query_emb = embedder.encode(query, convert_to_tensor=True)
    for s in strata:
        rel = util.cos_sim(query_emb, embedder.encode(s["text"], convert_to_tensor=True))[0][0].item()
        analysis["query_source_relations"].append({
            "source_ref": s["ref"], "query_relevance": float(rel),
            "relevance_category": "HØY" if rel > 0.6 else "MEDIUM" if rel > 0.4 else "LAV"
        })

    resp_emb = embedder.encode(response_text, convert_to_tensor=True)
    comb_emb = embedder.encode(" ".join([s["text"] for s in strata]), convert_to_tensor=True)
    g_score = util.cos_sim(resp_emb, comb_emb)[0][0].item()
    analysis["response_grounding"] = {
        "score": float(g_score),
        "assessment": "SOLID" if g_score > 0.7 else "MODERAT" if g_score > 0.5 else "SVAK"
    }
    return analysis

# --- RAG-LOGIKK (MIX) ---
def _fetch_ns(ns: str, qvec, top_k: int):
    try:
        res = pinecone_index.query(vector=qvec, top_k=top_k, include_metadata=True, namespace=ns)
        matches = res.get("matches", []) or []
        for m in matches: m["metadata"]["__ns"] = ns
        return matches
    except Exception: return []

def pinecone_search_logic(user_prompt: str, total_results: int = 6):
    if not (pinecone_index and embedder): return []
    print(f"\n🔍 Analyserer: '{user_prompt}'")
    enhanced = enhance_query(user_prompt)
    qvec = embedder.encode(enhanced).tolist()
    
    ns_quotas = {"primary": 6, "secondary": 4, "quotes": 3}
    pool = defaultdict(list)
    for ns, quota in ns_quotas.items():
        pool[ns] = _fetch_ns(ns, qvec, top_k=quota)

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
            if not cleaned_pool[ns]: continue
            pairs = [(user_prompt, m["metadata"]["text"]) for m in cleaned_pool[ns]]
            scores = reranker.predict(pairs)
            cleaned_pool[ns] = [x for _, x in sorted(zip(scores, cleaned_pool[ns]), key=lambda z: z[0], reverse=True)]

    final_selection = []
    if cleaned_pool["secondary"]: final_selection.append(cleaned_pool["secondary"].pop(0))
    if cleaned_pool["quotes"]: final_selection.append(cleaned_pool["quotes"].pop(0))
    
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
        if candidate not in final_selection: final_selection.append(candidate)
            
    print(f"   ✅ Returnerer {len(final_selection)} diverse kilder (Mix)")
    return final_selection

@app.post("/chat")
async def api_chat(req: Request):
    data = await req.json()
    user_prompt = data.get("message", "")
    
    if len(user_prompt.strip()) < 3: return JSONResponse({"error": "Spørsmål for kort"}, 400)
    
    matches = pinecone_search_logic(user_prompt, total_results=6)
    context = "\n---\n".join([m["metadata"]["text"] for m in matches])
    
    bio_context = ""
    for m in matches:
        if m["metadata"].get("__ns") == "secondary":
            bio_context = m["metadata"]["text"]
            break
    
    temporal_context = extract_temporal_context(user_prompt, FISKER_TIMELINE)
    strata = []
    for i, m in enumerate(matches):
        strata.append({
            "id": f"source_{i}", "text": m["metadata"].get("text", ""),
            "ref": f"{m['metadata'].get('author', 'Fisker')} ({m['metadata'].get('year', 'Arkiv')})",
            "year": m["metadata"].get("year"), "type": m["metadata"].get("__ns", "primary").upper()
        })

    # HYBRID EPISTEMISK PROMPT (DEN VIKTIGSTE ENDRINGEN)
    system_prompt = (
        "Du er arkitekten Kay Fisker (1893–1965). "
        "Når du svarer, skal du skille tydeligt mellem tre epistemiske niveauer: "
        "1. ARKIVFAKTA: Det der står direkte i kilderne. Her skal du være streng og henvise med [1], [2]. "
        "2. ARKIV-NÆR TOLKNING: Det man kan læse ud af sammenhængen i kilderne. Marker usikkerhed ('Det kan læses som...', 'Kilderne antyder...'). "
        "3. PRAKSISBASERET SYNTESE: Når arkivet er taust, kan du trække på bredere arkitekturhistorisk viden om tiden og funksjonalismen, "
        "MEN du skal markere dette tydeligt ('Set i lyset af tidens strømninger...', 'Min generelle praksis tilsiger...'). "
        "Du må aldrig opfinde konkrete årstal, adresser eller fakta som ikke findes. "
        "Hold svarene korte og i et nøgternt fagsprog."
    )
    
    if bio_context: system_prompt += f"\nBiografisk kontekst:\n{bio_context}\n"
    if temporal_context: system_prompt += f"\nRelevant tidsperiode:\n{temporal_context}\n"
    
    full_prompt = (
        f"System: {system_prompt}\n\n"
        f"### KILDEMATERIALE START ###\n{context}\n### KILDEMATERIALE SLUT ###\n\n"
        f"Spørgsmål: {user_prompt}\n\n"
        f"Kay Fisker:"
    )
    
    result = pipe(
        full_prompt, 
        max_new_tokens=180, # Litt mer rom for syntese
        temperature=0.2,    # Litt mer frihet enn 0.1, men kontrollert
        top_p=0.9, 
        repetition_penalty=1.15, 
        do_sample=True
    )
    
    generated_text = result[0]["generated_text"].split("Kay Fisker:")[-1].strip()
    generated_text = re.sub(r"^(System:|Spørgsmål:|###).*", "", generated_text, flags=re.MULTILINE).strip()
    
    genealogy = analyze_source_relations(strata, generated_text, user_prompt)
    visuals = extract_visuals(generated_text)

    state = "VISUEL_AKKUMULERING" if visuals else ("OVERMETNING" if len(strata) > 4 else "FRIKSJON")
    
    return {
        "text": generated_text, "strata": strata, "visuals": visuals,
        "state": state, "intensity": min(len(strata)/6, 1.0), "genealogy": genealogy
    }

@app.get("/tts")
async def get_tts(text: str):
    if tts is None: return JSONResponse({"error": "TTS utilgjengelig"}, 500)
    out = tts(text)
    sf.write("static/tts_output.wav", out["audio"], out["sampling_rate"])
    return FileResponse("static/tts_output.wav", media_type="audio/wav")

@app.get("/", response_class=HTMLResponse)
async def root_view():
    try:
        with open("static/index.html", "r") as f: return HTMLResponse(content=f.read())
    except: return HTMLResponse(content="<h1>static/index.html ikke funnet</h1>")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)