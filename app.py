import os
import json
import re
import torch
import numpy as np
import soundfile as sf
from datetime import datetime
from collections import defaultdict, Counter
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import hf_hub_download, login, whoami
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

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

# --- FOUCAULDIANSKE NØKKELORD (Arkitektonisk Diskurs) ---
GENEALOGICAL_CONCEPTS = [
    "funksjonalisme", "funktionalism", "tradisjon", "tradition", "modernisme", 
    "klassisisme", "bolig", "bebyggelse", "bymæssig", "monumentalitet",
    "materialitet", "tektonik", "proportioner", "skala", "rumlig",
    "social", "samfund", "arbejder", "kollektiv", "privat",
    "hygiejne", "sundhed", "lys", "luft",
    "standardisering", "industrialisering", "håndværk", "præfabrikation"
]

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

# --- FORBEDRET SEGMENT-EKSTRAKSJON (fra v.9.9) ---
def extract_most_relevant_excerpt(source_text: str, response_text: str, min_words: int = 20, max_words: int = 80) -> dict:
    """
    Finner det mest relevante utdraget fra kilden basert på FAKTISK semantisk overlapp.
    Returnerer både utdrag, forklaring og relevans-score.
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

def find_attributable_segments(source_text: str, response_text: str, threshold: float = 0.60) -> list:
    """
    ENDA STRENGERE segment-matching (threshold 0.60) for å unngå false positives.
    Finner KUN segmenter som FAKTISK bidrar direkte til responsen.
    """
    if not embedder:
        return []

    source_words = source_text.split()
    if len(source_words) < 10:
        return []

    segments = []
    window_size = 20  # Økt til 20 for bedre kontekst
    step = 8  # Større step for mindre overlapp

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
        if sim > threshold:  # STRENGERE: 0.60 (var 0.50)
            attributed.append({
                "text": segments[idx]["text"],
                "similarity": float(sim),
                "position": segments[idx]["start_idx"],
                "confidence": "high" if sim > 0.70 else "medium"
            })
    
    # Kun top 2 segmenter (var 3)
    attributed.sort(key=lambda x: x["similarity"], reverse=True)
    return attributed[:2]

# --- FOUCAULDIANSK GENEALOGISK ANALYSE (fra v.10.0) ---

def detect_discursive_shifts(strata: list) -> dict:
    """
    Detekterer DISKURSIVE BRUDD og begrepsforskyninger over tid.
    Inspirert av Foucault's "Archaeology of Knowledge".
    """
    if not embedder or len(strata) < 3:
        return {"shifts": [], "periods": []}
    
    # Sorter kilder temporalt
    sorted_strata = sorted([s for s in strata if s.get("year")], key=lambda x: int(x["year"]))
    
    if len(sorted_strata) < 3:
        return {"shifts": [], "periods": []}
    
    # Embed alle tekster
    texts = [s["text"] for s in sorted_strata]
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    
    # Beregn diskursiv avstand mellom påfølgende perioder
    shifts = []
    for i in range(len(sorted_strata) - 1):
        distance = cosine(
            embeddings[i].cpu().numpy(),
            embeddings[i+1].cpu().numpy()
        )
        
        # DISKURSIVT BRUDD hvis avstand > 0.35
        if distance > 0.35:
            shifts.append({
                "year_from": sorted_strata[i]["year"],
                "year_to": sorted_strata[i+1]["year"],
                "distance": float(distance),
                "type": "MAJOR_SHIFT" if distance > 0.5 else "MINOR_SHIFT",
                "source_a": sorted_strata[i]["ref"],
                "source_b": sorted_strata[i+1]["ref"]
            })
    
    # Identifiser diskursive PERIODER (clustering)
    if len(embeddings) >= 3:
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings.cpu().numpy())
        
        periods = []
        for label in set(labels):
            if label == -1:  # Noise
                continue
            indices = [i for i, l in enumerate(labels) if l == label]
            period_strata = [sorted_strata[i] for i in indices]
            years = [int(s["year"]) for s in period_strata]
            
            periods.append({
                "period_id": int(label),
                "year_range": f"{min(years)}-{max(years)}",
                "source_count": len(period_strata),
                "sources": [s["ref"] for s in period_strata],
                "coherence": "HIGH"  # Innenfor samme cluster = høy koherens
            })
    else:
        periods = []
    
    return {
        "shifts": shifts,
        "periods": periods,
        "total_sources": len(sorted_strata),
        "temporal_span": f"{sorted_strata[0]['year']}-{sorted_strata[-1]['year']}" if sorted_strata else None
    }

def trace_concept_genealogy(strata: list, concept: str) -> dict:
    """
    Sporer en BEGREPETS GENEALOGI gjennom kildene.
    Hvordan endrer betydningen av et begrep seg over tid?
    """
    if not strata:
        return {"concept": concept, "occurrences": [], "semantic_drift": []}
    
    occurrences = []
    
    for s in strata:
        text_lower = s["text"].lower()
        if concept.lower() in text_lower:
            # Finn kontekst rundt begrepet (±50 ord)
            words = s["text"].split()
            for i, word in enumerate(words):
                if concept.lower() in word.lower():
                    start = max(0, i - 25)
                    end = min(len(words), i + 25)
                    context = " ".join(words[start:end])
                    
                    occurrences.append({
                        "year": s.get("year", "unknown"),
                        "source_ref": s["ref"],
                        "context": context,
                        "source_id": s["id"]
                    })
                    break  # Kun første forekomst per kilde
    
    # Beregn SEMANTISK DRIFT hvis vi har embedder
    semantic_drift = []
    if embedder and len(occurrences) >= 2:
        contexts = [o["context"] for o in occurrences]
        context_embs = embedder.encode(contexts, convert_to_tensor=True)
        
        for i in range(len(occurrences) - 1):
            drift = cosine(
                context_embs[i].cpu().numpy(),
                context_embs[i+1].cpu().numpy()
            )
            
            semantic_drift.append({
                "from_year": occurrences[i]["year"],
                "to_year": occurrences[i+1]["year"],
                "drift_score": float(drift),
                "interpretation": "STABLE" if drift < 0.2 else "SHIFTING" if drift < 0.4 else "RADICAL_CHANGE"
            })
    
    return {
        "concept": concept,
        "total_occurrences": len(occurrences),
        "occurrences": occurrences,
        "semantic_drift": semantic_drift,
        "periods_active": list(set([o["year"] for o in occurrences]))
    }

def analyze_power_knowledge_nexus(strata: list) -> dict:
    """
    Analyserer MAKT/KUNNSKAPS-NEKSUS i kildene.
    - Hvem autoriserer Fisker? (sitater, referanser)
    - Hvilke institusjoner legitimerer utsagnene?
    - Hierarki av kildetyper
    """
    authority_markers = {
        "citations": [],
        "institutional_references": [],
        "authority_hierarchy": {}
    }
    
    # Søk etter autoritetsfigurer/institusjoner
    authority_patterns = [
        (r"(Le Corbusier|Asplund|Wright|Gropius|Mies)", "ARCHITECT"),
        (r"(Akademiet|Kunstakademiet|universitet|skole)", "INSTITUTION"),
        (r"(regering|departement|bygningsråd|boligkommission)", "STATE"),
        (r"(tidsskrift|journal|publikation|artikel)", "PUBLICATION")
    ]
    
    for s in strata:
        for pattern, auth_type in authority_patterns:
            matches = re.findall(pattern, s["text"], re.IGNORECASE)
            for match in matches:
                authority_markers["citations"].append({
                    "entity": match if isinstance(match, str) else match[0],
                    "type": auth_type,
                    "source_year": s.get("year"),
                    "source_ref": s["ref"]
                })
    
    # Hierarki basert på kildetype
    type_counts = Counter([s["type"] for s in strata])
    authority_markers["authority_hierarchy"] = {
        "PRIMARY": type_counts.get("PRIMARY", 0),
        "SECONDARY": type_counts.get("SECONDARY", 0),
        "QUOTES": type_counts.get("QUOTES", 0)
    }
    
    # Identifiser dominerende diskurs
    if type_counts.get("PRIMARY", 0) > type_counts.get("QUOTES", 0):
        dominant = "EIGENMACHT"  # Fisker's egen autoritet
    else:
        dominant = "BORROWED_AUTHORITY"  # Baserer seg på andre
    
    return {
        "authority_citations": authority_markers["citations"],
        "hierarchy": authority_markers["authority_hierarchy"],
        "dominant_discourse": dominant,
        "total_authority_markers": len(authority_markers["citations"])
    }

# --- KOMBINERT ANALYSE (Epistemisk + Genealogisk) ---

def analyze_source_relations(strata: list, response_text: str, query: str) -> dict:
    """
    EPISTEMISK ANALYSE med strengere vurdering (fra v.9.9)
    """
    if not embedder:
        return {}

    analysis = {
        "source_attributions": [],
        "source_comparisons": [],
        "query_source_relations": [],
        "response_grounding": {},
        "epistemic_levels": {}
    }

    # 1. Attributions med ENDA STRENGERE threshold (0.60)
    total_attributed = 0
    for s in strata:
        attributed = find_attributable_segments(s["text"], response_text, threshold=0.60)  # Økt fra 0.55
        if attributed:
            avg_similarity = sum(seg["similarity"] for seg in attributed) / len(attributed)
            
            # STRENGERE epistemisk vurdering
            if avg_similarity > 0.72:  # Økt fra 0.70
                level = "ARKIVFAKTA"
            elif avg_similarity > 0.62:  # Økt fra 0.58
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

def perform_full_genealogical_analysis(strata: list, response_text: str, query: str) -> dict:
    """
    KOMBINERT: Epistemisk analyse + Foucauldiansk genealogisk analyse
    """
    print("\n🔬 Utfører fullstendig genealogisk-epistemisk analyse...")
    
    # EPISTEMISK ANALYSE (fra v.9.9)
    epistemic_analysis = analyze_source_relations(strata, response_text, query)
    
    # GENEALOGISK ANALYSE (fra v.10.0)
    genealogical_analysis = {
        "discursive_shifts": detect_discursive_shifts(strata),
        "concept_genealogies": {},
        "power_knowledge": analyze_power_knowledge_nexus(strata),
        "temporal_distribution": {},
        "discontinuities": []
    }
    
    # Spor ALLE genealogiske begreper som finnes i kildene
    for concept in GENEALOGICAL_CONCEPTS:
        if any(concept.lower() in s["text"].lower() for s in strata):
            genealogical_analysis["concept_genealogies"][concept] = trace_concept_genealogy(strata, concept)
    
    # Temporal fordeling
    years = [int(s["year"]) for s in strata if s.get("year")]
    if years:
        genealogical_analysis["temporal_distribution"] = {
            "earliest": min(years),
            "latest": max(years),
            "span": max(years) - min(years),
            "median": int(np.median(years)),
            "decade_distribution": dict(Counter([y // 10 * 10 for y in years]))
        }
    
    # Identifiser DISKONTINUITETER
    if genealogical_analysis["discursive_shifts"]["shifts"]:
        major_shifts = [s for s in genealogical_analysis["discursive_shifts"]["shifts"] if s["type"] == "MAJOR_SHIFT"]
        genealogical_analysis["discontinuities"] = [{
            "year": shift["year_to"],
            "description": f"Major discursive break between {shift['year_from']} and {shift['year_to']}",
            "magnitude": shift["distance"]
        } for shift in major_shifts]
    
    # KOMBINER begge analyser
    combined = {
        **epistemic_analysis,
        **genealogical_analysis
    }
    
    print(f"   ✅ Epistemisk nivå: {epistemic_analysis['epistemic_levels'].get('primary_level', 'N/A')}")
    print(f"   ✅ {len(genealogical_analysis['concept_genealogies'])} begrepsgeneaologier")
    print(f"   ✅ {len(genealogical_analysis['discursive_shifts']['shifts'])} diskursive skift")
    print(f"   ✅ {len(genealogical_analysis['power_knowledge']['authority_citations'])} autoritetmarkører")
    
    return combined

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
    
    # HYBRID EPISTEMISK PROMPT (forbedret for v.10.2)
    system_prompt = (
        "Du er den danske arkitekt Kay Fisker (1893–1965). "
        "Du svarer UDELUKKENDE på dansk – aldrig norsk, svensk eller andet sprog.\n\n"
        
        "SPROGLIGE KRAV:\n"
        "- Brug KUN dansk ortografi og grammatik\n"
        "- Dansk: 'jeg', 'arkitektur', 'bygning', 'bolig'\n"
        "- IKKE norsk: 'jeg', 'arkitektur', 'bygning', 'bolig' (selv om det ser likt ud)\n"
        "- ALDRIG: 'også', 'eller', 'hvordan' med norsk uttale/betydning\n\n"
        
        "STILISTISKE KRAV:\n"
        "- Tal som Kay Fisker fra 1930-50erne: høflig, præcis, faglig\n"
        "- Brug korte, klare sætninger uden omsvøb\n"
        "- Undgå moderne jargon eller akademisk posering\n\n"
        
        "TRE EPISTEMISKE NIVEAUER:\n\n"
        
        "NIVEAU 1 — ARKIVFAKTA (direkte kildebaseret):\n"
        "- Kun når du citerer eller parafraserer eksplicit fra kilderne\n"
        "- Eksempel: 'I Byplanproblemer (1933) skriver jeg, at...'\n\n"
        
        "NIVEAU 2 — ARKIV-NÆR TOLKNING (kildebaseret syntese):\n"
        "- Når du kombinerer flere arkivkilder til en tolkning\n"
        "- Markér tydeligt: 'Mine skrifter viser...', 'I denne periode fremgår det...'\n"
        "- Kun begreber og tanker som findes i materialet\n\n"
        
        "NIVEAU 3 — FAGLIG KONTEKSTUALISERING (arkitekturhistorisk ramme):\n"
        "- Når arkivet er tavst, men du kan placere emnet i datidens arkitekturdiskurs\n"
        "- ALTID markeret: 'Som arkitekt i min generation...', 'I lyset af tidens strømninger...'\n"
        "- ALDRIG fremstillet som dit eget eksplicitte udsagn\n\n"
        
        "ABSOLUT FORBUDT:\n"
        "- Opfinde konkrete data, årstal eller fakta som ikke står i kilderne\n"
        "- Umarkerede generaliseringer fremstillet som arkivfakta\n"
        "- Blande dansk og norsk sprog\n\n"
        
        "Svar kort (2-4 sætninger), præcist og i et nøgternt fagsprog."
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
        max_new_tokens=300,  # Økt fra 200 for å unngå avkutting
        temperature=0.28,    # Litt høyere for mer naturlig Kay Fisker-tone
        top_p=0.90,          # Litt lavere for mer konsistent dansk
        repetition_penalty=1.15,
        do_sample=True
    )
    
    generated_text = result[0]["generated_text"].split("Kay Fisker:")[-1].strip()
    generated_text = re.sub(r"^(System:|Spørgsmål:|###).*", "", generated_text, flags=re.MULTILINE).strip()
    generated_text = re.sub(r"\n+", " ", generated_text).strip()
    
    # Bygg strata MED relevante utdrag (FORBEDRET - kjører i threadpool)
    strata = []
    for i, m in enumerate(matches):
        full_text = m["metadata"].get("text", "")
        
        # Kjør tung embedding-operasjon i threadpool
        excerpt_info = await run_in_threadpool(
            extract_most_relevant_excerpt, 
            full_text, 
            generated_text, 
            25, 
            85
        )
        
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

    # FULLSTENDIG ANALYSE (Epistemisk + Genealogisk) - kjører i threadpool
    genealogy = await run_in_threadpool(
        perform_full_genealogical_analysis, 
        strata, 
        generated_text, 
        user_prompt
    )
    visuals = extract_visuals(generated_text)
    
    # Bestem state basert på analyse
    epistemic_level = genealogy.get("epistemic_levels", {}).get("primary_level", "FRIKSJON")
    has_major_shifts = len([s for s in genealogy.get("discursive_shifts", {}).get("shifts", []) if s.get("type") == "MAJOR_SHIFT"]) > 0
    
    if visuals:
        state = "VISUEL_AKKUMULERING"
    elif has_major_shifts:
        state = "DISKURSIVT_BRUDD"
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