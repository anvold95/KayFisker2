"""
KAY FISKER LANGUAGE MODEL v10.15
================================
KOMBINERER:
- v10.14 arkitektur: Gemini som assistent, LoRA ser kilder + hint
- v10.6 prompt: Epistemiske niveauer, kildeprioritet, Fisker-stemme

Begge modeller ser kildene, Gemini gir fokus-hint,
men LoRA får strukturert prompt for syntese og vurdering.
"""

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

# v10.13: Gemini for faktaekstraksjon
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai ikke installeret - kører uden Gemini")

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

# v10.13: Gemini konfiguration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_model = None
if GEMINI_AVAILABLE and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Gemini 2.0 Flash konfigurert for faktaekstraksjon")
    except Exception as e:
        print(f"⚠️ Gemini konfiguration fejlede: {e}")
else:
    print("⚠️ Gemini ikke tilgjengelig - kjører kun med Mistral+LoRA")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "kay-fisker-arkiv")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768-dim (matcher Gemini-arkiv)

# =====================================================
# v10.13: TO-TRINNS ARKITEKTUR
# Trinn 1: Gemini for fakta | Trinn 2: LoRA for stemme
# Fallback til kun LoRA hvis Gemini ikke er tilgjengelig
# =====================================================
USE_MIXTRAL = os.environ.get("USE_MIXTRAL", "false").lower() == "true"

if USE_MIXTRAL:
    BASE_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    print("🧠 Konfigurert for Mixtral 8x7B (bedre reasoning, ingen LoRA)")
else:
    BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    print("🧠 Konfigurert for Mistral 7B + LoRA")

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
}

# --- FOUCAULDIANSKE NØKKELORD ---
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
    
    LORA = "anvold/fisker-lora-clean"
    
    if USE_MIXTRAL:
        print(f"🧩 Indlæser Mixtral 8x7B (uden LoRA) …")
    else:
        print(f"🧩 Indlæser Mistral 7B + LoRA-adapter …")
    
    # LoRA adapter fix (kun for Mistral 7B)
    if not USE_MIXTRAL:
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

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if USE_MIXTRAL:
        model = base_model
        print("✅ Mixtral 8x7B lastet (ren base-modell)")
    else:
        model = PeftModel.from_pretrained(base_model, LORA)
        model = model.merge_and_unload()
        print("✅ Mistral 7B + LoRA merged")
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    print("🎙️ Indlæser TTS …")
    try:
        from transformers import pipeline as hf_pipeline_tool
        tts = hf_pipeline_tool("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
        print("✅ TTS klar.")
    except Exception as e:
        print(f"⚠️ TTS fejlede: {e}")

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

def fix_common_ocr_errors(txt: str) -> str:
    """
    Fikser vanlige OCR-feil i arkivtekst.
    Kjøres på kildetekst FØR den sendes til modellen.
    """
    if not txt:
        return txt
    
    # 1. Årstall: "192829" → "1928-29", "193539" → "1935-39"
    def fix_year_range(match):
        digits = match.group(0)
        if len(digits) == 6:
            return f"{digits[:4]}-{digits[4:]}"
        elif len(digits) == 8:
            return f"{digits[:4]}-{digits[4:]}"
        return digits
    
    txt = re.sub(r'\b(19\d{4})\b', fix_year_range, txt)
    txt = re.sub(r'\b(19\d{6})\b', fix_year_range, txt)
    txt = re.sub(r'\b(20\d{4})\b', fix_year_range, txt)
    
    # 2. "Opførelsesaar" varianter
    txt = re.sub(r'Opf[øo]relsesa+r\s*:', 'Opførelsesår:', txt, flags=re.IGNORECASE)
    
    # 3. Spacing-fixes
    txt = re.sub(r'\s+,', ',', txt)
    txt = re.sub(r'\s+\.', '.', txt)
    
    # 4. Sammenslåtte ord (vanlige OCR-feil)
    ocr_compounds = {
        r'fremmestkunstneren': 'fremmest kunstneren',
        r'fremmestkunst': 'fremmest kunst',
        r'arkitektoniskeudtryk': 'arkitektoniske udtryk',
        r'boligberunderordne': 'boligber underordne',
        r'funktioneltgennemarbejdet': 'funktionelt gennemarbejdet',
        r'Corbusiersarbejde': 'Corbusiers arbejde',
        r'blivefr?ordringsles': 'blive fordringsløs',
        r'kampenfor': 'kampen for',
        r'forstaaelse': 'forståelse',
    }
    for pattern, replacement in ocr_compounds.items():
        txt = re.sub(pattern, replacement, txt, flags=re.IGNORECASE)
    
    return txt

def enhance_query(user_prompt: str) -> str:
    """
    MINIMAL query expansion v10.7
    - Kun konkrete værker får expansion
    - Alt annet: trust embedder
    """
    prompt_lower = user_prompt.lower()
    
    # KUN konkrete byggværker - ingen andre patterns
    works_map = {
        "vestersøhus": "Vestersøhus Vester Søgade 1935 Kay Fisker boligblokk",
        "dronningegården": "Dronningegården Dronningens Tværgade 1943 Kay Fisker",
        "vigerslev": "Vigerslev Allé Ivar Bentsen rækkehuse",
        "aarhus": "Aarhus Universitet C.F. Møller Kay Fisker hovedbygning",
        "hornbæk": "Hornbækhus 1923 Kay Fisker landsted",
        "bakkehus": "Bakkehusene Ivar Bentsen Thorkild Henningsen",
        "gullfoss": "Gullfosshus Artillerivej Kay Fisker",
    }
    
    for work, expansion in works_map.items():
        if work in prompt_lower:
            print(f"   🎯 Konkret værk detekteret: {work}")
            return expansion
    
    # Alt annet: Legg kun til "Kay Fisker" hvis mangler
    if "fisker" not in prompt_lower:
        return f"Kay Fisker {user_prompt}"
    
    return user_prompt  # INGEN expansion - trust embedder

def extract_temporal_context(user_prompt: str, timeline: str) -> str:
    """Henter relevante timeline-segmenter"""
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
    """Finner visuelle referanser"""
    found = []
    text_lower = text.lower()
    for key, data in VISUALS_DB.items():
        if key in text_lower:
            found.append({"keyword": key, "url": data["url"], "type": data["type"]})
    return found

# --- SEGMENT-EKSTRAKSJON ---
def extract_most_relevant_excerpt(source_text: str, response_text: str, min_words: int = 20, max_words: int = 80) -> dict:
    """Finner mest relevante utdrag fra kilden"""
    if not embedder or not source_text or not response_text:
        return {
            "excerpt": source_text[:200] if len(source_text) > 200 else source_text,
            "relevance": 0.0,
            "method": "fallback",
            "explanation": "Ingen semantisk analyse tilgængelig"
        }
    
    sentences = re.split(r'(?<=[.!?])\s+', source_text)
    if not sentences or len(sentences) < 3:
        return {
            "excerpt": source_text[:200],
            "relevance": 0.0,
            "method": "fallback",
            "explanation": "For kort kildetekst"
        }
    
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
            "explanation": "Ingen passende vinduer funnet"
        }
    
    response_emb = embedder.encode(response_text, convert_to_tensor=True)
    window_texts = [w["text"] for w in windows]
    window_embs = embedder.encode(window_texts, convert_to_tensor=True)
    similarities = util.cos_sim(response_emb, window_embs)[0].cpu().numpy()
    
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    
    if best_score > 0.65:
        explanation = "Høy semantisk overlapp"
    elif best_score > 0.50:
        explanation = "Moderat overlapp"
    elif best_score > 0.35:
        explanation = "Kontekstuell bakgrunn"
    else:
        explanation = "Minimal overlapp"
    
    return {
        "excerpt": windows[best_idx]["text"],
        "relevance": best_score,
        "method": "semantic_matching",
        "explanation": explanation
    }

def find_attributable_segments(source_text: str, response_text: str, threshold: float = 0.50) -> list:
    """Finner segmenter som bidrar til responsen"""
    if not embedder:
        return []

    source_words = source_text.split()
    if len(source_words) < 10:
        return []

    segments = []
    window_size = 20
    step = 8

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
        if sim > threshold:
            attributed.append({
                "text": segments[idx]["text"],
                "similarity": float(sim),
                "position": segments[idx]["start_idx"],
                "confidence": "high" if sim > 0.70 else "medium"
            })
    
    attributed.sort(key=lambda x: x["similarity"], reverse=True)
    return attributed[:2]

# --- GENEALOGISK ANALYSE ---

def detect_discursive_shifts(strata: list) -> dict:
    """
    Analyserer kilder over tid - viser tidslinje og kun ægte diskursive skift.
    Et ægte skift kræver: samme emne, men ændret holdning/terminologi.
    """
    if not embedder or len(strata) < 2:
        return {"timeline": [], "thematic_shifts": [], "source_overview": []}
    
    sorted_strata = sorted([s for s in strata if s.get("year")], key=lambda x: int(x["year"]))
    
    if len(sorted_strata) < 2:
        return {"timeline": [], "thematic_shifts": [], "source_overview": []}
    
    # 1. TIDSLINJE: Simpel oversigt over kilder over tid (uden "brud"-terminologi)
    timeline = []
    for s in sorted_strata:
        timeline.append({
            "year": s["year"],
            "source": s["ref"],
            "title": s.get("title", "Ukendt"),
            "type": "PRIMÆR" if "Kay Fisker" in s["ref"] and int(s["year"]) <= 1965 else "SEKUNDÆR"
        })
    
    # 2. TEMATISKE SKIFT: Kun når samme emneord optræder med forskellige kontekster
    # Dette er mere meningsfuldt end blot at sammenligne alle tekster
    thematic_shifts = []
    
    # Find nøgleord der optræder i flere kilder
    all_texts = [(s["year"], s["text"].lower(), s["ref"]) for s in sorted_strata]
    
    # Arkitektur-relevante termer at spore
    key_terms = ["rækkehus", "etage", "funktionel", "modernisme", "tradition", 
                 "bolig", "arbejder", "social", "form", "materiale"]
    
    for term in key_terms:
        occurrences = []
        for year, text, ref in all_texts:
            if term in text:
                # Find kontekst omkring termen
                idx = text.find(term)
                context = text[max(0, idx-50):min(len(text), idx+50)]
                occurrences.append({
                    "year": year,
                    "source": ref,
                    "context": context.strip()
                })
        
        # Kun interessant hvis termen optræder i flere år
        if len(occurrences) >= 2:
            years = [o["year"] for o in occurrences]
            if len(set(years)) >= 2:  # Forskellige år
                thematic_shifts.append({
                    "term": term,
                    "occurrences": occurrences,
                    "span": f"{min(years)}-{max(years)}"
                })
    
    # 3. KILDEOVERSIGT: Grupperet efter periode
    source_overview = {
        "total": len(sorted_strata),
        "span": f"{sorted_strata[0]['year']}-{sorted_strata[-1]['year']}" if sorted_strata else None,
        "primary_count": len([s for s in sorted_strata if int(s["year"]) <= 1965]),
        "secondary_count": len([s for s in sorted_strata if int(s["year"]) > 1965])
    }
    
    return {
        "timeline": timeline,
        "thematic_shifts": thematic_shifts[:5],  # Max 5 mest relevante
        "source_overview": source_overview,
        # Behold for bagudkompatibilitet, men tom
        "shifts": [],
        "periods": []
    }

def trace_concept_genealogy(strata: list, concept: str) -> dict:
    """Sporer begrepets genealogi"""
    if not strata:
        return {"concept": concept, "occurrences": [], "semantic_drift": []}
    
    occurrences = []
    
    for s in strata:
        text_lower = s["text"].lower()
        if concept.lower() in text_lower:
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
                    break
    
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
    """Analyserer makt/kunnskaps-neksus"""
    authority_markers = {
        "citations": [],
        "institutional_references": [],
        "authority_hierarchy": {}
    }
    
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
    
    type_counts = Counter([s["type"] for s in strata])
    authority_markers["authority_hierarchy"] = {
        "PRIMARY": type_counts.get("PRIMARY", 0),
        "SECONDARY": type_counts.get("SECONDARY", 0),
        "QUOTES": type_counts.get("QUOTES", 0)
    }
    
    # v10.11: Mer forståelig terminologi
    primary_count = type_counts.get("PRIMARY", 0)
    quotes_count = type_counts.get("QUOTES", 0)
    
    if primary_count > quotes_count:
        dominant = "FISKERS EGNE TEKSTER"
        dominant_explanation = f"Svaret hviler primært på {primary_count} av Fiskers egne tekster"
    elif quotes_count > primary_count:
        dominant = "SITATER OG REFERANSER"
        dominant_explanation = f"Svaret hviler primært på {quotes_count} sitater fra andre kilder"
    else:
        dominant = "BLANDET"
        dominant_explanation = "Svaret kombinerer Fiskers tekster med eksterne referanser"
    
    return {
        "authority_citations": authority_markers["citations"],
        "hierarchy": authority_markers["authority_hierarchy"],
        "dominant_discourse": dominant,
        "dominant_explanation": dominant_explanation,  # NY
        "total_authority_markers": len(authority_markers["citations"])
    }

# --- EPISTEMISK ANALYSE ---

def analyze_source_relations(strata: list, response_text: str, query: str) -> dict:
    """Epistemisk analyse"""
    if not embedder:
        return {}

    analysis = {
        "source_attributions": [],
        "source_comparisons": [],
        "query_source_relations": [],
        "response_grounding": {},
        "epistemic_levels": {}
    }

    total_attributed = 0
    for s in strata:
        attributed = find_attributable_segments(s["text"], response_text, threshold=0.60)
        if attributed:
            avg_similarity = sum(seg["similarity"] for seg in attributed) / len(attributed)
            
            if avg_similarity > 0.68:
                level = "ARKIVFAKTA"
            elif avg_similarity > 0.55:
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

    query_emb = embedder.encode(query, convert_to_tensor=True)
    for s in strata:
        rel = util.cos_sim(query_emb, embedder.encode(s["text"], convert_to_tensor=True))[0][0].item()
        analysis["query_source_relations"].append({
            "source_ref": s["ref"],
            "query_relevance": float(rel),
            "relevance_category": "HØY" if rel > 0.6 else "MEDIUM" if rel > 0.4 else "LAV"
        })

    resp_emb = embedder.encode(response_text, convert_to_tensor=True)
    comb_emb = embedder.encode(" ".join([s["text"] for s in strata]), convert_to_tensor=True)
    g_score = util.cos_sim(resp_emb, comb_emb)[0][0].item()
    
    analysis["response_grounding"] = {
        "score": float(g_score),
        "assessment": "SOLID" if g_score > 0.7 else "MODERAT" if g_score > 0.5 else "SVAK",
        "warning": None if g_score > 0.5 else "Begrenset kildes støtte"
    }
    
    if analysis["source_attributions"]:
        avg_attribution = sum(a["attribution_strength"] for a in analysis["source_attributions"]) / len(analysis["source_attributions"])
        high_conf_count = len([a for a in analysis["source_attributions"] if a["attribution_strength"] > 0.65])
        
        if avg_attribution > 0.70 and high_conf_count >= 2:
            overall_level = "ARKIVFAKTA"
        elif avg_attribution > 0.58 and high_conf_count >= 1:
            overall_level = "ARKIVNÆR FORTOLKNING"
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
            "warning": "Ingen direkte kildeattribusjon"
        }

    return analysis

def perform_full_genealogical_analysis(strata: list, response_text: str, query: str) -> dict:
    """Kombinert analyse"""
    print("\n🔬 Udfører genealogisk-epistemisk analyse...")
    
    epistemic_analysis = analyze_source_relations(strata, response_text, query)
    
    genealogical_analysis = {
        "discursive_shifts": detect_discursive_shifts(strata),
        "concept_genealogies": {},
        "power_knowledge": analyze_power_knowledge_nexus(strata),
        "temporal_distribution": {},
        "discontinuities": []
    }
    
    for concept in GENEALOGICAL_CONCEPTS:
        if any(concept.lower() in s["text"].lower() for s in strata):
            genealogical_analysis["concept_genealogies"][concept] = trace_concept_genealogy(strata, concept)
    
    years = [int(s["year"]) for s in strata if s.get("year")]
    if years:
        genealogical_analysis["temporal_distribution"] = {
            "earliest": min(years),
            "latest": max(years),
            "span": max(years) - min(years),
            "median": int(np.median(years)),
            "decade_distribution": dict(Counter([y // 10 * 10 for y in years]))
        }
    
    # Ny struktur: timeline og thematic_shifts i stedet for "shifts/brud"
    discursive = genealogical_analysis["discursive_shifts"]
    if discursive.get("thematic_shifts"):
        genealogical_analysis["thematic_continuities"] = [{
            "term": ts["term"],
            "span": ts["span"],
            "occurrences": len(ts["occurrences"])
        } for ts in discursive["thematic_shifts"]]
    
    combined = {
        **epistemic_analysis,
        **genealogical_analysis
    }
    
    print(f"   ✅ Epistemisk niveau: {epistemic_analysis['epistemic_levels'].get('primary_level', 'N/A')}")
    print(f"   ✅ {len(genealogical_analysis['concept_genealogies'])} begrebsgeneaologier")
    timeline_count = len(discursive.get("timeline", []))
    thematic_count = len(discursive.get("thematic_shifts", []))
    print(f"   ✅ Tidslinje: {timeline_count} kilder, {thematic_count} tematiske spor")
    
    return combined

# --- RAG-LOGIKK ---

def _fetch_ns(ns: str, qvec, top_k: int):
    """Henter matches fra namespace"""
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

def pinecone_search_logic(user_prompt: str, total_results: int = 8):
    """
    RAG-søk med PRIMARY-first strategi v10.10
    """
    if not (pinecone_index and embedder):
        print("⚠️ Pinecone eller embedder ikke tilgjengelig")
        return []
    
    print(f"\n🔍 Analyserer: '{user_prompt}'")
    enhanced = enhance_query(user_prompt)
    print(f"   ↳ Ekspandert til: '{enhanced}'")
    qvec = embedder.encode(enhanced).tolist()
    
    # Hent fra alle namespaces
    ns_quotas = {"primary": 10, "secondary": 4, "quotes": 3}
    pool = defaultdict(list)
    
    for ns, quota in ns_quotas.items():
        pool[ns] = _fetch_ns(ns, qvec, top_k=quota)
        print(f"   ↳ {ns}: {len(pool[ns])} matches")

    # Clean OCR
    cleaned_pool = defaultdict(list)
    for ns, matches in pool.items():
        for m in matches:
            md = m.get("metadata") or {}
            txt = md.get("text") or ""
            txt = normalize_orthography(txt)
            txt = fix_common_ocr_errors(txt)
            txt = clean_text(txt)
            if len(txt.split()) > 8:
                md["text"] = txt
                cleaned_pool[ns].append(m)

    # Rerank
    if reranker:
        print("   🔄 Reranking...")
        for ns in cleaned_pool:
            if not cleaned_pool[ns]:
                continue
            pairs = [(user_prompt, m["metadata"]["text"]) for m in cleaned_pool[ns]]
            scores = reranker.predict(pairs)
            cleaned_pool[ns] = [x for _, x in sorted(zip(scores, cleaned_pool[ns]), key=lambda z: z[0], reverse=True)]

    # PRIMARY-FIRST SELECTION
    final_selection = []
    
    # Start med 3 PRIMARY
    for _ in range(3):
        if cleaned_pool["primary"]:
            final_selection.append(cleaned_pool["primary"].pop(0))
    
    # Så 1 SECONDARY (kontekst)
    if cleaned_pool["secondary"]:
        final_selection.append(cleaned_pool["secondary"].pop(0))
    
    # Så 1 QUOTES (hvis tilgjengelig)
    if cleaned_pool["quotes"]:
        final_selection.append(cleaned_pool["quotes"].pop(0))
    
    # Fylle opp resten: PRIMARY først
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
    
    print(f"   ✅ Returnerer {len(final_selection)} kilder (PRIMARY-first)")
    
    for i, m in enumerate(final_selection):
        md = m["metadata"]
        title = md.get('title', md.get('__ns', '?').upper())
        year = md.get('year', '?')
        print(f"      [{i+1}] {title} ({year}) | {md.get('text', '')[:50]}...")
    
    return final_selection

# --- API ENDEPUNKT ---

@app.post("/chat")
async def api_chat(req: Request):
    data = await req.json()
    user_prompt = data.get("message", "")
    
    if len(user_prompt.strip()) < 3:
        return JSONResponse({"error": "Spørsmål for kort"}, 400)
    
    matches = pinecone_search_logic(user_prompt, total_results=8)
    
    if not matches:
        return JSONResponse({
            "error": "Ingen kilder funnet",
            "text": "Beklager, jeg kan ikke svare uden arkivet.",
            "strata": [],
            "visuals": [],
            "state": "FRAKOBLET",
            "intensity": 0.0,
            "genealogy": {},
            "version": "v10.15"
        }, 200)
    
    # =====================================================
    # v10.14: GEMINI SOM ASSISTENT, IKKE PORTVAKT
    # Begge modeller ser kildene - Gemini hjælper med fokus
    # =====================================================
    
    # Bygg kilde-kontekst
    source_lines = []
    for i, m in enumerate(matches, 1):
        text = m["metadata"].get("text", "")[:600]
        year = m["metadata"].get("year", "?")
        title = m["metadata"].get("title", "Arkiv")
        author = m["metadata"].get("author", "Kay Fisker")
        
        try:
            year_int = int(year) if year != "?" else 1950
        except:
            year_int = 1950
            
        if year_int > 1965 or author != "Kay Fisker":
            source_prefix = f"[SEKUNDÆRKILDE - '{title}' av {author}, {year}]"
        else:
            source_prefix = f"[PRIMÆRKILDE - '{title}', {year}]"
        
        source_lines.append(f"{source_prefix}\n{text}")
    
    context = "\n\n".join(source_lines)
    
    # =====================================================
    # TRINN 1: GEMINI LAGER FAKTA-HINT (valgfrit hjælpemiddel)
    # =====================================================
    gemini_summary = ""
    if gemini_model:
        try:
            gemini_prompt = f"""Læs disse kilder og find Kay Fiskers VURDERINGER og HOLDNINGER relevant for spørgsmålet.

KILDER:
{context}

SPØRGSMÅL: {user_prompt}

Svar på dansk. List 3-5 punkter. Fokusér på:
- Fiskers personlige vurderinger ("forekommer mig", "det fineste exempel", "jeg finder")
- Konkrete sammenligninger han laver mellem bygninger/arkitekter
- Hans sociale/ideologiske synspunkter på arkitektur
- Specifikke bygninger han nævner med årstal

List IKKE tekniske detaljer som etager, materialer, priser medmindre de er del af en vurdering."""

            response = gemini_model.generate_content(gemini_prompt)
            gemini_summary = response.text.strip()
            print(f"   ✅ Gemini-hint: {gemini_summary[:100]}...")
        except Exception as e:
            print(f"   ⚠️ Gemini fejlede (fortsætter uden): {e}")
            gemini_summary = ""
    
    # =====================================================
    # TRINN 2: MISTRAL+LORA - v10.15 KOMBINERT PROMPT
    # Arkitektur fra v10.14 + struktur fra v10.6
    # =====================================================
    
    # Bygg hint-sektion hvis Gemini gav noget
    hint_section = ""
    if gemini_summary:
        hint_section = f"""
### NØGLEPUNKTER (fra kilderne) ###
{gemini_summary}
### SLUT NØGLEPUNKTER ###
"""
    
    lora_prompt = f"""Du er Kay Fisker (1893–1965), dansk arkitekt og professor.
Du svarer på dansk baseret på kildematerialet.

### ARKIVKILDER ###
{context}
### SLUT KILDER ###
{hint_section}
KILDEPRIORITET:
1. PRIMÆRE KILDER (dine egne skrifter før 1965) - brug disse FØRST
2. Sekundære kilder (skrevet OM dig efter 1965) - kun for kontekst
3. Reformulér ALDRIG sekundære kilder som dine egne udsagn

EPISTEMISKE NIVEAUER:
• ARKIVFAKTA: Citér/parafraser fra dine tekster eksplicit
• ARKIV-NÆR: Kombiner kilder, markér det ("Mine skrifter viser...")
• KONTEKST: Når arkivet er tavst, kontekstualisér ("Som arkitekt dengang...")

DIN STEMME:
- Giv personlige vurderinger: "forekommer mig", "jeg finder", "det har moret mig"
- Sammenlign konkrete eksempler: bygninger, arkitekter, årstal
- Vær faglig men med holdning - du er professor, ikke leksikon

FORBUDT:
- Opfinde data som ikke står i kilderne
- Generiske modernisme-klichéer
- Blande dansk og norsk
- Tale om dig selv i 3. person

Svar på dansk i 3-5 sætninger. Vær konkret og vurderende.

Spørgsmål: {user_prompt}

Kay Fisker:"""
    
    # Generér med LoRA
    result = pipe(
        lora_prompt,
        max_new_tokens=400,
        temperature=0.40,
        top_p=0.88,
        top_k=45,
        repetition_penalty=1.22,
        do_sample=True
    )
    
    generated_text = result[0]["generated_text"].split("Kay Fisker:")[-1].strip()
    generated_text = re.sub(r"^(System:|Spørgsmål:|###|HUSK:|KRITISK|FAKTA).*", "", generated_text, flags=re.MULTILINE).strip()
    generated_text = re.sub(r"\n+", " ", generated_text).strip()
    
    # Bygg strata med FULL metadata for frontend
    strata = []
    for i, m in enumerate(matches):
        full_text = m["metadata"].get("text", "")
        
        excerpt_info = await run_in_threadpool(
            extract_most_relevant_excerpt, 
            full_text, 
            generated_text, 
            25, 
            85
        )
        
        # v10.11: Mer metadata for Arbejdsbord
        strata.append({
            "id": f"source_{i}",
            "text": full_text,
            "excerpt": excerpt_info["excerpt"],
            "excerpt_relevance": excerpt_info["relevance"],
            "excerpt_explanation": excerpt_info["explanation"],
            "ref": f"{m['metadata'].get('author', 'Kay Fisker')} ({m['metadata'].get('year', 'Arkiv')})",
            "year": m["metadata"].get("year"),
            "type": m["metadata"].get("__ns", "primary").upper(),
            "score": m.get("score", 0.0),
            # NY METADATA for frontend
            "title": m["metadata"].get("title", "Ukjent kilde"),
            "author": m["metadata"].get("author", "Kay Fisker"),
            "source_type": m["metadata"].get("source_type", "arkiv"),
            "publication": m["metadata"].get("publication", None),
            "page": m["metadata"].get("page", None),
        })

    genealogy = await run_in_threadpool(
        perform_full_genealogical_analysis, 
        strata, 
        generated_text, 
        user_prompt
    )
    
    visuals = extract_visuals(generated_text)
    
    epistemic_level = genealogy.get("epistemic_levels", {}).get("primary_level", "FRIKTION")
    has_thematic_traces = len(genealogy.get("discursive_shifts", {}).get("thematic_shifts", [])) > 0
    
    if visuals:
        state = "VISUEL_AKKUMULERING"
    elif has_thematic_traces:
        state = "TEMATISK_SPORING"
    elif epistemic_level == "ARKIVFAKTA":
        state = "ARKIV-DIREKTE"
    elif epistemic_level == "ARKIVNÆR FORTOLKNING":
        state = "FORTOLKNING"
    else:
        state = "SYNTESE"
    
    return {
        "text": generated_text,
        "strata": strata,
        "visuals": visuals,
        "state": state,
        "intensity": min(len(strata)/8, 1.0),
        "genealogy": genealogy,
        "version": "v10.15",
        "model": "mixtral-8x7b" if USE_MIXTRAL else "mistral-7b-lora"
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