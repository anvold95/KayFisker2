import os
import json
import re
import torch
import time
import numpy as np
import soundfile as sf
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import hf_hub_download, login, whoami

# Forsøk å importere din egne tekstrenser
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

# --- MODELL-LASTING ---
def load_model_logic():
    global pipe, tts, tokenizer, embedder, reranker, pinecone_index
    
    BASE = "mistralai/Mistral-7B-Instruct-v0.3"
    LORA = "anvold/fisker-lora-clean"

    print("🧩 Laster base-modell + LoRA-adapter …")
    
    # Rens adapter_config.json for HF Space kompatibilitet
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
    """Utvider query basert på intensjonsdeteksjon"""
    prompt_lower = user_prompt.lower()
    
    # Detekter spørsmålstype
    is_biographical = any(w in prompt_lower for w in ["hvem", "hvad", "når", "hvor", "liv", "karriere"])
    is_theoretical = any(w in prompt_lower for w in ["hvorfor", "hvordan", "prinsipper", "teori", "tanker"])
    is_specific = any(w in prompt_lower for w in ["bolig", "monumental", "skole", "bygning", "projekt"])
    
    # Strategisk expansion
    if is_biographical:
        return f"Kay Fisker biografi liv karriere {user_prompt}"
    elif is_theoretical:
        return f"arkitektonisk teori princip filosofi {user_prompt}"
    elif is_specific:
        return user_prompt  # Behold spesifikk query
    
    return f"arkitektur Kay Fisker {user_prompt}"

def extract_temporal_context(user_prompt: str, timeline: str) -> str:
    """Henter relevante timeline-segmenter basert på query"""
    if not timeline:
        return ""
    
    # Parse timeline
    periods = []
    for line in timeline.split("\n"):
        if match := re.match(r"(\d{4})[-–]?(\d{4})?: (.+)", line):
            start, end, event = match.groups()
            periods.append({
                "start": int(start),
                "end": int(end) if end else int(start),
                "event": event.strip()
            })
    
    # Finn relevante perioder (keyword matching)
    keywords = set(user_prompt.lower().split())
    relevant = []
    
    for p in periods:
        event_words = set(p["event"].lower().split())
        if keywords & event_words:  # Set intersection
            relevant.append(p)
    
    if relevant:
        return "\n".join([
            f"{p['start']}-{p['end']}: {p['event']}" 
            for p in relevant[:3]
        ])
    
    return ""

def diversify_sources(matches: list, max_per_year: int = 2, max_per_type: int = 3) -> list:
    """
    Sørger for temporal og type-basert spredning av kilder
    
    Args:
        matches: Liste av Pinecone matches
        max_per_year: Maks antall kilder fra samme år
        max_per_type: Maks antall kilder fra samme namespace
    """
    year_count = defaultdict(int)
    type_count = defaultdict(int)
    diversified = []
    
    for m in matches:
        year = m["metadata"].get("year", "unknown")
        doc_type = m["metadata"].get("__ns", "unknown")
        
        # Sjekk om vi kan inkludere denne kilden
        if (year_count[year] < max_per_year and 
            type_count[doc_type] < max_per_type):
            diversified.append(m)
            year_count[year] += 1
            type_count[doc_type] += 1
        
        if len(diversified) >= 6:
            break
    
    # Hvis vi ikke fikk nok diverse kilder, fyll opp med beste matches
    if len(diversified) < 4:
        for m in matches:
            if m not in diversified:
                diversified.append(m)
            if len(diversified) >= 6:
                break
    
    return diversified

# --- RAG-LOGIKK MED FORBEDRINGER ---
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
    """
    Forbedret RAG-søk med:
    - Query enhancement
    - Strategisk namespace-fordeling
    - Temporal diversifisering
    - Reranking
    """
    if not (pinecone_index and embedder): 
        return []
    
    print(f"\n🔍 Analyserer: '{user_prompt}'")
    
    # 1. Enhance query for bedre semantic matching
    enhanced = enhance_query(user_prompt)
    print(f"   ↳ Utvida til: '{enhanced}'")
    qvec = embedder.encode(enhanced).tolist()
    
    # 2. Strategisk namespace-fordeling (ikke 8×3=24, men 6+3+3=12)
    ns_quotas = {
        "primary": 6,    # Primærkilder (bøker, artikler)
        "secondary": 3,  # Biografisk/kontekstuelt
        "quotes": 3      # Sitater
    }
    weights = {
        "primary": 1.0, 
        "secondary": 1.3,  # Biografisk får litt høyere vekt
        "quotes": 0.5
    }
    
    pool = []
    for ns, quota in ns_quotas.items():
        matches = _fetch_ns(ns, qvec, top_k=quota)
        for m in matches:
            m["weight"] = weights[ns]
        pool += matches
        print(f"   ↳ {ns}: {len(matches)} matches")
    
    if not pool:
        print("   ⚠️ Ingen matches funnet")
        return []
    
    # 3. Filter og clean (din opprinnelige logikk)
    filtered = []
    for m in pool:
        md = m.get("metadata") or {}
        raw = (md.get("text") or "").strip()
        txt = normalize_orthography(raw)
        txt = clean_text(txt)
        
        # Kvalitetskontroll
        if not txt or len(txt.split()) < 8 or len(txt) > 800:
            continue
        
        # OCR quality scoring
        base_weight = m["weight"]
        
        # Penaliser svensk OCR-feil
        if (txt.count("ä") + txt.count("ö")) > 8 and (txt.count("æ") + txt.count("ø")) < 2:
            base_weight *= 0.6
        
        # Penaliser for mange spesialtegn (OCR-støy)
        special_chars = len(re.findall(r"[^a-zA-ZæøåÆØÅ0-9.,:;?!()\-\s]", txt))
        if special_chars > 15:
            base_weight *= 0.7
        
        m["metadata"]["text"] = txt
        m["weight"] = base_weight
        filtered.append(m)
    
    if not filtered:
        print("   ⚠️ Ingen kilder overlevde filtrering")
        filtered = pool[:4]  # Fallback
    
    print(f"   ✓ {len(filtered)} kilder etter filtrering")
    
    # 4. Reranking med cross-encoder
    if reranker is not None and filtered:
        print("   🔄 Reranker...")
        pairs = [(user_prompt, x["metadata"]["text"]) for x in filtered]
        scores = reranker.predict(pairs)
        filtered = [x for _, x in sorted(zip(scores, filtered), key=lambda z: z[0], reverse=True)]
    
    # 5. NYTT: Diversifiser for temporal og type-spredning
    filtered = diversify_sources(filtered, max_per_year=2, max_per_type=3)
    
    print(f"   ✅ Returnerer {len(filtered)} diverse kilder")
    
    # Debug output
    for i, m in enumerate(filtered[:total_results]):
        md = m["metadata"]
        print(f"      [{i+1}] {md.get('__ns', '?').upper()} | {md.get('year', '?')} | {md.get('text', '')[:60]}...")
    
    return filtered[:total_results]

# --- API ENDEPUNKTER ---
@app.post("/chat")
async def api_chat(req: Request):
    data = await req.json()
    user_prompt = data.get("message", "")
    
    # 1. RAG Logikk med forbedringer
    filtered_matches = pinecone_search_logic(user_prompt, total_results=6)
    
    # 2. Bygg kontekst
    context = "\n---\n".join([m["metadata"]["text"] for m in filtered_matches])
    
    # Bio-kontekst fra secondary sources
    bio_context = ""
    for m in filtered_matches:
        if m["metadata"].get("__ns") == "secondary":
            bio_context = m["metadata"]["text"]
            break
    
    # Temporal kontekst fra timeline
    temporal_context = extract_temporal_context(user_prompt, FISKER_TIMELINE)
    
    # 3. Formater 'Strata' for Diskursmaskinen
    strata = []
    for i, m in enumerate(filtered_matches):
        md = m["metadata"]
        strata.append({
            "text": md.get("text", ""),
            "ref": f"{md.get('author', 'Fisker')} ({md.get('year', 'Arkiv')})",
            "year": md.get("year"),
            "type": md.get("__ns", "primary").upper()
        })
    
    # 4. Forbedret system-prompt
    system_prompt = (
        "Du er Kay Fisker (1893–1965), dansk arkitekt og professor ved Kunstakademiet. "
        "Du svarer som deg selv, i nøgternt dansk fagsprog præget af præcision og disciplin. "
        "Skriv i korte, tydelige sætninger – højst fem til seks per svar. "
        "Dine svar skal handle om arkitektur, undervisning og bygningers samfundsmæssige rolle.\n\n"
    )
    
    if bio_context:
        system_prompt += f"Biografisk kontekst:\n{bio_context}\n\n"
    
    if temporal_context:
        system_prompt += f"Relevant tidsperiode:\n{temporal_context}\n\n"
    elif FISKER_TIMELINE:
        # Fallback: inkluder hele timeline hvis ingen spesifikk periode ble funnet
        system_prompt += f"Karriere-timeline:\n{FISKER_TIMELINE[:500]}...\n\n"
    
    system_prompt += (
        "Svar konkret basert på arkivmaterialet nedenfor. "
        "Hvis materialet ikke dækker spørgsmålet direkte, si kort at det ikke omtales i arkivet, "
        "men du kan evt. gi et kort perspektiv basert på din generelle praksis."
    )
    
    full_prompt = (
        f"System: {system_prompt}\n\n"
        f"Arkivmateriale:\n{context}\n\n"
        f"Bruger: {user_prompt}\n\n"
        f"Kay Fisker:"
    )
    
    # 5. Generer respons
    result = pipe(
        full_prompt, 
        max_new_tokens=160, 
        temperature=0.25,  # Litt høyere for variasjon
        top_p=0.85,
        repetition_penalty=1.2,
        do_sample=True  # Aktivert for litt mer naturlighet
    )
    
    generated_text = result[0]["generated_text"].split("Kay Fisker:")[-1].strip()
    
    # Fjern eventuelle residual prompts
    generated_text = re.sub(r"^(System:|Bruger:|Arkivmateriale:).*", "", generated_text, flags=re.MULTILINE).strip()
    
    # Beregn diskursiv tilstand
    state = "OVERMETNING" if len(strata) > 4 else "FRIKSJON" if len(strata) > 0 else "SEDIMENTERING"
    intensity = min(len(strata) / 6, 1.0)
    
    return {
        "text": generated_text,
        "strata": strata,
        "state": state,
        "intensity": intensity
    }

@app.get("/tts")
async def get_tts(text: str):
    if tts is None: 
        return JSONResponse({"error": "TTS utilgjengelig"}, status_code=500)
    
    output = tts(text)
    audio_data = output.get("audio")
    sr = output.get("sampling_rate", 22050)
    audio_path = "static/tts_output.wav"
    sf.write(audio_path, audio_data, sr)
    return FileResponse(audio_path, media_type="audio/wav")

@app.get("/", response_class=HTMLResponse)
async def root_view():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception:
        return HTMLResponse(content="<h1>static/index.html ikke funnet</h1>")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)