import os
import json
import re
import torch
import time
import numpy as np
import soundfile as sf
from datetime import datetime
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

# --- RAG OG SØKE-LOGIKK (Gjenopprettet fra original) ---
def normalize_orthography(txt: str) -> str:
    if not txt: return txt
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    txt = re.sub(r"[ \t]*\n[ \t]*", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def _fetch_ns(ns: str, qvec, top_k: int):
    try:
        res = pinecone_index.query(vector=qvec, top_k=top_k, include_metadata=True, namespace=ns)
        matches = res.get("matches", []) or []
        for m in matches:
            m["metadata"]["__ns"] = ns
        return matches
    except Exception as e:
        print(f"⚠️ Feil i namespace '{ns}': {e}")
        return []

def pinecone_search_logic(user_prompt: str, k: int = 8):
    if not (pinecone_index and embedder): return []
    
    print(f"\n🧠 Analyserer diskurs: '{user_prompt}'")
    qvec = embedder.encode(user_prompt).tolist()
    namespaces = ["primary", "secondary", "quotes"]
    weights = {"primary": 1.0, "secondary": 1.3, "quotes": 0.5}

    pool = []
    for ns in namespaces:
        pool += _fetch_ns(ns, qvec, top_k=k)

    if not pool: return []

    filtered = []
    for m in pool:
        md = m.get("metadata") or {}
        raw = (md.get("text") or "").strip()
        txt = normalize_orthography(raw)
        txt = clean_text(txt)
        
        if not txt or len(txt.split()) < 8 or len(txt) > 800: continue
        
        base_weight = weights.get(md.get("__ns"), 1.0)
        # Vekting basert på OCR-kvalitet og språk
        if (txt.count("ä") + txt.count("ö")) > 8 and (txt.count("æ") + txt.count("ø")) < 2:
            base_weight *= 0.6
        if len(re.findall(r"[^a-zA-ZæøåÆØÅ0-9.,:;?!()\-\s]", txt)) > 15:
            base_weight *= 0.7
            
        m["metadata"]["text"] = txt
        m["weight"] = base_weight
        filtered.append(m)

    if not filtered: filtered = pool[:4]

    # Reranking
    if reranker is not None:
        pairs = [(user_prompt, x["metadata"]["text"]) for x in filtered]
        scores = reranker.predict(pairs)
        filtered = [x for _, x in sorted(zip(scores, filtered), key=lambda z: z[0], reverse=True)]

    return filtered[:6]

# --- API ENDEPUNKTER ---
@app.post("/chat")
async def api_chat(req: Request):
    data = await req.json()
    user_prompt = data.get("message", "")
    
    # 1. RAG Logikk
    filtered_matches = pinecone_search_logic(user_prompt)
    context = "\n---\n".join([m["metadata"]["text"] for m in filtered_matches])
    
    # Bio-kontekst for system-prompt
    bio_context = ""
    for m in filtered_matches:
        if m["metadata"].get("__ns") == "secondary":
            bio_context = m["metadata"]["text"]
            break

    # 2. Formater 'Strata' for Diskursmaskinen
    strata = []
    for i, m in enumerate(filtered_matches):
        md = m["metadata"]
        strata.append({
            "text": md.get("text", ""),
            "ref": f"{md.get('author', 'Fisker')} ({md.get('year', 'Arkiv')})",
            "year": md.get("year"),
            "type": md.get("__ns", "primary").upper()
        })

    # 3. Den fulle system-prompten (gjenopprettet)
    system_prompt = (
        "Du er Kay Fisker (1893–1965), dansk arkitekt og professor. "
        "Du svarer som deg selv, i nøgternt dansk fagsprog præget af præcision og disciplin. "
        "Skriv i korte, tydelige sætninger – højst fem til seks per svar. "
        "Dine svar skal handle om arkitektur, undervisning og bygningers samfundsmæssige rolle.\n\n"
        f"Kontekst fra Martin Søberg:\n{bio_context}\n\n"
        f"Timeline:\n{FISKER_TIMELINE}\n\n"
        "Svar konkret basert på materialet. Hvis materialet ikke dækker spørgsmålet, si kort at det ikke omtales."
    )
    
    full_prompt = f"System: {system_prompt}\n\nArkivmateriale:\n{context}\n\nBruger: {user_prompt}\n\nSvar:"
    
    result = pipe(
        full_prompt, 
        max_new_tokens=160, 
        temperature=0.2, 
        top_p=0.8,
        repetition_penalty=1.2,
        do_sample=False
    )
    generated_text = result[0]["generated_text"].split("Svar:")[-1].strip()

    return {
        "text": generated_text,
        "strata": strata,
        "state": "OVERMETNING" if len(strata) > 4 else "FRIKSJON" if len(strata) > 0 else "SEDIMENTERING",
        "intensity": len(strata) / 6
    }

@app.get("/tts")
async def get_tts(text: str):
    if tts is None: return JSONResponse({"error": "TTS utilgjengelig"}, status_code=500)
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