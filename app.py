import os
os.environ["HF_HOME"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"


import os, json, re, torch, threading
from text_cleaner import clean_text
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline as hf_pipeline
from huggingface_hub import hf_hub_download, login, whoami
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone

# --------------------
# Last timeline
# --------------------
with open("timeline_fisker.txt", "r", encoding="utf-8") as f:
    FISKER_TIMELINE = f.read().strip()

# --------------------
# FastAPI app
# --------------------
app = FastAPI()

from fastapi.responses import HTMLResponse, FileResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

# Mount static *etterpå* på egen sti
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------
# Globale variabler
# --------------------
pipe = None
tts = None
tokenizer = None
pinecone_index = None
embedder = None
reranker = None

# --------------------
# Miljøvariabler
# --------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Logget inn på Hugging Face som:", whoami().get("name"))
    except Exception as e:
        print("⚠️ Kunne ikke logg\e inn på Hugging Face:", e)
else:
    print("⚠️ Ingen HF_TOKEN funnet – prøver uten innlogging.")

# --------------------
# Modell-last (asynkron)
# --------------------
def load_model():
    global pipe, tts, tokenizer, pinecone_index, embedder, reranker

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel

    BASE = "mistralai/Mistral-7B-Instruct-v0.3"
    LORA = "anvold/fisker-lora-clean"

    print("🧩 Laster base-modell + LoRA-adapter …")

    # Rens adapter-config
    adapter_path = hf_hub_download(LORA, "adapter_config.json")
    with open(adapter_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    allowed_keys = {
        "base_model_name_or_path", "bias", "inference_mode", "lora_alpha",
        "lora_dropout", "r", "target_modules", "task_type", "peft_type",
        "fan_in_fan_out", "use_rslora", "alpha_pattern", "rank_pattern"
    }
    for k in list(cfg.keys()):
        if k not in allowed_keys:
            cfg.pop(k)
    with open(adapter_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, LORA)
    model = model.merge_and_unload()
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print("🎙️ Laster TTS …")
    try:
        tts = hf_pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
        print("✅ TTS klar.")
    except Exception as e:
        print("⚠️ Kunne ikke laste TTS:", e)
        tts = None

    # Pinecone RAG
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    INDEX_NAME = os.environ.get("INDEX_NAME", "kay-fisker-corpus")
    EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(INDEX_NAME)
            embedder = SentenceTransformer(EMBED_MODEL)
            reranker = CrossEncoder("BAAI/bge-reranker-base")
            print(f"✅ Pinecone RAG aktivert ({INDEX_NAME}).")
        except Exception as e:
            print("❌ Feil ved Pinecone:", e)
            pinecone_index = None
    else:
        print("⚠️ Ingen PINECONE_API_KEY – RAG deaktivert.")

    print("✅ Modell og LoRA klart.")

@app.on_event("startup")
async def startup_event():
    load_model()  # Ikke i egen tråd


# --------------------
# Hjelpefunksjoner
# --------------------
def normalize_orthography(txt: str) -> str:
    if not txt:
        return txt
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    txt = re.sub(r"[ \t]*\n[ \t]*", " ", txt)
    txt = re.sub(r"\bpaa\b", "på", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def sanitize(out: str) -> str:
    return out.strip()

# --------------------
# Chat
# --------------------
def chat(user_prompt: str):
    if pipe is None:
        return "Modellen laster fortsatt inn – prøv igjen om et øyeblikk."
    # (du kan her lime inn din eksisterende kontekstlogikk med pinecone_search)
    prompt = f"Du er Kay Fisker. Spørsmål: {user_prompt}\nSvar:"
    result = pipe(prompt, max_new_tokens=400, temperature=0.3)
    out = result[0]["generated_text"].split("Svar:", 1)[-1].strip()
    return sanitize(out)

def chat_with_audio(user_prompt: str):
    text = chat(user_prompt)
    audio_path = None
    if tts is not None and text:
        try:
            audio = tts(text)
            audio_path = "tts_output.wav"
            with open(audio_path, "wb") as f:
                f.write(audio["audio"])
        except Exception as e:
            print("⚠️ Feil i TTS:", e)
    return text, audio_path

# --------------------
# API-endepunkter
# --------------------
@app.post("/chat")
async def api_chat(req: Request):
    data = await req.json()
    msg = data.get("message", "")
    text, audio_path = chat_with_audio(msg)
    if audio_path:
        return JSONResponse({"text": text, "audio_url": f"/{audio_path}"})
    return JSONResponse({"text": text})

@app.get("/tts")
async def get_audio():
    if os.path.exists("tts_output.wav"):
        return FileResponse("tts_output.wav", media_type="audio/wav")
    return JSONResponse({"error": "Ingen lyd generert"})

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<html><body><h3>Spør Fisker kjører ✅</h3></body></html>"
