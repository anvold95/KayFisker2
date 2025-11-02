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
# Pinecone-søkefunksjon
# --------------------
def pinecone_search(query: str, k: int = 5):
    """Returnerer en liste med relevante tekstblokker og kildereferanser fra Pinecone."""
    try:
        if pinecone_index is None or embedder is None:
            print("⚠️ Pinecone ikke aktivert – returnerer tomt resultat.")
            return [], []

        # Lag embedding for spørringen
        query_vec = embedder.encode(query).tolist()

        # Søk i Pinecone
        res = pinecone_index.query(vector=query_vec, top_k=k, include_metadata=True)

        matches = res.get("matches", [])
        if not matches:
            return [], []

        # Hent tekstblokker og kilder
        texts = []
        sources = []
        for m in matches:
            meta = m.get("metadata", {})
            text = meta.get("text", "")
            src = meta.get("source", "")
            if text:
                texts.append(text.strip())
            if src:
                sources.append(src)

        return texts, sources

    except Exception as e:
        print("❌ Feil i pinecone_search:", e)
        return [], []

# --------------------
# Chatfunksjon (revidert)
# --------------------
def chat(user_prompt: str):
    try:
        # Hvis modellen får et veldig kort input, legg til en liten "presisering"
        if len(user_prompt.strip()) < 4:
            user_prompt = "Hej, hvordan arbejdede du som arkitekt?"

        # Hent et lite biografisk grunnlag (Martin Søberg + tidslinje)
        bio_blocks, _ = pinecone_search("Kay Fiskers liv og virke", k=3)
        bio_context = " ".join(bio_blocks[:2]) if bio_blocks else ""

        # Hent kontekst relatert til spørsmålet
        context_blocks, sources = pinecone_search(user_prompt, k=8)
        context = "\n\n---\n".join(context_blocks) if context_blocks else ""

        if not context:
            return "Dette er ikke omtalt i mine tekster."

        # Systemrolle og ramme
        system_prompt = (
            "Du er Kay Fisker (1893–1965), dansk arkitekt og professor ved Kunstakademiets Arkitektskole. "
            "Du svarer som deg selv, i nøkternt dansk fagsprog preget av presisjon og disiplin. "
            "Unngå symbolikk, poesi eller idealistiske vendinger. "
            "Dine svar skal handle om arkitektur, undervisning, formgivning og bygningers samfundsmæssige rolle.\n\n"
            "Kort faglig beskrivelse (Martin Søberg):\n"
            f"{bio_context}\n\n"
            "Oversikt over ditt liv og virke:\n"
            f"{FISKER_TIMELINE}\n\n"
            "Svar konkret, saklig og uten ornamentikk. "
            "Hvis materialet ikke dekker spørsmålet, si kort at det ikke omtales."
        )

        # Regler og arbeidsmåte
        rules = """Regler:
- Svar KUN med støtte i 'Kontekst' nedenfor. Ikke finn på noe.
- Parafrasér kort og hold deg til epoken og materialet.
- Unngå manifest-fraser og allegorisk språk.
- Gi konkrete vurderinger før refleksjon.
- Maks 8–10 setninger.
- Hvis grunnlag mangler: skriv kort at dette ikke omtales og avslutt."""

        plan = """Arbeidsmåte:
1) Les konteksten og identifiser 2–3 relevante setninger.
2) Formuler svaret som faglig prosa, uten metaforer.
3) Avslutt nøkternt hvis materialet er utilstrekkelig."""

        # Endelig prompt
        full_prompt = (
            f"{system_prompt}\n\n{rules}\n\n{plan}\n\n"
            f"Kontekst:\n{context}\n\n"
            f"Spørgsmål: {user_prompt}\n\nSvar:"
        )

        # Generering
        result = pipe(
            full_prompt,
            max_new_tokens=350,
            temperature=0.25,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

        out = result[0]["generated_text"]
        if "Svar:" in out:
            out = out.split("Svar:", 1)[-1].strip()

        out = sanitize(out)
        if sources:
            out += "\n\nKilder:\n- " + "\n- ".join(sources)

        return out or "Dette er ikke omtalt i mine tekster."

    except Exception as e:
        print("❌ Feil i chat:", e)
        return "Feil i prosesseringen."


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

