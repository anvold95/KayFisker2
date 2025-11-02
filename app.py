import os
os.environ["HF_HOME"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"

import json, re, torch
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline as hf_pipeline
from huggingface_hub import hf_hub_download, login, whoami
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from text_cleaner import clean_text  # ✅ integrert

# --------------------
# Last timeline
# --------------------
with open("timeline_fisker.txt", "r", encoding="utf-8") as f:
    FISKER_TIMELINE = f.read().strip()

# --------------------
# FastAPI app
# --------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

# --------------------
# Globale variabler
# --------------------
pipe = None
tts = None
tokenizer = None
embedder = None
reranker = None
pinecone_primary = None
pinecone_secondary = None
pinecone_quotes = None

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
        print("⚠️ Kunne ikke logge inn på Hugging Face:", e)
else:
    print("⚠️ Ingen HF_TOKEN funnet – prøver uten innlogging.")

# --------------------
# Modell-last
# --------------------
def load_model():
    global pipe, tts, tokenizer, embedder, reranker
    global pinecone_primary, pinecone_secondary, pinecone_quotes

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel

    BASE = "mistralai/Mistral-7B-Instruct-v0.3"
    LORA = "anvold/fisker-lora-clean"

    print("🧩 Laster base-modell + LoRA-adapter …")

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

   # --------------------
# Pinecone (RAG)
# --------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "kay-fisker-corpus-1024")
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

pinecone_index = None
embedder = None
reranker = None

if not PINECONE_API_KEY:
    print("⚠️ Ingen PINECONE_API_KEY funnet – RAG deaktivert.")
else:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        embedder = SentenceTransformer(EMBED_MODEL)
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        print(f"✅ Pinecone RAG aktivert på index '{INDEX_NAME}' (embed={EMBED_MODEL}).")
    except Exception as e:
        print("❌ Feil ved tilkobling til Pinecone:", e)
        pinecone_index = None

# --------------------
# Tekst-normalisering (OCR/ortografi)
# --------------------
def normalize_orthography(txt: str) -> str:
    if not txt:
        return txt
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    txt = re.sub(r"[ \t]*\n[ \t]*", " ", txt)
    txt = re.sub(r"\bpaa\b", "på", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

_CAPTION_REGEX = re.compile(r"\b(Fig\.?|Figur|Foto|Pl\.?|Plate|Billede|Plan|Snit|Facade|Kort|Tegning)\b", re.I)
_TECH_TERMS = re.compile(r"(typolog|lejlighed|bolig|målestok|facade|snit|byrum|trappe|køkken|bad|toilet|opgang)", re.I)

def _is_good_context(txt: str) -> bool:
    if not txt:
        return False
    words = txt.split()
    if len(words) < 12:
        return False
    if not _TECH_TERMS.search(txt):
        return False
    return True

def _fetch_ns(ns: str, qvec, top_k: int):
    try:
        res = pinecone_index.query(vector=qvec, top_k=top_k, include_metadata=True, namespace=ns)
        matches = res.get("matches", []) or []
        for m in matches:
            m["metadata"]["__ns"] = ns
        return matches
    except Exception:
        return []

def pinecone_search(user_prompt: str, k: int = 8):
    if not (pinecone_index and embedder):
        print("⚠️ pinecone/embedding ikke aktiv.")
        return [], []

    qvec = embedder.encode(user_prompt).tolist()
    namespaces = ["primary", "secondary", "quotes"]
    weights = {"primary": 1.0, "secondary": 1.5, "quotes": 0.3}
    pool = []
    for ns in namespaces:
        m = _fetch_ns(ns, qvec, top_k=k)
        print(f"🔍 {ns}: {len(m)} treff")
        for item in m:
            item["weight"] = weights.get(ns, 0.5)
            pool.append(item)

    pre = []
    for m in pool:
        md = m.get("metadata") or {}
        raw = (md.get("text") or "").strip()
        txt = normalize_orthography(raw)
        txt = clean_text(txt)
        if not _is_good_context(txt):
            continue
        m["metadata"]["text"] = txt
        pre.append(m)

    if not pre:
        print("⚠️ Ingen godkjent kontekst etter filtrering.")
        return [], []

    pre.sort(key=lambda x: x.get("weight", 1.0), reverse=True)

    if reranker is not None:
        pairs = [(user_prompt, x["metadata"]["text"]) for x in pre]
        scores = reranker.predict(pairs)
        pre = [x for _, x in sorted(zip(scores, pre), key=lambda z: z[0], reverse=True)]

    context_blocks, sources = [], []
    for m in pre[:6]:
        md = m["metadata"]
        context_blocks.append(md["text"].strip())
        src = " / ".join(
            x for x in [md.get("author"), md.get("title"), str(md.get("year") or ""), md.get("pages")]
            if x
        )
        if src:
            sources.append(src)
    sources = list(dict.fromkeys(sources))
    print("🧱 context_blocks hentet:", len(context_blocks))
    return context_blocks, sources



@app.on_event("startup")
async def startup_event():
    load_model()

# --------------------
# Hjelpefunksjoner
# --------------------
def sanitize(out: str) -> str:
    """Behold original utgangstekst uten aggressiv filtrering."""
    return out.strip()
# --------------------
# Chatfunksjon
# --------------------
def chat(user_prompt: str):
    try:
        if len(user_prompt.strip()) < 4:
            user_prompt = "Hej, hvordan arbejdede du som arkitekt?"

        bio_blocks, _ = pinecone_search("Kay Fiskers liv og virke", k=3)
        bio_context = " ".join(bio_blocks[:2]) if bio_blocks else ""

        context, sources = hierarchical_search(user_prompt)
        if not context:
            return "Dette er ikke omtalt i mine tekster."

        system_prompt = (
            "Du er Kay Fisker (1893–1965), dansk arkitekt og professor ved Kunstakademiets Arkitektskole. "
            "Du svarer som deg selv, i nøgternt dansk fagsprog præget af præcision og disciplin. "
            "Undgå symbolik, poesi og ideologiske manifest. "
            "Dine svar skal handle om arkitektur, undervisning, formgivning og bygningers samfundsmæssige rolle.\n\n"
            "Arkivmateriale er hentet fra tre kilder:\n"
            "- Primær: dine egne artikler og forelesninger (prioriteres først)\n"
            "- Sekundær: analyser av Martin Søberg og samtidige\n"
            "- Sitater: korte utdrag som illustrerer poenger\n\n"
            "Du må kun støtte deg på det følgende materialet.\n\n"
            "Kort faglig beskrivelse (Martin Søberg):\n"
            f"{bio_context}\n\n"
            "Oversikt over dit liv og virke:\n"
            f"{FISKER_TIMELINE}\n\n"
            "Svar konkret, saklig og uten ornamentikk. "
            "Hvis materialet ikke dækker spørgsmålet, si kort at det ikke omtales."
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"Arkivmateriale (hentet fra primær-, sekundær- og sitatkilder):\n{context}\n\n"
            f"Spørgsmål: {user_prompt}\n\nSvar:"
        )

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

        print(f"🔎 Søk i namespace '{namespace}' → {len(matches)} treff")


def hierarchical_search(query: str):
    """Bruker felles søk over alle namespaces."""
    if pinecone_index is None:
        return "", []

    context_blocks, sources = pinecone_search(query, k=8)
    if not context_blocks:
        print("⚠️ Ingen treff fra RAG.")
        return "", []

    context = "\n\n---\n".join(context_blocks)
    return context, sources



# --------------------
# Chat + TTS
# --------------------
def chat_with_audio(user_prompt: str):
    text = chat(user_prompt)
    audio_path = None
    if tts is not None and text:
        try:
            audio = tts(text)
            audio_path = "tts_output.wav"
            sf.write(audio_path, audio["audio"], samplerate=22050)
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
