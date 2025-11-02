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

    # Pinecone RAG
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_primary = pc.Index("kay-fisker-primary")
            pinecone_secondary = pc.Index("kay-fisker-secondary")
            pinecone_quotes = pc.Index("kay-fisker-quotes")

            embedder = SentenceTransformer(EMBED_MODEL)
            reranker = CrossEncoder("BAAI/bge-reranker-base")
            print("✅ Pinecone RAG aktivert (3 indekser).")
        except Exception as e:
            print("❌ Feil ved Pinecone:", e)
    else:
        print("⚠️ Ingen PINECONE_API_KEY – RAG deaktivert.")

    print("✅ Modell og LoRA klart.")


@app.on_event("startup")
async def startup_event():
    load_model()

# --------------------
# Hjelpefunksjoner
# --------------------
def sanitize(out: str) -> str:
    return out.strip()


def pinecone_search(query: str, k: int = 5, index=None):
    try:
        if index is None or embedder is None:
            return [], []

        query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        res = index.query(vector=query_vec, top_k=k * 3, include_metadata=True)
        matches = res.get("matches", [])
        if not matches:
            return [], []

        pairs = [(query, m["metadata"].get("text", "")) for m in matches if m.get("metadata")]
        sources = [m["metadata"].get("source", "") for m in matches if m.get("metadata")]

        if reranker is not None and pairs:
            scores = reranker.predict(pairs)
            ranked = [x for _, x in sorted(zip(scores, matches), reverse=True)]
        else:
            ranked = matches

        texts, srcs = [], []
        for m in ranked[:k]:
            meta = m.get("metadata", {})
            if meta.get("text"):
                cleaned = clean_text(meta["text"])
                texts.append(cleaned.strip())
            if meta.get("source"):
                srcs.append(meta["source"])
        return texts, srcs
    except Exception as e:
        print("❌ Feil i pinecone_search:", e)
        return [], []


def hierarchical_search(query: str, k_primary=6, k_secondary=3, k_quotes=2):
    try:
        results, sources = [], []

        # 1️⃣ Primær: Fiskers egne tekster
        p_blocks, p_sources = pinecone_search(query, k=k_primary, index=pinecone_primary)
        results.extend(p_blocks); sources.extend(p_sources)

        # 2️⃣ Sekundær: Søberg m.fl.
        s_blocks, s_sources = pinecone_search(query, k=k_secondary, index=pinecone_secondary)
        results.extend(s_blocks); sources.extend(s_sources)

        # 3️⃣ Sitater
        q_blocks, q_sources = pinecone_search(query, k=k_quotes, index=pinecone_quotes)
        results.extend(q_blocks); sources.extend(q_sources)

        if not results:
            return "", []

        context = "\n\n---\n".join(results)
        sources = list(dict.fromkeys(sources))
        return context, sources
    except Exception as e:
        print("❌ Feil i hierarchical_search:", e)
        return "", []

# --------------------
# Chatfunksjon
# --------------------
def chat(user_prompt: str):
    try:
        if len(user_prompt.strip()) < 4:
            user_prompt = "Hej, hvordan arbejdede du som arkitekt?"

        bio_blocks, _ = pinecone_search("Kay Fiskers liv og virke", k=3, index=pinecone_secondary)
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
