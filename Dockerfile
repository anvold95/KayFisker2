# Base image med GPU-støtte
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Installer systempakker
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Sett arbeidsmappe
WORKDIR /app

# Kopier filer
COPY . .

# Oppdater pip og installer avhengigheter
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt || true

# Installer ekstra pakker (dersom ikke i requirements)
RUN pip install --no-cache-dir \
    fastapi uvicorn \
    "transformers>=4.35.0" \
    "torch>=2.1.0" \
    "sentence-transformers" \
    "peft" \
    "huggingface-hub" \
    "pinecone" \
    "espnet"

# Sett Hugging Face cache-mappe
ENV HF_HOME=/app/cache
ENV HF_HUB_CACHE=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# (valgfritt) Forhåndslast LoRA-adapter for raskere oppstart
# NB: krever HF_TOKEN satt i build environment (Settings → Variables → HF_TOKEN)
# RUN huggingface-cli download anvold/fisker-lora-clean --local-dir /app/cache/fisker-lora

# Eksponer porten Hugging Face bruker
EXPOSE 7860

# Start FastAPI-serveren
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
