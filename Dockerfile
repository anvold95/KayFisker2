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

# Installer avhengigheter
RUN pip install --upgrade pip
RUN pip install -r requirements.txt || true

# Installer FastAPI og uvicorn hvis ikke i requirements
RUN pip install fastapi uvicorn "transformers>=4.35.0" "torch>=2.1.0" "sentence-transformers" "peft" "huggingface-hub" "pinecone" "espnet"

# Eksponer porten Hugging Face bruker
EXPOSE 7860

# Start FastAPI-serveren
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
