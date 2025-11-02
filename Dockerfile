FROM python:3.10-slim

# Installer systemavhengigheter
RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app
COPY . /app

# Installer Python-pakker
RUN pip install --no-cache-dir -r requirements.txt

# Eksponer Hugging Face standardport (7860)
EXPOSE 7860

# Start FastAPI-serveren
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
