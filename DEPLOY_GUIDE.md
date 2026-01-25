# Kay Fisker LLM - Pay-Per-Use Deploy Guide

Denne guiden forklarer hvordan du setter opp en **gratis hosting** med **pay-per-use** LLM inference.

## Arkitektur

```
Bruker → HF Spaces (gratis) → Replicate API (pay-per-use) → Svar
              ↓
         Pinecone (gratis)
```

## Steg 1: Sett opp Replicate-konto

1. Gå til [replicate.com](https://replicate.com) og lag en konto
2. Gå til **Account Settings** → **API tokens**
3. Kopier din API-token (starter med `r8_...`)
4. Legg til et betalingskort under **Billing**

## Steg 2: Deploy din modell til Replicate

### Alternativ A: Push merged modell (enklest)

```bash
# Installer Cog
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
chmod +x /usr/local/bin/cog

# Gå til modell-mappen
cd replicate_model

# Login til Replicate
cog login

# Push modellen (dette tar 10-20 minutter første gang)
cog push r8.im/DITT_BRUKERNAVN/kay-fisker-mistral
```

### Alternativ B: Bruk Replicate Web UI

1. Gå til [replicate.com/create](https://replicate.com/create)
2. Velg "Import from GitHub" eller last opp filene
3. Pek til `replicate_model/` mappen

## Steg 3: Konfigurer HF Spaces

1. Gå til [huggingface.co/new-space](https://huggingface.co/new-space)
2. Velg **Docker** som SDK
3. Velg **CPU basic** (gratis!)
4. Last opp disse filene:
   - `app_replicate.py` → rename til `app.py`
   - `requirements_replicate.txt` → rename til `requirements.txt`
   - `text_cleaner.py`
   - `timeline_fisker.txt`
   - `static/` mappen
   - `Dockerfile_replicate` → rename til `Dockerfile`

5. Legg til **Secrets** i Space settings:
   - `REPLICATE_API_TOKEN` = din Replicate API-token
   - `REPLICATE_MODEL` = `DITT_BRUKERNAVN/kay-fisker-mistral:latest`
   - `PINECONE_API_KEY` = din Pinecone nøkkel
   - `INDEX_NAME` = `kay-fisker-arkiv-dansk`

## Steg 4: Oppdater Dockerfile

Lag `Dockerfile_replicate`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Installer dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopier app
COPY . .

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Kostnadsestimat

| Komponent | Kostnad |
|-----------|---------|
| HF Spaces CPU | Gratis |
| Pinecone (free tier) | Gratis |
| Replicate inference | ~$0.0023/sek |

### Eksempel månedskostnad:

| Bruk | Antall spørringer | ~Kostnad/mnd |
|------|-------------------|--------------|
| Lav | 100 | ~$2-3 |
| Medium | 500 | ~$10-15 |
| Høy | 2000 | ~$40-50 |

## Feilsøking

### "REPLICATE_API_TOKEN ikke satt"
→ Sjekk at du har lagt til token i HF Space secrets

### "Model not found"
→ Sjekk at `REPLICATE_MODEL` er riktig formatert: `brukernavn/modellnavn:versjon`

### Treg respons
→ Første kall etter inaktivitet tar lenger (cold start ~30-60 sek)
→ Etterfølgende kall er raske (~2-5 sek)

## Ferdig!

Din Kay Fisker-modell kjører nå på:
- **Gratis** hosting (HF Spaces)
- **Pay-per-use** LLM (Replicate)
- **Gratis** vektordatabase (Pinecone)

Du betaler kun når noen faktisk bruker modellen! 🎉
