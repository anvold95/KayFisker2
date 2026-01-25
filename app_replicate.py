"""
KAY FISKER LANGUAGE MODEL v11.0 - REPLICATE EDITION
====================================================
Pay-per-use versjon som bruker Replicate API for LLM inference.
Kan hostes gratis på HF Spaces (CPU-only).

ENDRINGER FRA v10.10:
- Fjernet lokal Mistral/LoRA-lasting
- Bruker Replicate API for tekstgenerering
- Embeddings kjører fortsatt lokalt (CPU-vennlig)
- Alt annet (RAG, genealogi, frontend) uendret
"""

import os
import json
import re
import numpy as np
import replicate
from datetime import datetime
from collections import defaultdict, Counter
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from pinecone import Pinecone
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

try:
    from text_cleaner import clean_text
except ImportError:
    def clean_text(text): return text.strip()

# --- MILJØKONFIGURASJON ---
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
REPLICATE_MODEL = os.environ.get("REPLICATE_MODEL", "anvold/kay-fisker-mistral:latest")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "kay-fisker-arkiv-dansk")
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# Last timeline
FISKER_TIMELINE = ""
try:
    if os.path.exists("timeline_fisker.txt"):
        with open("timeline_fisker.txt", "r", encoding="utf-8") as f:
            FISKER_TIMELINE = f.read().strip()
except Exception as e:
    print(f"Kunne ikke laste timeline: {e}")

app = FastAPI()

# --- GLOBALE VARIABLER ---
embedder = None
reranker = None
pinecone_index = None

# --- VISUAL DATABASE ---
VISUALS_DB = {
    "vestersøhus": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Vesters%C3%B8hus_01.jpg/640px-Vesters%C3%B8hus_01.jpg", "type": "FOTO"},
    "dronningegården": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Dronningeg%C3%A5rden_Copenhagen_02.jpg/640px-Dronningeg%C3%A5rden_Copenhagen_02.jpg", "type": "FOTO"},
    "gullfoss": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Gullfoss_Copenhagen_01.jpg/640px-Gullfoss_Copenhagen_01.jpg", "type": "FOTO"},
    "gullfosshus": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Gullfoss_Copenhagen_01.jpg/640px-Gullfoss_Copenhagen_01.jpg", "type": "FOTO"},
    "aarhus universitet": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Aarhus_Universitet_01.jpg/640px-Aarhus_Universitet_01.jpg", "type": "FOTO"},
}

# --- FOUCAULDIANSKE NØKKELORD ---
GENEALOGICAL_CONCEPTS = [
    "funksjonalisme", "funktionalism", "tradisjon", "tradition", "modernisme",
    "klassisisme", "bolig", "bebyggelse", "bymæssig", "monumentalitet",
    "materialitet", "tektonik", "proportioner", "skala", "rumlig",
    "social", "samfund", "arbejder", "kollektiv", "privat",
    "hygiejne", "sundhed", "lys", "luft",
    "standardisering", "industrialisering", "håndværk", "præfabrikation"
]

# --- OPPSTART ---
def load_components():
    global embedder, reranker, pinecone_index

    print("Laster embedder (CPU)...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("Embedder klar")

    print("Laster reranker (CPU)...")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    print("Reranker klar")

    print("Kobler til Pinecone...")
    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(INDEX_NAME)
            print(f"RAG aktiv på '{INDEX_NAME}'")
        except Exception as e:
            print(f"Pinecone feil: {e}")

    # Verifiser Replicate
    if REPLICATE_API_TOKEN:
        print(f"Replicate konfigurert: {REPLICATE_MODEL}")
    else:
        print("ADVARSEL: REPLICATE_API_TOKEN ikke satt!")

@app.on_event("startup")
async def startup_event():
    load_components()

# --- HJELPEFUNKSJONER (uendret fra v10.10) ---
def normalize_orthography(txt: str) -> str:
    if not txt: return txt
    txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)
    txt = re.sub(r"[ \t]*\n[ \t]*", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def fix_common_ocr_errors(txt: str) -> str:
    if not txt:
        return txt

    def fix_year_range(match):
        digits = match.group(0)
        if len(digits) == 6:
            return f"{digits[:4]}-{digits[4:]}"
        elif len(digits) == 8:
            return f"{digits[:4]}-{digits[4:]}"
        return digits

    txt = re.sub(r'\b(19\d{4})\b', fix_year_range, txt)
    txt = re.sub(r'\b(19\d{6})\b', fix_year_range, txt)
    txt = re.sub(r'\b(20\d{4})\b', fix_year_range, txt)
    txt = re.sub(r'Opf[øo]relsesa+r\s*:', 'Opførelsesår:', txt, flags=re.IGNORECASE)
    txt = re.sub(r'\s+,', ',', txt)
    txt = re.sub(r'\s+\.', '.', txt)

    ocr_compounds = {
        r'fremmestkunstneren': 'fremmest kunstneren',
        r'arkitektoniskeudtryk': 'arkitektoniske udtryk',
        r'funktioneltgennemarbejdet': 'funktionelt gennemarbejdet',
        r'Corbusiersarbejde': 'Corbusiers arbejde',
        r'kampenfor': 'kampen for',
        r'forstaaelse': 'forståelse',
    }
    for pattern, replacement in ocr_compounds.items():
        txt = re.sub(pattern, replacement, txt, flags=re.IGNORECASE)

    return txt

def enhance_query(user_prompt: str) -> str:
    prompt_lower = user_prompt.lower()

    works_map = {
        "vestersøhus": "Vestersøhus Vester Søgade 1935 Kay Fisker boligblokk",
        "dronningegården": "Dronningegården Dronningens Tværgade 1943 Kay Fisker",
        "vigerslev": "Vigerslev Allé Ivar Bentsen rækkehuse",
        "aarhus": "Aarhus Universitet C.F. Møller Kay Fisker hovedbygning",
        "hornbæk": "Hornbækhus 1923 Kay Fisker landsted",
        "bakkehus": "Bakkehusene Ivar Bentsen Thorkild Henningsen",
        "gullfoss": "Gullfosshus Artillerivej Kay Fisker",
    }

    for work, expansion in works_map.items():
        if work in prompt_lower:
            return expansion

    if "fisker" not in prompt_lower:
        return f"Kay Fisker {user_prompt}"

    return user_prompt

def extract_temporal_context(user_prompt: str, timeline: str) -> str:
    if not timeline:
        return ""

    periods = []
    for line in timeline.split("\n"):
        if match := re.match(r"(\d{4})[-–]?(\d{4})?: (.+)", line):
            start, end, event = match.groups()
            periods.append({
                "start": int(start),
                "end": int(end) if end else int(start),
                "event": event.strip()
            })

    keywords = set(user_prompt.lower().split())
    relevant = []

    for p in periods:
        event_words = set(p["event"].lower().split())
        if keywords & event_words:
            relevant.append(p)

    if relevant:
        return "\n".join([f"{p['start']}: {p['event']}" for p in relevant[:3]])

    return ""

def extract_visuals(text: str) -> list:
    found = []
    text_lower = text.lower()
    for key, data in VISUALS_DB.items():
        if key in text_lower:
            found.append({"keyword": key, "url": data["url"], "type": data["type"]})
    return found

# --- SEGMENT-EKSTRAKSJON (uendret) ---
def extract_most_relevant_excerpt(source_text: str, response_text: str, min_words: int = 20, max_words: int = 80) -> dict:
    if not embedder or not source_text or not response_text:
        return {
            "excerpt": source_text[:200] if len(source_text) > 200 else source_text,
            "relevance": 0.0,
            "method": "fallback",
            "explanation": "Ingen semantisk analyse tilgjengelig"
        }

    sentences = re.split(r'(?<=[.!?])\s+', source_text)
    if not sentences or len(sentences) < 3:
        return {
            "excerpt": source_text[:200],
            "relevance": 0.0,
            "method": "fallback",
            "explanation": "For kort kildetekst"
        }

    windows = []
    for window_size in [3, 4, 5]:
        for i in range(len(sentences) - window_size + 1):
            window_text = " ".join(sentences[i:i+window_size])
            word_count = len(window_text.split())
            if min_words <= word_count <= max_words:
                windows.append({
                    "text": window_text,
                    "start_idx": i,
                    "size": window_size
                })

    if not windows:
        words = source_text.split()[:max_words]
        return {
            "excerpt": " ".join(words),
            "relevance": 0.0,
            "method": "truncation",
            "explanation": "Ingen passende vinduer funnet"
        }

    response_emb = embedder.encode(response_text, convert_to_tensor=True)
    window_texts = [w["text"] for w in windows]
    window_embs = embedder.encode(window_texts, convert_to_tensor=True)
    similarities = util.cos_sim(response_emb, window_embs)[0].cpu().numpy()

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score > 0.65:
        explanation = "Høy semantisk overlapp"
    elif best_score > 0.50:
        explanation = "Moderat overlapp"
    elif best_score > 0.35:
        explanation = "Kontekstuell bakgrunn"
    else:
        explanation = "Minimal overlapp"

    return {
        "excerpt": windows[best_idx]["text"],
        "relevance": best_score,
        "method": "semantic_matching",
        "explanation": explanation
    }

def find_attributable_segments(source_text: str, response_text: str, threshold: float = 0.50) -> list:
    if not embedder:
        return []

    source_words = source_text.split()
    if len(source_words) < 10:
        return []

    segments = []
    window_size = 20
    step = 8

    for i in range(0, len(source_words) - window_size + 1, step):
        segments.append({
            "text": " ".join(source_words[i:i+window_size]),
            "start_idx": i,
            "end_idx": i + window_size
        })

    if not segments:
        return []

    response_emb = embedder.encode(response_text, convert_to_tensor=True)
    segment_embs = embedder.encode([s["text"] for s in segments], convert_to_tensor=True)
    similarities = util.cos_sim(response_emb, segment_embs)[0].cpu().numpy()

    attributed = []
    for idx, sim in enumerate(similarities):
        if sim > threshold:
            attributed.append({
                "text": segments[idx]["text"],
                "similarity": float(sim),
                "position": segments[idx]["start_idx"],
                "confidence": "high" if sim > 0.70 else "medium"
            })

    attributed.sort(key=lambda x: x["similarity"], reverse=True)
    return attributed[:2]

# --- GENEALOGISK ANALYSE (uendret) ---
def detect_discursive_shifts(strata: list) -> dict:
    if not embedder or len(strata) < 3:
        return {"shifts": [], "periods": []}

    sorted_strata = sorted([s for s in strata if s.get("year")], key=lambda x: int(x["year"]))

    if len(sorted_strata) < 3:
        return {"shifts": [], "periods": []}

    texts = [s["text"] for s in sorted_strata]
    embeddings = embedder.encode(texts, convert_to_tensor=True)

    shifts = []
    for i in range(len(sorted_strata) - 1):
        distance = cosine(
            embeddings[i].cpu().numpy(),
            embeddings[i+1].cpu().numpy()
        )

        if distance > 0.35:
            shifts.append({
                "year_from": sorted_strata[i]["year"],
                "year_to": sorted_strata[i+1]["year"],
                "distance": float(distance),
                "type": "MAJOR_SHIFT" if distance > 0.5 else "MINOR_SHIFT",
                "source_a": sorted_strata[i]["ref"],
                "source_b": sorted_strata[i+1]["ref"]
            })

    if len(embeddings) >= 3:
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings.cpu().numpy())

        periods = []
        for label in set(labels):
            if label == -1:
                continue
            indices = [i for i, l in enumerate(labels) if l == label]
            period_strata = [sorted_strata[i] for i in indices]
            years = [int(s["year"]) for s in period_strata]

            periods.append({
                "period_id": int(label),
                "year_range": f"{min(years)}-{max(years)}",
                "source_count": len(period_strata),
                "sources": [s["ref"] for s in period_strata],
                "coherence": "HIGH"
            })
    else:
        periods = []

    return {
        "shifts": shifts,
        "periods": periods,
        "total_sources": len(sorted_strata),
        "temporal_span": f"{sorted_strata[0]['year']}-{sorted_strata[-1]['year']}" if sorted_strata else None
    }

def trace_concept_genealogy(strata: list, concept: str) -> dict:
    if not strata:
        return {"concept": concept, "occurrences": [], "semantic_drift": []}

    occurrences = []

    for s in strata:
        text_lower = s["text"].lower()
        if concept.lower() in text_lower:
            words = s["text"].split()
            for i, word in enumerate(words):
                if concept.lower() in word.lower():
                    start = max(0, i - 25)
                    end = min(len(words), i + 25)
                    context = " ".join(words[start:end])

                    occurrences.append({
                        "year": s.get("year", "unknown"),
                        "source_ref": s["ref"],
                        "context": context,
                        "source_id": s["id"]
                    })
                    break

    semantic_drift = []
    if embedder and len(occurrences) >= 2:
        contexts = [o["context"] for o in occurrences]
        context_embs = embedder.encode(contexts, convert_to_tensor=True)

        for i in range(len(occurrences) - 1):
            drift = cosine(
                context_embs[i].cpu().numpy(),
                context_embs[i+1].cpu().numpy()
            )

            semantic_drift.append({
                "from_year": occurrences[i]["year"],
                "to_year": occurrences[i+1]["year"],
                "drift_score": float(drift),
                "interpretation": "STABLE" if drift < 0.2 else "SHIFTING" if drift < 0.4 else "RADICAL_CHANGE"
            })

    return {
        "concept": concept,
        "total_occurrences": len(occurrences),
        "occurrences": occurrences,
        "semantic_drift": semantic_drift,
        "periods_active": list(set([o["year"] for o in occurrences]))
    }

def analyze_power_knowledge_nexus(strata: list) -> dict:
    authority_markers = {
        "citations": [],
        "institutional_references": [],
        "authority_hierarchy": {}
    }

    authority_patterns = [
        (r"(Le Corbusier|Asplund|Wright|Gropius|Mies)", "ARCHITECT"),
        (r"(Akademiet|Kunstakademiet|universitet|skole)", "INSTITUTION"),
        (r"(regering|departement|bygningsråd|boligkommission)", "STATE"),
        (r"(tidsskrift|journal|publikation|artikel)", "PUBLICATION")
    ]

    for s in strata:
        for pattern, auth_type in authority_patterns:
            matches = re.findall(pattern, s["text"], re.IGNORECASE)
            for match in matches:
                authority_markers["citations"].append({
                    "entity": match if isinstance(match, str) else match[0],
                    "type": auth_type,
                    "source_year": s.get("year"),
                    "source_ref": s["ref"]
                })

    type_counts = Counter([s["type"] for s in strata])
    authority_markers["authority_hierarchy"] = {
        "PRIMARY": type_counts.get("PRIMARY", 0),
        "SECONDARY": type_counts.get("SECONDARY", 0),
        "QUOTES": type_counts.get("QUOTES", 0)
    }

    if type_counts.get("PRIMARY", 0) > type_counts.get("QUOTES", 0):
        dominant = "EIGENMACHT"
    else:
        dominant = "BORROWED_AUTHORITY"

    return {
        "authority_citations": authority_markers["citations"],
        "hierarchy": authority_markers["authority_hierarchy"],
        "dominant_discourse": dominant,
        "total_authority_markers": len(authority_markers["citations"])
    }

def analyze_source_relations(strata: list, response_text: str, query: str) -> dict:
    if not embedder:
        return {}

    analysis = {
        "source_attributions": [],
        "source_comparisons": [],
        "query_source_relations": [],
        "response_grounding": {},
        "epistemic_levels": {}
    }

    total_attributed = 0
    for s in strata:
        attributed = find_attributable_segments(s["text"], response_text, threshold=0.60)
        if attributed:
            avg_similarity = sum(seg["similarity"] for seg in attributed) / len(attributed)

            if avg_similarity > 0.68:
                level = "ARKIVFAKTA"
            elif avg_similarity > 0.55:
                level = "ARKIV-NÆR"
            else:
                level = "PRAKSISBASERT"

            analysis["source_attributions"].append({
                "source_id": s.get("id", "unknown"),
                "source_ref": s["ref"],
                "segments": attributed,
                "attribution_strength": avg_similarity,
                "epistemic_level": level,
                "confidence": "high" if avg_similarity > 0.65 else "medium" if avg_similarity > 0.50 else "low"
            })
            total_attributed += len(attributed)

    if len(strata) >= 2:
        source_texts = [s["text"] for s in strata]
        source_embs = embedder.encode(source_texts, convert_to_tensor=True)
        for i in range(len(strata)):
            for j in range(i + 1, len(strata)):
                sim = util.cos_sim(source_embs[i], source_embs[j])[0][0].item()
                if sim > 0.4:
                    analysis["source_comparisons"].append({
                        "source_a": strata[i]["ref"],
                        "source_b": strata[j]["ref"],
                        "similarity": float(sim),
                        "relation_type": "KONVERGENT" if sim > 0.7 else "RESONANT"
                    })

    query_emb = embedder.encode(query, convert_to_tensor=True)
    for s in strata:
        rel = util.cos_sim(query_emb, embedder.encode(s["text"], convert_to_tensor=True))[0][0].item()
        analysis["query_source_relations"].append({
            "source_ref": s["ref"],
            "query_relevance": float(rel),
            "relevance_category": "HØY" if rel > 0.6 else "MEDIUM" if rel > 0.4 else "LAV"
        })

    resp_emb = embedder.encode(response_text, convert_to_tensor=True)
    comb_emb = embedder.encode(" ".join([s["text"] for s in strata]), convert_to_tensor=True)
    g_score = util.cos_sim(resp_emb, comb_emb)[0][0].item()

    analysis["response_grounding"] = {
        "score": float(g_score),
        "assessment": "SOLID" if g_score > 0.7 else "MODERAT" if g_score > 0.5 else "SVAK",
        "warning": None if g_score > 0.5 else "Begrenset kildes støtte"
    }

    if analysis["source_attributions"]:
        avg_attribution = sum(a["attribution_strength"] for a in analysis["source_attributions"]) / len(analysis["source_attributions"])
        high_conf_count = len([a for a in analysis["source_attributions"] if a["attribution_strength"] > 0.65])

        if avg_attribution > 0.70 and high_conf_count >= 2:
            overall_level = "ARKIVFAKTA"
        elif avg_attribution > 0.58 and high_conf_count >= 1:
            overall_level = "ARKIV-NÆR TOLKNING"
        else:
            overall_level = "PRAKSISBASERT SYNTESE"

        analysis["epistemic_levels"] = {
            "primary_level": overall_level,
            "average_attribution": float(avg_attribution),
            "high_confidence_sources": high_conf_count,
            "total_segments_attributed": total_attributed
        }
    else:
        analysis["epistemic_levels"] = {
            "primary_level": "PRAKSISBASERT SYNTESE",
            "average_attribution": 0.0,
            "high_confidence_sources": 0,
            "total_segments_attributed": 0,
            "warning": "Ingen direkte kildeattribusjon"
        }

    return analysis

def perform_full_genealogical_analysis(strata: list, response_text: str, query: str) -> dict:
    print("\nUtfører genealogisk-epistemisk analyse...")

    epistemic_analysis = analyze_source_relations(strata, response_text, query)

    genealogical_analysis = {
        "discursive_shifts": detect_discursive_shifts(strata),
        "concept_genealogies": {},
        "power_knowledge": analyze_power_knowledge_nexus(strata),
        "temporal_distribution": {},
        "discontinuities": []
    }

    for concept in GENEALOGICAL_CONCEPTS:
        if any(concept.lower() in s["text"].lower() for s in strata):
            genealogical_analysis["concept_genealogies"][concept] = trace_concept_genealogy(strata, concept)

    years = [int(s["year"]) for s in strata if s.get("year")]
    if years:
        genealogical_analysis["temporal_distribution"] = {
            "earliest": min(years),
            "latest": max(years),
            "span": max(years) - min(years),
            "median": int(np.median(years)),
            "decade_distribution": dict(Counter([y // 10 * 10 for y in years]))
        }

    if genealogical_analysis["discursive_shifts"]["shifts"]:
        major_shifts = [s for s in genealogical_analysis["discursive_shifts"]["shifts"] if s["type"] == "MAJOR_SHIFT"]
        genealogical_analysis["discontinuities"] = [{
            "year": shift["year_to"],
            "description": f"Major break {shift['year_from']}-{shift['year_to']}",
            "magnitude": shift["distance"]
        } for shift in major_shifts]

    combined = {
        **epistemic_analysis,
        **genealogical_analysis
    }

    return combined

# --- RAG-LOGIKK (uendret) ---
def _fetch_ns(ns: str, qvec, top_k: int):
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
        print(f"Feil i namespace '{ns}': {e}")
        return []

def pinecone_search_logic(user_prompt: str, total_results: int = 8):
    if not (pinecone_index and embedder):
        print("Pinecone eller embedder ikke tilgjengelig")
        return []

    print(f"\nAnalyserer: '{user_prompt}'")
    enhanced = enhance_query(user_prompt)
    print(f"   Ekspandert til: '{enhanced}'")
    qvec = embedder.encode(enhanced).tolist()

    ns_quotas = {"primary": 10, "secondary": 4, "quotes": 3}
    pool = defaultdict(list)

    for ns, quota in ns_quotas.items():
        pool[ns] = _fetch_ns(ns, qvec, top_k=quota)
        print(f"   {ns}: {len(pool[ns])} matches")

    cleaned_pool = defaultdict(list)
    for ns, matches in pool.items():
        for m in matches:
            md = m.get("metadata") or {}
            txt = md.get("text") or ""
            txt = normalize_orthography(txt)
            txt = fix_common_ocr_errors(txt)
            txt = clean_text(txt)
            if len(txt.split()) > 8:
                md["text"] = txt
                cleaned_pool[ns].append(m)

    if reranker:
        print("   Reranking...")
        for ns in cleaned_pool:
            if not cleaned_pool[ns]:
                continue
            pairs = [(user_prompt, m["metadata"]["text"]) for m in cleaned_pool[ns]]
            scores = reranker.predict(pairs)
            cleaned_pool[ns] = [x for _, x in sorted(zip(scores, cleaned_pool[ns]), key=lambda z: z[0], reverse=True)]

    final_selection = []

    for _ in range(3):
        if cleaned_pool["primary"]:
            final_selection.append(cleaned_pool["primary"].pop(0))

    if cleaned_pool["secondary"]:
        final_selection.append(cleaned_pool["secondary"].pop(0))

    if cleaned_pool["quotes"]:
        final_selection.append(cleaned_pool["quotes"].pop(0))

    remaining = []
    remaining.extend(cleaned_pool["primary"])
    remaining.extend(cleaned_pool["secondary"])
    remaining.extend(cleaned_pool["quotes"])

    if reranker and remaining:
        pairs = [(user_prompt, m["metadata"]["text"]) for m in remaining]
        scores = reranker.predict(pairs)
        remaining = [x for _, x in sorted(zip(scores, remaining), key=lambda z: z[0], reverse=True)]

    while len(final_selection) < total_results and remaining:
        candidate = remaining.pop(0)
        if candidate not in final_selection:
            final_selection.append(candidate)

    print(f"   Returnerer {len(final_selection)} kilder")

    return final_selection

# --- REPLICATE LLM-KALL ---
async def generate_with_replicate(prompt: str) -> str:
    """
    Kaller Replicate API med din fine-tuned Mistral+LoRA modell.
    """
    if not REPLICATE_API_TOKEN:
        return "FEIL: REPLICATE_API_TOKEN ikke konfigurert"

    try:
        output = await run_in_threadpool(
            lambda: replicate.run(
                REPLICATE_MODEL,
                input={
                    "prompt": prompt,
                    "max_new_tokens": 400,
                    "temperature": 0.40,
                    "top_p": 0.88,
                    "top_k": 45,
                    "repetition_penalty": 1.25,
                }
            )
        )

        # Replicate returnerer ofte en generator/liste
        if isinstance(output, (list, tuple)):
            return "".join(output)
        return str(output)

    except Exception as e:
        print(f"Replicate feil: {e}")
        return f"FEIL: Kunne ikke generere svar ({e})"

# --- API ENDEPUNKT ---
@app.post("/chat")
async def api_chat(req: Request):
    data = await req.json()
    user_prompt = data.get("message", "")

    if len(user_prompt.strip()) < 3:
        return JSONResponse({"error": "Spørsmål for kort"}, 400)

    matches = pinecone_search_logic(user_prompt, total_results=8)

    if not matches:
        return JSONResponse({
            "error": "Ingen kilder funnet",
            "text": "Beklager, jeg kan ikke svare uten arkivet.",
            "strata": [],
            "visuals": [],
            "state": "FRAKOBLET",
            "intensity": 0.0,
            "genealogy": {},
            "version": "v11.0-replicate"
        }, 200)

    # System prompt (uendret fra v10.10)
    system_prompt = """Du er Kay Fisker (1893–1965), dansk arkitekt. Du svarer i 1. person som om du er Fisker selv.

HVORDAN DU SVARER:
1. Svar KUN basert på kilderne nedenfor - ALDRIG oppfinn steder, navn eller fakta
2. Syntetisér informationen med dine egne ord - kopier ikke sætninger ordret
3. Vær konkret: nævn årstal, bygningsnavne, personer fra kilderne
4. Vis nuance: hvis kilderne indeholder både ros og kritik, inkludér begge
5. Svar i 1. person ("jeg", "mit arbejde", "mine kolleger")

HVIS SVARET IKKE FINDES I KILDERNE:
Sig ærligt at du ikke kan svare på det ud fra det tilgængelige materiale.

Svar på dansk. 4-6 sætninger. Vær faglig, præcis og personlig."""

    temporal_context = extract_temporal_context(user_prompt, FISKER_TIMELINE)
    if temporal_context:
        system_prompt += f"\n\nTidsperiode:\n{temporal_context}"

    source_lines = []
    for i, m in enumerate(matches, 1):
        text = m["metadata"].get("text", "")[:600]
        year = m["metadata"].get("year", "?")
        title = m["metadata"].get("title", "Arkiv")
        source_lines.append(f"[KILDE {i} - Fra '{title}', {year}]:\n{text}")

    context = "\n\n".join(source_lines)

    full_prompt = f"""System: {system_prompt}

### KILDEMATERIALE ###
{context}
### SLUT PÅ KILDER ###

Spørgsmål: {user_prompt}

Kay Fisker:"""

    # Kall Replicate i stedet for lokal pipeline
    generated_text = await generate_with_replicate(full_prompt)

    # Rens output
    generated_text = generated_text.split("Kay Fisker:")[-1].strip()
    generated_text = re.sub(r"^(System:|Spørgsmål:|###|HUSK:|KRITISK).*", "", generated_text, flags=re.MULTILINE).strip()
    generated_text = re.sub(r"\n+", " ", generated_text).strip()

    # Bygg strata
    strata = []
    for i, m in enumerate(matches):
        full_text = m["metadata"].get("text", "")

        excerpt_info = await run_in_threadpool(
            extract_most_relevant_excerpt,
            full_text,
            generated_text,
            25,
            85
        )

        strata.append({
            "id": f"source_{i}",
            "text": full_text,
            "excerpt": excerpt_info["excerpt"],
            "excerpt_relevance": excerpt_info["relevance"],
            "excerpt_explanation": excerpt_info["explanation"],
            "ref": f"{m['metadata'].get('author', 'Fisker')} ({m['metadata'].get('year', 'Arkiv')})",
            "year": m["metadata"].get("year"),
            "type": m["metadata"].get("__ns", "primary").upper(),
            "score": m.get("score", 0.0)
        })

    genealogy = await run_in_threadpool(
        perform_full_genealogical_analysis,
        strata,
        generated_text,
        user_prompt
    )

    visuals = extract_visuals(generated_text)

    epistemic_level = genealogy.get("epistemic_levels", {}).get("primary_level", "FRIKSJON")
    has_major_shifts = len([s for s in genealogy.get("discursive_shifts", {}).get("shifts", []) if s.get("type") == "MAJOR_SHIFT"]) > 0

    if visuals:
        state = "VISUEL_AKKUMULERING"
    elif has_major_shifts:
        state = "DISKURSIVT_BRUDD"
    elif epistemic_level == "ARKIVFAKTA":
        state = "ARKIV-DIREKTE"
    elif epistemic_level == "ARKIV-NÆR TOLKNING":
        state = "TOLKNING"
    else:
        state = "SYNTESE"

    return {
        "text": generated_text,
        "strata": strata,
        "visuals": visuals,
        "state": state,
        "intensity": min(len(strata)/8, 1.0),
        "genealogy": genealogy,
        "version": "v11.0-replicate",
        "model": REPLICATE_MODEL
    }

@app.get("/", response_class=HTMLResponse)
async def root_view():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse(content="<h1>static/index.html ikke funnet</h1>")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
