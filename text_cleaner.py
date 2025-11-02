import re, unicodedata
from transformers import pipeline as hf_pipeline


# sett True for ekstra “smart” LM-reparasjon (litt tregere, men bedre)
USE_LM_REPAIR = True
_fill = None
if USE_LM_REPAIR:
    try:
        _fill = hf_pipeline("fill-mask", model="bert-base-multilingual-cased")
    except Exception:
        _fill = None  # fortsett uten LM hvis modellen ikke kan lastes

# hyppige suffikser i dansk/norsk som OCR ofte splitter etter et mellomrom
_SUFFIXES = (
    "ning", "ningen", "ninger", "ningens", "nigner", "nig",  # OCR-varianter
    "skab", "skabet", "skaber", "skaberne", "skabets",
    "mæssig", "mæssige", "mæssigt",
    "kunst", "kunsten", "kunstens",
    "byggeri", "byggeriet", "byggerier",
    "arbejde", "arbejdet", "arbejdere", "arbejderne",
    "former", "forhold", "funktion", "funktioner",
)
# ordstammer vi ofte ser i materialet
_STEMS = (
    "bygn", "arkitek", "bolig", "samfund", "typolog", "material",
    "konstruk", "funktion", "målestok", "facade", "snit", "byrum",
)

# faste, sikre erstatninger for typiske OCR-glipper
_SAFE_SUBS = [
    # ligaturer og rare kontrolltegn
    (r"\u00AD", ""),                           # myk bindestrek
    (r"[\u200B\u200C\u200D\uFEFF]", ""),       # zero-width
    (r"ﬁ", "fi"), (r"ﬂ", "fl"),
    # normaliser typografiske apostrofer/anførselstegn
    (r"[“”«»„‟]", '"'), (r"[’‚`´]", "'"),
    # normaliser tankestrek
    (r"[–—]", "-"),
]

# små, sikre OCR-rettelser du gjerne kan utvide etter hvert
_SAFE_DICT = {
    "arkitektuer": "arkitektur",
    "arkiteteker": "arkitekter",
    "arkitetekter": "arkitekter",
    "arkitetekterne": "arkitekterne",
    "bolibyggeri": "boligbyggeri",
    "maalestok": "målestok",   # hvis eldre OCR uten æ/ø/å
    "bygn ing": "bygning",
    "bygn ings": "bygnings",
}

def _apply_safe_subs(s: str) -> str:
    for pat, rep in _SAFE_SUBS:
        s = re.sub(pat, rep, s)
    return s

def _join_hyphen_linebreaks(s: str) -> str:
    # ord splittet med bindestrek + linjeskift
    s = re.sub(r"(\w+)-\s*\n+\s*(\w+)", r"\1\2", s)
    # ord splittet med bindestrek uten linjeskift (OCR-artefakt)
    s = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", s)
    return s

def _join_space_broken_morphemes(s: str) -> str:
    # 1) join “stem + <suffix>” når OCR har lagt inn et mellomrom
    suffix_alt = "|".join(map(re.escape, _SUFFIXES))
    stem_alt = "|".join(map(re.escape, _STEMS))
    # eksempel: bygn ingskunst → bygningskunst
    s = re.sub(rf"\b(({stem_alt}))\s+({suffix_alt})\b", r"\1\3", s, flags=re.IGNORECASE)

    # 2) join enkel konsonant/vokal-splitt: “kun st” → “kunst”
    s = re.sub(r"\b(\w{3,})\s+(\w{2,})\b", lambda m: _heuristic_join(m.group(1), m.group(2)), s)
    return s

def _heuristic_join(left: str, right: str) -> str:
    # forsøk kun når det ligner en morfem-splitt
    if len(left) < 3 or len(right) < 2:
        return f"{left} {right}"
    # typisk OCR-mønster: venstre ender på konsonant, høyre starter med “ning/kun/skab/…”
    if any(right.lower().startswith(suf[:3]) for suf in _SUFFIXES):
        return left + right
    # domeneord-stammer
    if any(left.lower().startswith(st) for st in _STEMS) and len(right) <= 8:
        return left + right
    return f"{left} {right}"

def _lm_repair(s: str) -> str:
    if _fill is None:
        return s
    # reparer kun mistenkelige “ord-ord” og “ord ord” som ser ut som splitt
    # 1) ord-ord
    for a, b in set(re.findall(r"\b([A-Za-zÆØÅæøå]{2,})-([A-Za-zÆØÅæøå]{2,})\b", s)):
        joined = f"{a}{b}"
        prompt = f"{a} [MASK] {b}"
        try:
            guess = _fill(prompt, top_k=1)[0]["token_str"].strip()
            if 2 <= len(guess) <= 20 and guess.isalpha():
                s = s.replace(f"{a}-{b}", guess)
            else:
                s = s.replace(f"{a}-{b}", joined)
        except Exception:
            s = s.replace(f"{a}-{b}", joined)

    # 2) mistenkelige “bygn nings”, “arkitek ter” osv.
    for a, b in set(re.findall(r"\b([A-Za-zÆØÅæøå]{3,})\s+([A-Za-zÆØÅæøå]{2,})\b", s)):
        if any(b.lower().startswith(suf[:3]) for suf in _SUFFIXES) or any(a.lower().startswith(st) for st in _STEMS):
            joined = f"{a}{b}"
            prompt = f"{a} [MASK] {b}"
            try:
                guess = _fill(prompt, top_k=1)[0]["token_str"].strip()
                if 2 <= len(guess) <= 20 and guess.isalpha():
                    s = s.replace(f"{a} {b}", guess)
                else:
                    s = s.replace(f"{a} {b}", joined)
            except Exception:
                s = s.replace(f"{a} {b}", joined)
    return s

def clean_text(s: str) -> str:
    """Beste OCR-rens: normaliser, lukk linjedelinger, gjenoppbygg morfemer, valgfritt LM-repair."""
    if not s:
        return s

    # 0) unicode normalisering (bevarer æøå)
    s = unicodedata.normalize("NFC", s)

    # 1) sikre erstatninger + fjern ligaturer/soft hyphen
    s = _apply_safe_subs(s)

    # 2) fjern harde CR, normaliser whitespace
    s = s.replace("\r", " ")

    # 3) slå sammen ord som er splittet med bindestrek/linjeskift
    s = _join_hyphen_linebreaks(s)

    # 4) join morfemer som er feil splittet av OCR (“bygn ingskunst” osv.)
    s = _join_space_broken_morphemes(s)

    # 5) sikre, manuelle mikrofiks (kan utvides etter behov)
    for wrong, right in _SAFE_DICT.items():
        s = re.sub(rf"\b{re.escape(wrong)}\b", right, s, flags=re.IGNORECASE)

    # 6) LM-basert finreparasjon (valgfritt)
    if USE_LM_REPAIR:
        s = _lm_repair(s)

    # 7) rydd opp: fjern doble mellomrom/linjer
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s