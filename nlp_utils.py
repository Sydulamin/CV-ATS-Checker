from __future__ import annotations
import spacy
from nltk.corpus import stopwords
from typing import Set, Iterable, Dict

# Load spaCy medium model (has word vectors for similarity)
nlp = spacy.load("en_core_web_md")
STOP = set(stopwords.words('english'))

# Expand as needed — acts as aliases and phrase equivalents
SYNONYM_MAP: Dict[str, Set[str]] = {
    "aws": {"amazon web services", "aws"},
    "k8s": {"kubernetes", "k8s"},
    "docker": {"docker", "containers"},
    "ci/cd": {"ci/cd", "continuous integration", "continuous delivery", "continuous deployment"},
    "postgres": {"postgres", "postgresql"},
    "gcp": {"google cloud", "gcp"},
    "azure": {"microsoft azure", "azure"},
    "rest api": {"rest", "rest api", "restful"},
    "nlp": {"nlp", "natural language processing"},
}

# A tiny skill taxonomy to categorize terms in reports
SKILL_BUCKETS = {
    "Cloud": {"aws", "gcp", "azure"},
    "DevOps": {"docker", "kubernetes", "k8s", "ci/cd"},
    "Databases": {"postgres", "postgresql", "mysql", "mongodb"},
    "Backend": {"python", "django", "fastapi", "rest", "rest api"},
}


def normalize(text: str) -> Set[str]:
    """Tokenize + lowercase + lemmatize + drop stopwords & non-alpha."""
    doc = nlp((text or '').lower())
    toks = {
        tok.lemma_.strip()
        for tok in doc
        if tok.is_alpha and tok.lemma_ not in STOP
    }
    # Quick phrase capture: add frequent bigrams for REST API etc.
    # (Very light heuristic just for MVP)
    words = [t.text for t in doc if t.is_alpha and t.lemma_ not in STOP]
    bigrams = {f"{a} {b}" for a, b in zip(words, words[1:])}
    toks |= {bg for bg in bigrams if any(bg in v or bg in k for k, v in SYNONYM_MAP.items())}
    return toks


def expand_with_synonyms(tokens: Set[str]) -> Set[str]:
    expanded = set(tokens)
    for canonical, variants in SYNONYM_MAP.items():
        if canonical in tokens or (tokens & variants):
            expanded |= variants | {canonical}
    return expanded


def similarity(a: str, b: str) -> float:
    """spaCy vector similarity between two short terms (0..1)."""
    da, db = nlp(a), nlp(b)
    if not da.vector_norm or not db.vector_norm:
        return 0.0
    return float(da.similarity(db))


def fuzzy_match(jd_tokens: Iterable[str], resume_tokens: Iterable[str], thresh: float = 0.82):
    """Return (exact_matches, fuzzy_matches_map). fuzzy map: jd_term -> resume_term."""
    jd = list(jd_tokens)
    rs = list(resume_tokens)
    exact = sorted(set(jd) & set(rs))
    fuzzy = {}
    for j in jd:
        if j in exact:
            continue
        best_term, best_sim = None, 0.0
        for r in rs:
            s = similarity(j, r)
            if s > best_sim:
                best_term, best_sim = r, s
        if best_sim >= thresh:
            fuzzy[j] = {"matched_with": best_term, "score": round(best_sim, 3)}