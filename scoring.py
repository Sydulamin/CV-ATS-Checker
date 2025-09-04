from __future__ import annotations
from typing import Dict, List
from nlp_utils import normalize, expand_with_synonyms, fuzzy_match, SKILL_BUCKETS
import os
from dotenv import load_dotenv

load_dotenv()
W_KEY = float(os.getenv('WEIGHT_KEYWORDS', 0.65))
W_COMP = float(os.getenv('WEIGHT_COMPLETENESS', 0.20))
W_FMT = float(os.getenv('WEIGHT_FORMAT', 0.15))
MAX_JD = int(os.getenv('MAX_JD_KEYWORDS', 400))

SECTION_HINTS = ["experience", "education", "skill", "project", "certification", "summary"]


def section_completeness(resume_text: str) -> Dict[str, bool]:
    low = (resume_text or '').lower()
    return {h: (h in low) for h in SECTION_HINTS}


def format_warnings(resume_text: str) -> List[str]:
    warns = []
    lines = (resume_text or '').splitlines()
    if len(lines) < max(4, len((resume_text or '')) // 600):
        warns.append("Text appears sparsely line-broken. Avoid multi-column layouts, text boxes, or tables.")
    if (resume_text or '').count('\t') > 15:
        warns.append("Excessive tab characters detected. Use simple bullets instead of tables.")
    return warns


def bucketize(terms: List[str]):
    buckets = {k: [] for k in SKILL_BUCKETS}
    other = []
    for t in terms:
        placed = False
        for k, vocab in SKILL_BUCKETS.items():
            if t in vocab:
                buckets[k].append(t)
                placed = True
                break
        if not placed:
            other.append(t)
    return buckets, other


def compute_score(resume_text: str, jd_text: str) -> Dict:
    """
    Compute CV score based on keyword match with Job Description,
    section completeness, and format warnings.
    """
    # --- STRICT keyword match between CV and JD only ---
    cv_keywords = expand_with_synonyms(normalize(resume_text))
    jd_keywords = expand_with_synonyms(normalize(jd_text))

    if len(jd_keywords) > MAX_JD:
        jd_keywords = set(list(jd_keywords)[:MAX_JD])

    exact, fuzzy = fuzzy_match(jd_keywords, cv_keywords)

    matched_set = set(exact) | set(fuzzy.keys())
    missing = sorted(list(set(jd_keywords) - matched_set))

    # --- Subscores ---
    kw_score = len(matched_set) / max(1, len(jd_keywords))  # keyword coverage
    sect = section_completeness(resume_text)
    comp_score = sum(1 for v in sect.values() if v) / len(sect)
    warns = format_warnings(resume_text)
    fmt_score = 1.0 if not warns else max(0.0, 1.0 - 0.25 * len(warns))

    total_score = round(100.0 * (W_KEY * kw_score + W_COMP * comp_score + W_FMT * fmt_score), 2)

    # --- Bucketize ---
    matched_sorted = sorted(list(matched_set))
    matched_buckets, matched_other = bucketize(matched_sorted)
    missing_buckets, missing_other = bucketize(missing)

    return {
        "score": total_score,
        "subscores": {
            "keywords": round(100 * kw_score, 1),
            "completeness": round(100 * comp_score, 1),
            "format": round(100 * fmt_score, 1),
        },
        "matched": matched_sorted,
        "missing": missing,
        "fuzzy_matches": fuzzy,
        "sections": sect,
        "warnings": warns,
        "matched_buckets": matched_buckets,
        "matched_other": matched_other,
        "missing_buckets": missing_buckets,
        "missing_other": missing_other,
    }
