import re
from typing import Set, Iterable, Dict, List, Tuple
from collections import defaultdict
from difflib import get_close_matches, SequenceMatcher
import os
import uuid

# -----------------------------
# Upload dir helpers (unchanged)
# -----------------------------
UPLOAD_DIR = "uploaded_cvs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_file(upload_file):
    ext = os.path.splitext(upload_file.filename)[1]
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())

    return file_path


# -----------------------------
# NLP setup with safe fallbacks
# -----------------------------
# We try medium model for vectors; fall back to small; fall back to blank.
# Code gracefully degrades similarity features if no vectors available.
import spacy
from spacy.language import Language

def _load_spacy() -> Language:
    for model in ("en_core_web_md", "en_core_web_sm"):
        try:
            return spacy.load(model)
        except Exception:
            continue
    # Last resort: blank English pipeline with basic rules
    nlp_blank = spacy.blank("en")
    if "sentencizer" not in nlp_blank.pipe_names:
        nlp_blank.add_pipe("sentencizer")
    return nlp_blank

nlp = _load_spacy()

# NLTK stopwords/wordnet with fallbacks
try:
    from nltk.corpus import stopwords, wordnet
    _STOP = set(stopwords.words("english"))
    _WORDNET_OK = True
except Exception:
    _STOP = {
        "a","an","the","and","or","but","if","while","with","for","to","from","by","on","in","of",
        "is","are","was","were","be","been","being","as","at","this","that","these","those","it",
        "its","into","about","over","under","after","before","between","within","without","not"
    }
    _WORDNET_OK = False


# -----------------------------
# Synonyms / aliases (broadened)
# Still used by normalize(); safe to extend later
# -----------------------------
SYNONYM_MAP: Dict[str, Set[str]] = {
    # Tech
    "aws": {"amazon web services", "aws"},
    "k8s": {"kubernetes", "k8s"},
    "docker": {"docker", "containers"},
    "ci/cd": {"ci/cd", "continuous integration", "continuous delivery", "continuous deployment"},
    "postgres": {"postgres", "postgresql"},
    "gcp": {"google cloud", "gcp"},
    "azure": {"microsoft azure", "azure"},
    "rest api": {"rest", "rest api", "restful"},
    "nlp": {"nlp", "natural language processing"},

    # Finance
    "accounting": {"accounting", "bookkeeping", "financial accounting"},
    "financial analysis": {"financial analysis", "finance analysis", "analyst"},
    "auditing": {"auditing", "audit"},
    "budgeting": {"budgeting", "budget planning", "cost planning"},
    "tax": {"tax", "taxation"},

    # Marketing & Sales
    "seo": {"seo", "search engine optimization"},
    "sem": {"sem", "search engine marketing"},
    "content marketing": {"content marketing", "content strategy"},
    "crm": {"crm", "customer relationship management"},
    "lead generation": {"lead gen", "lead generation"},

    # HR
    "recruitment": {"recruitment", "talent acquisition", "hiring"},
    "payroll": {"payroll", "salary processing"},
    "onboarding": {"onboarding", "induction"},

    # Healthcare
    "patient care": {"patient care", "clinical care"},
    "medical billing": {"medical billing", "healthcare billing"},
    "clinical research": {"clinical research", "research study"},

    # Education
    "curriculum design": {"curriculum design", "course design"},
    "classroom management": {"classroom management", "class management"},
    "assessment": {"assessment", "exam creation", "test creation"},
}

# -----------------------------
# IMPORTANT_SKILLS placeholder
# NOTE: Kept for compatibility, but now computed dynamically from the JD.
# -----------------------------
IMPORTANT_SKILLS: Set[str] = set()

# -----------------------------
# Deprecated fixed buckets (kept name for compatibility)
# We'll fill this dynamically as semantic "clusters" per JD.
# -----------------------------
SKILL_BUCKETS: Dict[str, Set[str]] = {}


# -----------------------------
# Text normalization & helpers
# -----------------------------
def normalize(text: str) -> str:
    """Lowercase, strip, and map known synonyms to a canonical key."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    for canonical, variants in SYNONYM_MAP.items():
        if text == canonical or text in variants:
            return canonical
    return text

def _lemmatize(token) -> str:
    # Works even if no tagger/lemmatizer present
    lemma = getattr(token, "lemma_", None) or token.text
    lemma = lemma.lower()
    return re.sub(r"[^a-z0-9\-\+\.\s]", "", lemma).strip()


# -----------------------------
# Keyword / phrase extraction (domain-agnostic)
# -----------------------------
def extract_keywords(text: str) -> Set[str]:
    """
    Extract domain-agnostic keywords & phrases:
    - Unigrams (alpha) excluding stopwords
    - Noun chunks (if pipeline supports)
    - Named entities of types that often indicate skills/subjects (ORG, PRODUCT, FAC, WORK_OF_ART)
    """
    if not text:
        return set()

    doc = nlp(text.lower())
    keywords: Set[str] = set()

    # tokens
    for token in doc:
        if token.is_alpha:
            lemma = _lemmatize(token)
            if lemma and lemma not in _STOP and len(lemma) > 1:
                keywords.add(normalize(lemma))

    # noun chunks (if available)
    if hasattr(doc, "noun_chunks"):
        for chunk in doc.noun_chunks:
            ph = normalize(chunk.text.strip())
            if ph and ph not in _STOP and len(ph) > 1:
                keywords.add(ph)

    # named entities (optional; keep broad but useful)
    if getattr(doc, "ents", None):
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "FAC", "WORK_OF_ART", "NORP"}:
                ph = normalize(ent.text)
                if ph and ph not in _STOP and len(ph) > 1:
                    keywords.add(ph)

    # Clean trivial leftovers (single-char etc.)
    keywords = {k for k in keywords if len(k) > 1}

    return keywords


# -----------------------------
# Similarity (robust fallbacks)
# -----------------------------
def _spacy_cosine(a: str, b: str) -> float:
    da, db = nlp(a), nlp(b)
    if not getattr(da, "vector_norm", 0) or not getattr(db, "vector_norm", 0):
        return 0.0
    # spaCy similarity is cosine on mean vectors if vectors exist
    return float(da.similarity(db))

def _string_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def similarity(a: str, b: str) -> float:
    """
    Prefer semantic similarity if vectors available; otherwise fall back to string similarity.
    """
    sem = _spacy_cosine(a, b)
    if sem > 0:
        return sem
    return _string_sim(a, b)


# -----------------------------
# Optional dynamic synonym expansion using WordNet
# -----------------------------
def _expand_with_wordnet(term: str) -> Set[str]:
    if not _WORDNET_OK:
        return set()
    syns = set()
    from nltk.corpus import wordnet as wn
    for syn in wn.synsets(term):
        for l in syn.lemma_names():
            clean = normalize(l.replace("_", " "))
            if clean and clean != term:
                syns.add(clean)
    return syns


# -----------------------------
# JD-driven "important skills"
# -----------------------------
def extract_required_skills(jd_text: str) -> Set[str]:
    """
    Treat *all* JD keywords as required. This keeps the logic universal.
    You can later add heuristics (e.g., words near 'required', 'must have').
    """
    req = extract_keywords(jd_text)
    # light synonym expansion for recall
    expanded = set(req)
    for term in list(req):
        expanded |= _expand_with_wordnet(term)
    return expanded


# -----------------------------
# Matching logic (kept signature)
# -----------------------------
def match_skills(resume_text: str, jd_text: str, thresh: float = 0.82) -> Tuple[Set[str], Set[str], Dict[str, Dict[str, object]]]:
    """
    Return (matched, not_matched, fuzzy_matches)
    - matched: canonical JD terms that are covered in the resume (exact or fuzzy)
    - not_matched: JD terms missing from the resume
    - fuzzy_matches: {jd_term: {"matched_with": resume_term, "score": float}}
    """
    resume_tokens = extract_keywords(resume_text)
    jd_tokens = extract_required_skills(jd_text)

    matched = set()
    not_matched = set(jd_tokens)
    fuzzy_matches: Dict[str, Dict[str, object]] = {}

    # Fast exact overlap first
    for jd in list(jd_tokens):
        if jd in resume_tokens:
            matched.add(jd)
            not_matched.discard(jd)

    # Fuzzy matching
    for jd in list(not_matched):
        best_term, best_sim = None, 0.0

        # Try synonym expands of JD for better recall
        jd_variants = {jd} | _expand_with_wordnet(jd)

        for jdv in jd_variants:
            # quick lexicographic close match shortcut
            close = get_close_matches(jdv, resume_tokens, n=1, cutoff=0.88)
            if close:
                best_term, best_sim = close[0], 0.88
                break

            # semantic/string similarity search
            for r in resume_tokens:
                s = similarity(jdv, r)
                if s > best_sim:
                    best_term, best_sim = r, s

        if best_sim >= thresh:
            matched.add(jd)
            not_matched.discard(jd)
            fuzzy_matches[jd] = {"matched_with": best_term, "score": round(float(best_sim), 3)}

    return matched, not_matched, fuzzy_matches


# -----------------------------
# Dynamic semantic buckets for UI (kept function name & return shape)
# -----------------------------
def _build_dynamic_buckets(all_terms: Set[str], max_clusters: int = 6, sim_cutoff: float = 0.78) -> Dict[str, Set[str]]:
    """
    Lightweight greedy clustering by similarity to form semantic buckets.
    Produces labels like 'Cluster 1: <anchor-term>'.
    """
    terms = list(sorted(all_terms))
    clusters: List[Tuple[str, Set[str]]] = []  # [(anchor, set), ...]

    for t in terms:
        placed = False
        for i, (anchor, members) in enumerate(clusters):
            # similarity to cluster anchor
            s = similarity(t, anchor)
            if s >= sim_cutoff:
                members.add(t)
                placed = True
                break
        if not placed:
            clusters.append((t, {t}))
            if len(clusters) >= max_clusters:
                # Put remaining terms in the closest existing cluster
                continue

    # Label clusters
    labeled: Dict[str, Set[str]] = {}
    for idx, (anchor, members) in enumerate(clusters, start=1):
        label = f"Cluster {idx}: {anchor}"
        labeled[label] = members

    return labeled


def categorize_skills(matched: set, not_matched: set):
    """
    Kept signature & output structure, but buckets are dynamic clusters.
    Returns:
      categorized: { bucket_name: {"matched": [...], "not_matched": [...]} }
      important: {"matched": [...], "not_matched": [...]}
    """
    global SKILL_BUCKETS
    all_jd_terms = set(matched) | set(not_matched)
    # Build dynamic clusters of JD terms
    SKILL_BUCKETS = _build_dynamic_buckets(all_jd_terms)

    categorized = defaultdict(lambda: {"matched": [], "not_matched": []})
    for bucket, skills in SKILL_BUCKETS.items():
        for skill in sorted(skills):
            if skill in matched:
                categorized[bucket]["matched"].append(skill)
            elif skill in not_matched:
                categorized[bucket]["not_matched"].append(skill)

    # IMPORTANT_SKILLS is now JD-driven (keep name & shape for compatibility)
    important_all = extract_required_skills(" ".join(all_jd_terms)) if all_jd_terms else set()
    important = {
        "matched": [s for s in sorted(important_all) if s in matched],
        "not_matched": [s for s in sorted(important_all) if s in not_matched],
    }

    return categorized, important


# -----------------------------
# HTML Report (kept function & structure)
# -----------------------------
def generate_html_report(categorized: dict, important: dict, fuzzy_matches: dict):
    html = """
    <html>
    <head>
        <title>Resume Skill Match Report</title>
        <meta charset="utf-8" />
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.5; }
            h1 { margin-bottom: 0; }
            .sub { color: #666; margin-top: 4px; }
            h2 { margin-top: 24px; }
            ul { list-style: none; padding-left: 0; margin: 8px 0; }
            li::before { margin-right: 6px; font-weight: bold; }
            .matched li::before { content: "✔ "; color: #188038; }
            .not-matched li::before { content: "❌ "; color: #d93025; }
            .fuzzy li::before { content: "➤ "; color: #e37400; }
            .bucket { padding: 12px 14px; border: 1px solid #eee; border-radius: 12px; margin-bottom: 14px; }
            .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
            .chip { display:inline-block; border:1px solid #eee; border-radius:999px; padding:4px 10px; margin:2px 6px 2px 0; }
            .section { margin-top: 16px; }
            .muted { color:#777; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>Resume Skill Match Report</h1>
        <div class="sub">Domain-agnostic, JD-driven matching</div>
    """

    # Buckets
    for bucket, skills in categorized.items():
        html += f"<div class='bucket'><h2>{bucket}</h2>"
        html += "<div class='two-col'>"
        # Matched
        html += "<div><h3>Matched</h3>"
        if skills["matched"]:
            html += "<div>"
            for skill in skills["matched"]:
                html += f"<span class='chip'>{skill}</span>"
            html += "</div>"
        else:
            html += "<div class='muted'>No matches in this cluster.</div>"
        html += "</div>"

        # Not matched
        html += "<div><h3>Missing (from JD)</h3>"
        if skills["not_matched"]:
            html += "<div>"
            for skill in skills["not_matched"]:
                html += f"<span class='chip'>{skill}</span>"
            html += "</div>"
        else:
            html += "<div class='muted'>Fully covered.</div>"
        html += "</div>"

        html += "</div></div>"

    # Important (JD-driven)
    html += "<h2>Important Skills (from JD)</h2>"
    if important.get("matched"):
        html += "<div class='matched'><ul>"
        for skill in important["matched"]:
            html += f"<li>{skill}</li>"
        html += "</ul></div>"
    if important.get("not_matched"):
        html += "<div class='not-matched'><ul>"
        for skill in important["not_matched"]:
            html += f"<li>{skill}</li>"
        html += "</ul></div>"
    if not important.get("matched") and not important.get("not_matched"):
        html += "<div class='muted'>No important skills detected.</div>"

    # Fuzzy matches
    if fuzzy_matches:
        html += "<h2>Fuzzy Matches</h2><div class='fuzzy'><ul>"
        for jd, info in fuzzy_matches.items():
            html += f"<li>{jd} ➤ matched with {info['matched_with']} (score: {info['score']})</li>"
        html += "</ul></div>"

    html += "</body></html>"
    return html


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    resume_text = """
    Experienced professional with strengths in financial analysis, budgeting, and stakeholder communication.
    Managed SEO campaigns and CRM workflows for e-commerce; familiar with content marketing and lead generation.
    Tools: Excel, Power BI, Salesforce.
    """

    jd_text = """
    We seek a Finance & Marketing Operations Specialist with skills in accounting, financial analysis, auditing,
    budgeting, CRM, SEO, content marketing, campaign management, and stakeholder communication.
    Experience with Excel or BI tools is required.
    """

    matched, not_matched, fuzzy_matches = match_skills(resume_text, jd_text)
    categorized, important = categorize_skills(matched, not_matched)

    report = generate_html_report(categorized, important, fuzzy_matches)

    with open("resume_match_report.html", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ Resume match report generated: resume_match_report.html")
