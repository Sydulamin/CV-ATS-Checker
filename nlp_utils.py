import re
from typing import Set, Iterable, Dict
from collections import defaultdict
from difflib import get_close_matches
import spacy
from nltk.corpus import stopwords
import os
import uuid


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
# Load NLP model
# -----------------------------
nlp = spacy.load("en_core_web_md")
STOP = set(stopwords.words('english'))

# -----------------------------
# Synonyms / aliases
# -----------------------------
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

# -----------------------------
# Skill buckets
# -----------------------------
SKILL_BUCKETS = {
    "Frontend": {"html", "css", "javascript", "react", "vue", "angular"},
    "Backend": {"python", "django", "fastapi", "rest api", "nodejs"},
    "Databases": {"postgres", "mysql", "mongodb"},
    "Cloud": {"aws", "gcp", "azure"},
    "DevOps": {"docker", "kubernetes", "ci/cd", "jenkins"},
}

# -----------------------------
# Important skills
# -----------------------------
IMPORTANT_SKILLS = {"python", "django", "aws", "docker", "react", "rest api"}

# -----------------------------
# Normalize text
# -----------------------------
def normalize(text: str) -> str:
    text = text.lower().strip()
    for canonical, variants in SYNONYM_MAP.items():
        if text in variants or text == canonical:
            return canonical
    return text

# -----------------------------
# Extract keywords
# -----------------------------
def extract_keywords(text: str) -> Set[str]:
    doc = nlp(text.lower())
    keywords = set()

    for token in doc:
        if token.is_alpha and token.lemma_ not in STOP:
            keywords.add(normalize(token.lemma_))

    # Capture noun phrases (e.g., "rest api")
    for chunk in doc.noun_chunks:
        keywords.add(normalize(chunk.text.strip()))

    return keywords

# -----------------------------
# SpaCy similarity for fuzzy match
# -----------------------------
def similarity(a: str, b: str) -> float:
    da, db = nlp(a), nlp(b)
    if not da.vector_norm or not db.vector_norm:
        return 0.0
    return float(da.similarity(db))

# -----------------------------
# Match skills
# -----------------------------
def match_skills(resume_text: str, jd_text: str, thresh: float = 0.82):
    resume_tokens = extract_keywords(resume_text)
    jd_tokens = extract_keywords(jd_text)

    matched = set()
    not_matched = set(jd_tokens)
    fuzzy_matches = {}

    for jd in jd_tokens:
        if jd in resume_tokens:
            matched.add(jd)
            not_matched.discard(jd)
        else:
            # Fuzzy spaCy similarity
            best_term, best_sim = None, 0.0
            for r in resume_tokens:
                s = similarity(jd, r)
                if s > best_sim:
                    best_term, best_sim = r, s
            if best_sim >= thresh:
                matched.add(jd)
                not_matched.discard(jd)
                fuzzy_matches[jd] = {"matched_with": best_term, "score": round(best_sim, 3)}

            # Fallback difflib
            elif get_close_matches(jd, resume_tokens, n=1, cutoff=0.85):
                matched.add(jd)
                not_matched.discard(jd)

    return matched, not_matched, fuzzy_matches

# -----------------------------
# Categorize skills and important
# -----------------------------
def categorize_skills(matched: set, not_matched: set):
    categorized = defaultdict(lambda: {"matched": [], "not_matched": []})
    for bucket, skills in SKILL_BUCKETS.items():
        for skill in skills:
            if skill in matched:
                categorized[bucket]["matched"].append(skill)
            elif skill in not_matched:
                categorized[bucket]["not_matched"].append(skill)

    important = {
        "matched": [s for s in IMPORTANT_SKILLS if s in matched],
        "not_matched": [s for s in IMPORTANT_SKILLS if s in not_matched],
    }

    return categorized, important

# -----------------------------
# Generate HTML report
# -----------------------------
def generate_html_report(categorized: dict, important: dict, fuzzy_matches: dict):
    html = """
    <html>
    <head>
        <title>Resume Skill Match Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h2 { margin-top: 20px; }
            ul { list-style: none; padding-left: 0; }
            li::before { margin-right: 6px; font-weight: bold; }
            .matched li::before { content: "✔ "; color: green; }
            .not-matched li::before { content: "❌ "; color: red; }
            .fuzzy li::before { content: "➤ "; color: orange; }
        </style>
    </head>
    <body>
        <h1>Resume Skill Match Report</h1>
    """

    for bucket, skills in categorized.items():
        html += f"<h2>{bucket}</h2>"
        if skills["matched"]:
            html += "<div class='matched'><ul>"
            for skill in skills["matched"]:
                html += f"<li>{skill}</li>"
            html += "</ul></div>"
        if skills["not_matched"]:
            html += "<div class='not-matched'><ul>"
            for skill in skills["not_matched"]:
                html += f"<li>{skill}</li>"
            html += "</ul></div>"

    # Important
    html += "<h2>Important Skills</h2>"
    if important["matched"]:
        html += "<div class='matched'><ul>"
        for skill in important["matched"]:
            html += f"<li>{skill}</li>"
        html += "</ul></div>"
    if important["not_matched"]:
        html += "<div class='not-matched'><ul>"
        for skill in important["not_matched"]:
            html += f"<li>{skill}</li>"
        html += "</ul></div>"

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
    I am a software engineer with experience in Python, Django, AWS, Docker, and React.
    I have also worked with PostgreSQL and CI/CD pipelines.
    """

    jd_text = """
    We are looking for a developer skilled in Python, Django, REST API, FastAPI, React, Vue, Angular,
    AWS, Docker, Kubernetes, and MySQL.
    """

    matched, not_matched, fuzzy_matches = match_skills(resume_text, jd_text)
    categorized, important = categorize_skills(matched, not_matched)

    report = generate_html_report(categorized, important, fuzzy_matches)

    with open("resume_match_report.html", "w") as f:
        f.write(report)

    print("✅ Resume match report generated: resume_match_report.html")



