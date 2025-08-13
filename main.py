from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Tuple
import io
import re
from docx import Document
from pdfminer.high_level import extract_text
from pdf2image import convert_from_bytes
import pytesseract
import spacy
from nltk.corpus import stopwords

# Load spaCy medium English model for semantic similarity
nlp = spacy.load("en_core_web_md")

# Stopwords
stop_words = set(stopwords.words("english"))

# Allowed file extensions
ALLOWED_EXTS = [".pdf", ".docx"]

# FastAPI app
app = FastAPI(title="Professional ATS CV Checker", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

templates = Jinja2Templates(directory="templates")


# -------------------
# File Utilities
# -------------------
def get_extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def extract_text_from_upload(upload_file) -> Tuple[str, str]:
    """Extract text from PDF/DOCX. For scanned PDFs, use OCR."""
    try:
        name = getattr(upload_file, "filename", "")
        ext = get_extension(name)
        if ext not in ALLOWED_EXTS:
            raise ValueError("Only .pdf and .docx are supported")

        upload_file.file.seek(0)
        data = upload_file.file.read()

        # DOCX extraction
        if ext == ".docx":
            try:
                doc = Document(io.BytesIO(data))
                text = "\n".join(p.text for p in doc.paragraphs)
                return text, ext
            except Exception as e:
                print(f"DOCX extraction error: {e}")
                return "", ext

        # PDF extraction
        if ext == ".pdf":
            # First try text extraction
            try:
                text = extract_text(io.BytesIO(data))
                if text.strip():
                    return text, ext
            except Exception as e:
                print(f"PDF text extraction error: {e}")

            # If empty, try OCR
            try:
                images = convert_from_bytes(data)
                text = "\n".join([pytesseract.image_to_string(img) for img in images])
                return text, ext
            except Exception as e:
                print(f"OCR error: {e}")
                return "", ext

        return "", ext

    except Exception as e:
        print(f"Unexpected extraction error: {e}")
        return "", ""


# -------------------
# Keyword & Section Utilities
# -------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def extract_keywords(text: str):
    text = clean_text(text)
    words = set(text.split())
    return words - stop_words


# Section weighting: give more importance to Skills & Experience
SECTION_WEIGHTS = {
    "experience": 2,
    "skills": 2,
    "education": 1,
    "projects": 1
}


def check_cv_format(text: str):
    warnings = []

    sections = ["experience", "education", "skills", "projects"]
    for s in sections:
        if s not in text.lower():
            warnings.append(f"Section '{s}' is missing")

    # Check email
    if not re.search(r"\b[\w.-]+@[\w.-]+\.\w{2,4}\b", text):
        warnings.append("No valid email found")

    # Check phone
    if not re.search(r"\+?\d[\d\s-]{7,}\d", text):
        warnings.append("No valid phone number found")

    # Check bullet points consistency
    bullets = len(re.findall(r"[\u2022\-*]", text))
    if bullets < 3:
        warnings.append("Few bullet points found; consider using bullets for clarity")

    return warnings


# -------------------
# Semantic Matching
# -------------------
def semantic_match(resume_keywords, jd_keywords):
    matched = set()
    missing = set(jd_keywords)
    resume_doc = nlp(" ".join(resume_keywords))
    for word in jd_keywords:
        word_doc = nlp(word)
        max_sim = max([word_doc.similarity(token) for token in resume_doc])
        if max_sim >= 0.75:  # similarity threshold
            matched.add(word)
            missing.discard(word)
    return matched, missing


# -------------------
# Scoring
# -------------------
def compute_score(resume_text: str, job_description: str):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)

    # Semantic matching
    matched, missing = semantic_match(resume_keywords, jd_keywords)

    # Weight by section
    keyword_score = 0
    total_weight = 0
    for s, w in SECTION_WEIGHTS.items():
        if s in resume_text.lower():
            keyword_score += len(matched & extract_keywords(s)) * w
            total_weight += w
    keyword_score = int(len(matched) / len(jd_keywords) * 100) if jd_keywords else 0

    # Formatting
    warnings = check_cv_format(resume_text)
    format_score = max(0, 100 - len(warnings) * 10)

    # Final score
    final_score = int((keyword_score + format_score) / 2)

    return {
        "score": final_score,
        "keyword_score": keyword_score,
        "format_score": format_score,
        "matched": list(matched),
        "missing": list(missing),
        "warnings": warnings,
    }


# -------------------
# Routes
# -------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/check", response_class=HTMLResponse)
async def check_cv(
    request: Request, resume: UploadFile = File(...), job_description: str = Form(...)
):
    try:
        result_pair = extract_text_from_upload(resume)
        resume_text, ext = result_pair

        result = compute_score(resume_text, job_description)

        return templates.TemplateResponse("index.html", {"request": request, "result": result})

    except Exception as e:
        print(f"Error in /check: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})
