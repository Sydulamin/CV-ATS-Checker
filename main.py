import os
import io
import re
import uuid
from typing import Tuple, Optional
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from docx import Document
from pdfminer.high_level import extract_text
from pdf2image import convert_from_bytes
import pytesseract
import spacy
import nltk

# -----------------------
# NLTK Stopwords Fix
# -----------------------
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_dir)
from nltk.corpus import stopwords

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)
    STOP_WORDS = set(stopwords.words("english"))

# -----------------------
# spaCy
# -----------------------
try:
    nlp = spacy.load("en_core_web_md")
except Exception:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Professional ATS CV Checker", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
templates = Jinja2Templates(directory="templates")

# -----------------------
# ensure uploads dir exists and serve static uploads
# -----------------------
uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# -----------------------
# Allowed extensions
# -----------------------
ALLOWED_EXTS = [".pdf", ".docx"]

# -----------------------
# Temp result store
# -----------------------
RESULTS_CACHE = {}

# -----------------------
# Utilities
# -----------------------
def get_extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file to ./uploads and return the relative url path (uploads/filename).
    """
    filename = os.path.basename(getattr(upload_file, "filename", "file"))
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    dest_name = f"{uuid.uuid4().hex}_{safe_name}"
    dest_path = os.path.join(uploads_dir, dest_name)

    # ensure we read from start
    upload_file.file.seek(0)
    with open(dest_path, "wb") as out_f:
        out_f.write(upload_file.file.read())

    # return a path that will be served at /uploads/<name>
    return f"uploads/{dest_name}"

def extract_text_from_upload(upload_file) -> Tuple[str, str]:
    """Extract text from DOCX/PDF, fallback to OCR."""
    name = getattr(upload_file, "filename", "")
    ext = get_extension(name)
    if ext not in ALLOWED_EXTS:
        raise ValueError("Only .pdf and .docx are supported")

    upload_file.file.seek(0)
    data = upload_file.file.read()

    if ext == ".docx":
        try:
            doc = Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
            return text, ext
        except Exception:
            return "", ext

    if ext == ".pdf":
        try:
            text = extract_text(io.BytesIO(data))
            if text.strip():
                return text, ext
        except Exception:
            pass
        # OCR fallback
        try:
            images = convert_from_bytes(data)
            text = "\n".join([pytesseract.image_to_string(img) for img in images])
            return text, ext
        except Exception:
            return "", ext

    return "", ext

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

def extract_keywords(text: str):
    text = clean_text(text)
    words = set(text.split())
    return words - STOP_WORDS

def check_cv_format(text: str):
    warnings = []
    sections = ["experience", "education", "skills", "projects"]
    for s in sections:
        if s not in text.lower():
            warnings.append(f"Section '{s}' is missing")
    if not re.search(r"\b[\w.-]+@[\w.-]+\.\w{2,4}\b", text):
        warnings.append("No valid email found")
    if not re.search(r"\+?\d[\d\s-]{7,}\d", text):
        warnings.append("No valid phone number found")
    bullets = len(re.findall(r"[\u2022\-*]", text))
    if bullets < 3:
        warnings.append("Few bullet points; consider using bullets for clarity")
    return warnings

def semantic_match(resume_keywords, jd_keywords):
    matched = set()
    missing = set(jd_keywords)

    if not resume_keywords:
        return matched, missing

    resume_doc = nlp(" ".join(resume_keywords))
    # filter out empty tokens
    resume_tokens = [t for t in resume_doc if (hasattr(t, "orth_") and t.orth_.strip())]

    if not resume_tokens:
        return matched, missing

    for word in jd_keywords:
        word_doc = nlp(word)
        sims = []
        for token in resume_tokens:
            try:
                sims.append(word_doc.similarity(token))
            except Exception:
                continue
        max_sim = max(sims) if sims else 0
        if max_sim >= 0.75:
            matched.add(word)
            missing.discard(word)
    return matched, missing

def compute_score(resume_text: str, job_description: str):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    matched, missing = semantic_match(resume_keywords, jd_keywords)

    keyword_score = int(len(matched) / len(jd_keywords) * 100) if jd_keywords else 0
    warnings = check_cv_format(resume_text)
    format_score = max(0, 100 - len(warnings) * 10)
    final_score = int((keyword_score + format_score) / 2)

    # Suggestion
    suggestion = []
    if warnings:
        suggestion.append("Fix formatting issues")
    if missing:
        suggestion.append("Add missing important skills/keywords")
    if final_score < 60:
        suggestion.append("CV may not meet job requirements")

    return {
        "score": final_score,
        "keyword_score": keyword_score,
        "format_score": format_score,
        "matched": list(matched),
        "missing": list(missing),
        "warnings": warnings,
        "suggestion": suggestion,
        "cv_text": resume_text[:5000]  # limit to first 5000 chars
    }

# -----------------------
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, id: Optional[str] = None):
    result = None
    if id:
        result = RESULTS_CACHE.pop(id, None)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.post("/check", response_class=HTMLResponse)
async def check_cv(request: Request, resume: UploadFile = File(...), job_description: str = Form(...)):
    try:
        # Save the file
        file_path = save_upload_file(resume)

        # Extract text for analysis
        resume_text, ext = extract_text_from_upload(resume)

        result = compute_score(resume_text, job_description)

        # Add CV file path for preview — points to mounted /uploads
        result['cv_file_url'] = f"/{file_path}"

        rid = uuid.uuid4().hex
        RESULTS_CACHE[rid] = result
        return RedirectResponse(url=f"/?id={rid}", status_code=303)
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})
