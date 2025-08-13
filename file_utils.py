from typing import Tuple
import io
from docx import Document
from pdfminer.high_level import extract_text

ALLOWED_EXTS = ['.pdf', '.docx']

def get_extension(filename: str) -> str:
    return '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

def extract_text_from_upload(upload_file) -> Tuple[str, str]:
    """Return (text, ext). Always returns a tuple, never None."""
    try:
        name = getattr(upload_file, 'filename', '')
        ext = get_extension(name)
        print(f"extract_text_from_upload: filename={name}, ext={ext}")

        if ext not in ALLOWED_EXTS:
            raise ValueError('Only .pdf and .docx are supported')

        upload_file.file.seek(0)
        data = upload_file.file.read()

        if ext == '.pdf':
            with io.BytesIO(data) as bio:
                try:
                    text = extract_text(bio)
                    if text is None:
                        text = ""
                except Exception as e:
                    print(f"PDF extraction error: {e}")
                    text = ""
                return text, ext

        if ext == '.docx':
            with io.BytesIO(data) as bio:
                try:
                    doc = Document(bio)
                    text = '\n'.join(p.text for p in doc.paragraphs)
                except Exception as e:
                    print(f"DOCX extraction error: {e}")
                    text = ""
                return text, ext

        # fallback: always return a tuple
        return "", ext

    except Exception as e_outer:
        # If anything fails, return empty text with extension
        print(f"Unexpected error in extract_text_from_upload: {e_outer}")
        return "", ""
