# ATS CV Score Checker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

A **professional, open-source web-based ATS (Applicant Tracking System) CV/Resume checker** built with **FastAPI** and **Bootstrap 5**. This application analyzes resumes against a job description and provides a **fit score**, highlighting **matched and missing keywords**, as well as **formatting warnings**, to help candidates optimize their CVs for modern ATS systems.

---

## Features

* **Keyword Matching**: Compares resume keywords with job description keywords.
* **Stopwords Ignored**: Common words like "and", "for", "the" are excluded for precise scoring.
* **Semantic Matching**: Uses **spaCy embeddings** to match similar terms (e.g., "Software Developer" ≈ "Software Engineer").
* **Section Weighting**: Skills & Experience sections weighted higher for scoring.
* **CV Format Check**: Detects missing sections, email, phone number, bullet points, and inconsistent formatting.
* **PDF & DOCX Support**: Extracts text from uploaded PDFs and DOCX files.
* **OCR for Scanned PDFs**: Uses **pytesseract** and **pdf2image** to extract text from scanned documents.
* **Professional Scoring**: Returns keyword score, format score, and overall ATS fit score.
* **Dashboard View**: Displays results with matched/missing keywords and formatting warnings in a clean UI.

---

## Tech Stack

* **Backend**: FastAPI
* **Frontend**: Jinja2 + Bootstrap 5
* **NLP**: spaCy, NLTK (stopwords)
* **PDF/DOCX Extraction**: pdfminer.six, python-docx
* **OCR**: pytesseract, pdf2image
* **Deployment**: Can run locally or on cloud servers with Uvicorn

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/sydulamin/ats-cv-score-checker.git
cd ats-cv-score-checker
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the FastAPI server**:

```bash
uvicorn main:app --reload
```

4. **Open in browser**:
   Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and start checking resumes.

---

## Usage

1. Upload your resume (PDF or DOCX).
2. Paste the job description.
3. Click **Check ATS Score**.
4. View the results:

   * Final Score
   * Keyword Score
   * Format Score
   * Matched / Missing Keywords
   * Formatting Warnings

---

## Contribution

This project is open-source, and contributions are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit your changes (`git commit -m "Add feature"`)
5. Push to the branch (`git push origin feature-name`)
6. Open a Pull Request

### Contributions may include:

* Adding new features (e.g., advanced analytics, multi-language support)
* Bug fixes and optimizations
* Improving UI/UX and dashboard visuals
* Enhancing NLP matching and scoring logic

---

## Future Improvements

* Interactive analytics charts for keyword coverage and fit trends.
* Advanced NLP matching using contextual embeddings (BERT, Sentence Transformers).
* Ranking multiple candidates automatically.
* Multi-language resume support.
* More robust PDF/scan parsing with OCR enhancements.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Feel free to use, modify, and distribute it under the open-source license.

---

## Acknowledgements

* FastAPI
* spaCy
* NLTK
* pdfminer.six
* python-docx
* pytesseract
* Bootstrap

---

*If you want, you can also create a ready-to-use `requirements.txt` for all dependencies so anyone can just `pip install -r requirements.txt` and run the project immediately.*
