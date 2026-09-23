"""
utils/resume_parser.py
----------------------
PDF text extraction entry point for Flask file upload objects.
Delegates to the richer utils/resume.py engine.
"""

from utils.resume import extract_text as _extract


def extract_text(pdf_file) -> str:
    """
    Accept a Flask FileStorage object and return extracted plain text (lowercased).
    Reads the file bytes once and passes them to the PyMuPDF / pdfminer engine.
    """
    file_bytes = pdf_file.read()
    text = _extract(file_bytes)
    return text.lower()
