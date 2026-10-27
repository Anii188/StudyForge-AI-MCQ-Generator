"""Extract text from common study-file formats."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path


def extract_uploaded_text(filename: str, file_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return file_bytes.decode("utf-8", errors="replace").strip()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix == ".docx":
        from docx import Document
        document = Document(BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()).strip()
    raise ValueError("Unsupported file type. Upload a .txt, .pdf, or .docx file.")
