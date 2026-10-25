"""
Document parser for PDF, DOCX, and plain text files.

Design decisions:
- PyMuPDF (fitz) for PDF: fast, no external deps, handles most PDFs well.
- python-docx for DOCX: standard library, extracts paragraphs + tables.
- Plain text / Markdown: read as-is with encoding detection.
- Returns clean text with preserved paragraph structure.
- Does NOT do chunking here — that's the retrieval layer's job.
  This separation keeps ingestion format-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_document(file_path: str | Path) -> str:
    """
    Parse a document file and return clean text.

    Args:
        file_path: Path to PDF, DOCX, TXT, or MD file.

    Returns:
        Extracted text as a single string with paragraph breaks preserved.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix in (".txt", ".md", ".markdown", ".text"):
        return _parse_text(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported: .pdf, .docx, .txt, .md"
        )


def parse_text(text: str) -> str:
    """
    Clean raw text input (for when the user passes text directly
    rather than a file). Normalizes whitespace and strips control chars.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove null bytes and other control chars (except \n, \t)
    text = "".join(
        c for c in text if c == "\n" or c == "\t" or (ord(c) >= 32)
    )
    return text.strip()


# ─── Format-specific parsers ────────────────────────────────────────────────


def _parse_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())
        else:
            logger.warning("PDF page %d is empty or image-only", page_num + 1)
    doc.close()

    if not pages:
        logger.warning("No text extracted from PDF: %s", path.name)
        return ""

    return "\n\n".join(pages)


def _parse_docx(path: Path) -> str:
    """Extract text from DOCX including paragraphs and table cells."""
    from docx import Document

    doc = Document(str(path))
    parts = []

    # Extract paragraphs
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)

    # Extract table contents (often RFP requirements are in tables)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)

    return "\n\n".join(parts)


def _parse_text(path: Path) -> str:
    """Read plain text / markdown file."""
    # Try UTF-8 first, fall back to latin-1
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            return parse_text(text)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode text file: {path}")
