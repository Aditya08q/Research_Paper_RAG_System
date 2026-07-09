"""
PDF loading and text extraction.

Design choice: we extract page-by-page (not the whole document as one blob)
because we want to preserve page numbers as metadata. That metadata later
flows all the way through to the citation the user sees — "this answer came
from page 4 of paper.pdf" — which would be impossible if we flattened all
pages into a single string up front.
"""

from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be opened or contains no extractable text."""


def load_pdf(file_path: Path) -> list[Document]:
    """
    Extract text from a PDF, one page at a time, and wrap each page in a
    LangChain Document so metadata (source filename + page number) travels
    alongside the text through the rest of the pipeline.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        A list of Document objects, one per non-empty page.

    Raises:
        PDFLoadError: if the file is corrupted, encrypted, or has no
            extractable text at all (e.g. a scanned image PDF with no OCR,
            which this project intentionally does not support per the
            `ocr: false` constraint).
    """
    try:
        doc = fitz.open(file_path)
    except Exception as exc:  # PyMuPDF raises its own generic exceptions
        logger.error("Failed to open PDF '%s': %s", file_path.name, exc)
        raise PDFLoadError(f"Could not open '{file_path.name}'. It may be corrupted.") from exc

    if doc.is_encrypted:
        doc.close()
        logger.error("PDF '%s' is password-protected", file_path.name)
        raise PDFLoadError(f"'{file_path.name}' is password-protected and cannot be read.")

    pages: list[Document] = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:  # skip blank pages (common in scanned/converted PDFs)
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name, "page": page_number},
                )
            )

    doc.close()

    if not pages:
        logger.error("PDF '%s' contains no extractable text", file_path.name)
        raise PDFLoadError(
            f"'{file_path.name}' has no extractable text. "
            "It may be a scanned image PDF, which this project does not support."
        )

    logger.info("Extracted %d non-empty pages from '%s'", len(pages), file_path.name)
    return pages
