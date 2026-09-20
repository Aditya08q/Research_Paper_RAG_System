from pathlib import Path

import fitz 
from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
   


def load_pdf(file_path: Path) -> list[Document]:
    
    try:
        doc = fitz.open(file_path)
    except Exception as exc:  
        logger.error("Failed to open PDF '%s': %s", file_path.name, exc)
        raise PDFLoadError(f"Could not open '{file_path.name}'. It may be corrupted.") from exc

    if doc.is_encrypted:
        doc.close()
        logger.error("PDF '%s' is password-protected", file_path.name)
        raise PDFLoadError(f"'{file_path.name}' is password-protected and cannot be read.")

    pages: list[Document] = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:  
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
