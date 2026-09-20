from fastapi import APIRouter, HTTPException, UploadFile, File
from app.core.config import settings
from app.core.logging import get_logger
from app.models.response import UploadResponse
from app.services.chunking import chunk_documents
from app.services.pdf_loader import PDFLoadError, load_pdf
from app.services.vector_store import VectorStoreError, add_chunks

logger = get_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a single PDF file, index it into the vector store, and return a
    summary of what was extracted.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.upload_max_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {settings.upload_max_size_mb}MB size limit.",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dest_path = settings.pdf_storage_dir / file.filename
    dest_path.write_bytes(contents)
    logger.info("Saved upload '%s' (%.2f MB)", file.filename, size_mb)

    try:
        pages = load_pdf(dest_path)
        chunks = chunk_documents(pages)
        add_chunks(chunks)
    except PDFLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UploadResponse(
        filename=file.filename,
        pages_extracted=len(pages),
        chunks_created=len(chunks),
    
    )
