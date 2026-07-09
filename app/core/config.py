"""
Application configuration.

All tunable values live here so no other module hardcodes a "magic number" —
this satisfies the project's `avoid_magic_numbers` coding standard and gives
a single place to tweak behavior (e.g. chunk size, top_k) while learning.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """
    Typed application settings, loaded from environment variables (or a .env
    file) with sensible defaults for local development.
    """

    # --- Paths -------------------------------------------------------------
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    pdf_storage_dir: Path = base_dir / "data" / "pdfs"
    chroma_persist_dir: Path = base_dir / "data" / "chroma_db"

    # --- Chunking ------------------------------------------------------------
    chunk_size: int = 500
    chunk_overlap: int = 100

    # --- Embeddings ----------------------------------------------------------
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # --- Vector store ----------------------------------------------------------
    chroma_collection_name: str = "research_papers"
    retrieval_top_k: int = 5

    # --- Groq LLM --------------------------------------------------------
    # Groq exposes an OpenAI-compatible chat completions endpoint, so we
    # reuse the `openai` Python client pointed at their base_url instead of
    # a Groq-specific SDK.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.2
    groq_max_tokens: int = 1024

    # --- Server ----------------------------------------------------------------
    upload_max_size_mb: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single shared instance — imported everywhere instead of re-instantiating
# Settings(), which keeps configuration centralized (avoid_global_variables
# is respected because this is an explicit, typed singleton, not a loose
# module-level dict).
settings = Settings()

# Ensure required directories exist at import time so services never have to
# check for this themselves.
settings.pdf_storage_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
