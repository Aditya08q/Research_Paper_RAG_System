from pathlib import Path

from langchain_core.prompts import PromptTemplate

from app.core.config import settings
from app.core.logging import get_logger
from app.models.response import ChatResponse, SourceCitation
from app.services.llm_service import LLMService, LLMServiceError
from app.services.vector_store import VectorStoreError, similarity_search

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_prompt.txt"


class RAGPipelineError(Exception):


def _load_prompt_template() -> PromptTemplate:
    template_text = _PROMPT_PATH.read_text(encoding="utf-8")
    return PromptTemplate(template=template_text, input_variables=["context", "question"])


def _format_context(results: list[tuple]) -> str:
    parts = []
    for doc, _score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(question: str) -> ChatResponse:
    try:
        results = similarity_search(question)
    except VectorStoreError as exc:
        raise RAGPipelineError(str(exc)) from exc

    if not results:
        logger.info("No chunks retrieved for question: %s", question)
        return ChatResponse(
            answer="This information was not found in the uploaded papers.",
            sources=[],
        )

    context = _format_context(results)
    prompt_template = _load_prompt_template()
    full_prompt = prompt_template.format(context=context, question=question)

    try:
        llm = LLMService()
        answer = llm.generate(full_prompt)
    except LLMServiceError as exc:
        raise RAGPipelineError(str(exc)) from exc

    sources = [
        SourceCitation(
            filename=doc.metadata.get("source", "unknown"),
            page_number=doc.metadata.get("page", 0),
            chunk_text=doc.page_content,
            similarity_score=round(score, 4),
        )
        for doc, score in results
    ]

    logger.info("Generated answer for question using %d source chunks", len(sources))
    return ChatResponse(answer=answer, sources=sources)
