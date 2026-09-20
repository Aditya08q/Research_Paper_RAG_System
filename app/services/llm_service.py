import time

from openai import OpenAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMServiceError(Exception):
    """Raised when the Groq API call fails or returns an unusable response."""


class LLMService:
    """Thin wrapper around the Groq chat completions endpoint."""

    def __init__(self) -> None:
        if not settings.groq_api_key:
            logger.error("GROQ_API_KEY is not set")
            raise LLMServiceError(
                "Groq API key is missing. Set GROQ_API_KEY in your .env file."
            )
        self._client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )

    def generate(self, prompt: str) -> str:
        """
        Send a fully-constructed prompt to Groq and return its text response.

        Args:
            prompt: The complete prompt (context + question), already built
                by rag_pipeline.py using the PromptTemplate.

        Returns:
            The model's answer as plain text.

        Raises:
            LLMServiceError: if the API call fails for any reason (network,
                auth, rate limit, malformed response).
        """
        start = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
            )
        except Exception as exc:
            logger.error("Groq API call failed: %s", exc)
            raise LLMServiceError("The language model request failed.") from exc
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info("Groq call latency: %.1f ms", elapsed_ms)

        if not response.choices:
            raise LLMServiceError("Groq returned an empty response.")

        return response.choices[0].message.content or ""
