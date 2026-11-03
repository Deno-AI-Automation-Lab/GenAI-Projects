"""LLM / embedding factories and runtime bootstrap (TLS, .env export, LangSmith).

All chat completions go through LangChain's ChatOpenAI (never the raw OpenAI SDK).
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import PROJECT_ROOT, Settings, get_settings

log = logging.getLogger(__name__)
_BOOTSTRAPPED = False


class LLMUnavailableError(RuntimeError):
    """Raised when no OpenAI key is configured (callers fall back gracefully)."""


def bootstrap_runtime(settings: Settings | None = None) -> None:
    """One-time process setup.

    * .env export: pydantic-settings reads .env into `Settings` only; libraries such as
      LangSmith/LangChain read os.environ directly, so .env is also exported here
      (override=False: real environment variables still win).
    * TLS: use the OS trust store via `truststore` (handles corporate proxies with
      their own root CA) or an explicitly bundled CA file. SSL verification is
      NEVER disabled.
    * LangSmith: enable tracing only when an API key is present, so a missing key
      never floods the logs with 401 errors.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    s = settings or get_settings()
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as exc:  # pragma: no cover - truststore optional at runtime
        log.warning("truststore not active: %s", exc)
    if s.ca_bundle and s.ca_bundle.exists():
        # requests / httpx / openai honour these variables
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(s.ca_bundle))
        os.environ.setdefault("SSL_CERT_FILE", str(s.ca_bundle))

    has_ls_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    if s.langsmith_tracing and has_ls_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", s.langsmith_project)
        log.info("LangSmith tracing ON (project=%s, endpoint=%s)", os.environ["LANGSMITH_PROJECT"],
                 os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))
    else:
        if s.langsmith_tracing:
            log.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing - tracing disabled")
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    _BOOTSTRAPPED = True


def _require_key(s: Settings) -> str:
    key = s.openai_api_key.get_secret_value()
    if not key:
        raise LLMUnavailableError("OPENAI_API_KEY is not set in .env")
    return key


@lru_cache(maxsize=8)
def get_chat_llm(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """Cached ChatOpenAI client with timeouts and retries (LangSmith-traced automatically)."""
    s = get_settings()
    return ChatOpenAI(
        model=model or s.chat_model,
        temperature=temperature,
        api_key=_require_key(s),
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
    )


@lru_cache(maxsize=2)
def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """OpenAI embeddings used for every FAISS index."""
    s = get_settings()
    return OpenAIEmbeddings(
        model=model or s.embedding_model,
        api_key=_require_key(s),
        max_retries=s.llm_max_retries,
        request_timeout=s.llm_timeout_s,
    )
