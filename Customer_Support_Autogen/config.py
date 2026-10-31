"""Central configuration. Values come from the .env file (or real environment variables)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    # Low temperature = fewer invented facts
    temperature: float = field(default_factory=lambda: float(os.getenv("MODEL_TEMPERATURE", "0.1")))

    # Search
    search_max_results: int = field(default_factory=lambda: _int("SEARCH_MAX_RESULTS", 5))
    search_region: str = field(default_factory=lambda: os.getenv("SEARCH_REGION", "in-en"))
    # DuckDuckGo first; the other ddgs engines are only used if DuckDuckGo returns nothing.
    # Set both to "duckduckgo" to use DuckDuckGo only.
    search_backend_text: str = field(
        default_factory=lambda: os.getenv("SEARCH_BACKEND_TEXT", "duckduckgo,bing,brave,yahoo"))
    search_backend_news: str = field(
        default_factory=lambda: os.getenv("SEARCH_BACKEND_NEWS", "duckduckgo,bing,yahoo"))

    # Loop guards (terminations)
    max_messages: int = field(default_factory=lambda: _int("TEAM_MAX_MESSAGES", 8))
    timeout_seconds: int = field(default_factory=lambda: _int("TEAM_TIMEOUT_SECONDS", 120))
    max_revisions: int = field(default_factory=lambda: _int("MAX_REVISIONS", 1))
    max_tokens_per_query: int = field(default_factory=lambda: _int("TEAM_MAX_TOKENS", 30000))

    # Memory
    memory_file: Path = field(
        default_factory=lambda: Path(os.getenv("MEMORY_FILE", BASE_DIR / "data" / "memory.json"))
    )
    memory_max_entries: int = field(default_factory=lambda: _int("MEMORY_MAX_ENTRIES", 20))

    # Observability
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", BASE_DIR / "logs")))

    def validate(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")


settings = Settings()
