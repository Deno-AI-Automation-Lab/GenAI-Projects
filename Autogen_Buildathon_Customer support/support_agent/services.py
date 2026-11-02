"""Long-lived service container shared by the API, mail worker, scripts and tests."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock

from .config import Settings, get_settings
from .memory import MemoryStore
from .planner import RouterIndex, load_router
from .retrieval import IndexRegistry
from .sheets import TicketLogger

log = logging.getLogger(__name__)


@dataclass
class SupportServices:
    settings: Settings
    memory: MemoryStore
    registry: IndexRegistry
    logger: TicketLogger
    _router: RouterIndex | None = None
    _lock: Lock = field(default_factory=Lock)

    @property
    def router(self) -> RouterIndex:
        """Lazy-load the router; an empty router still works (product = unknown → escalate)."""
        with self._lock:
            if self._router is None:
                try:
                    self._router = load_router(self.settings)
                except Exception as exc:
                    log.error("Router index unavailable: %s", exc)
                    self._router = RouterIndex(store=None, products=[])
            return self._router

    def reload_indexes(self) -> None:
        with self._lock:
            self._router = None
        self.registry.clear()


def build_services(settings: Settings | None = None) -> SupportServices:
    s = settings or get_settings()
    return SupportServices(settings=s, memory=MemoryStore(s.db_path), registry=IndexRegistry(s),
                           logger=TicketLogger(s))
