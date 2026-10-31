"""Watches every event the team produces: live console view + JSONL audit trail."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    MemoryQueryEvent,
    SelectSpeakerEvent,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)

_ICONS = {"router": "🧭", "quality_observer": "🔍", "user": "🙋"}


class TraceObserver:
    """Records how the agents work: who spoke, which tool ran, tokens used."""

    def __init__(self, log_dir: Path, verbose: bool = True) -> None:
        self.verbose = verbose
        log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = log_dir / f"trace-{datetime.now():%Y%m%d}.jsonl"
        self.logger = logging.getLogger("support_bot.trace")
        self.reset()

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tool_calls = 0
        self.speakers: list[str] = []

    def _write(self, record: dict[str, Any]) -> None:
        with self.trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _say(self, text: str) -> None:
        if self.verbose:
            print(f"   \033[2m{text}\033[0m")

    def on_event(self, run_id: str, item: BaseAgentEvent | BaseChatMessage) -> None:
        usage = getattr(item, "models_usage", None)
        if usage:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens

        source = getattr(item, "source", "?")
        icon = _ICONS.get(source, "🤖")
        if isinstance(item, ToolCallRequestEvent):
            self.tool_calls += sum(1 for c in item.content if c.name in ("web_search", "get_weather"))
            for call in item.content:
                self._say(f"{icon} {source} → tool {call.name}({call.arguments})")
        elif isinstance(item, ToolCallExecutionEvent):
            for res in item.content:
                status = "error" if res.is_error else f"{len(res.content)} chars"
                self._say(f"{icon} {source} ← tool result ({status})")
        elif isinstance(item, MemoryQueryEvent):
            self._say(f"{icon} {source} loaded {len(item.content)} memory item(s)")
        elif isinstance(item, SelectSpeakerEvent):
            pass
        elif isinstance(item, BaseChatMessage) and source != "user":
            self.speakers.append(source)
            preview = item.to_text().replace("\n", " ")[:110]
            self._say(f"{icon} {source}: {preview}")

        self._write(
            {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "run_id": run_id,
                "type": type(item).__name__,
                "source": source,
                "content": item.to_text()[:2000],
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            }
        )

    def on_finish(self, run_id: str, summary: dict[str, Any]) -> None:
        self._write({"ts": datetime.now().isoformat(timespec="milliseconds"), "run_id": run_id,
                     "type": "RunSummary", **summary})
        self.logger.info("run %s finished: %s", run_id, summary)
