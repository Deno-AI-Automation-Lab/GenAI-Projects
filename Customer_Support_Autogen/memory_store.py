"""Conversation memory shared by the router and the specialists.

A custom AutoGen `Memory`: before an agent replies, `update_context` injects
  * long-term user facts (saved with /remember), and
  * the most recent conversation turns, oldest first,
so follow-ups like "what about its box office?" keep their context.
Every turn is saved (including out-of-scope ones) to a JSON file, so memory
survives restarts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from autogen_core import CancellationToken
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage

logger = logging.getLogger("support_bot.memory")


class ConversationMemory(Memory):
    def __init__(self, path: Path, max_entries: int = 20, context_turns: int = 6) -> None:
        self.path = Path(path)
        self.max_entries = max_entries      # turns kept on disk
        self.context_turns = context_turns  # turns shown to the agents
        self.facts: list[str] = []
        self.turns: list[dict[str, str]] = []
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read memory file %s: %s", self.path, exc)
            return
        if isinstance(data, list):  # file from the previous version
            for row in data:
                text = row.get("content", "")
                if text.startswith("User fact: "):
                    self.facts.append(text.removeprefix("User fact: "))
                elif "\n" in text and "answered:" in text:
                    head, answer = text.split("\n", 1)
                    agent, _, answer = answer.partition(" answered:\n")
                    self.turns.append({"time": row.get("metadata", {}).get("time", ""),
                                       "question": head.removeprefix("Earlier the user asked: "),
                                       "agent": agent, "answer": answer})
            self._save()
        else:
            self.facts = list(data.get("facts", []))
            self.turns = list(data.get("turns", []))[-self.max_entries :]
        logger.info("Loaded %d facts and %d turns from %s", len(self.facts), len(self.turns), self.path)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"facts": self.facts, "turns": self.turns}, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ---- helpers used by the app ---------------------------------------
    async def remember_turn(self, question: str, agent: str, answer: str, resolved: str = "") -> None:
        """Store one finished Q&A turn so follow-up questions have context."""
        self.turns.append({"time": datetime.now().isoformat(timespec="seconds"),
                           "question": question, "resolved": resolved or question,
                           "agent": agent, "answer": answer})
        self.turns = self.turns[-self.max_entries :]
        self._save()

    async def remember_fact(self, fact: str) -> None:
        """Store a user preference or fact (e.g. 'I am vegetarian', 'I live in Chennai')."""
        fact = fact.strip()
        if fact and fact not in self.facts:
            self.facts.append(fact)
            self._save()

    def history_text(self, n: int | None = None) -> str:
        turns = self.turns[-(n or self.context_turns) :]
        return "\n\n".join(f"[{t['time'][:16]}] You: {t['question']}\n({t['agent']})\n{t['answer']}" for t in turns)

    def __len__(self) -> int:
        return len(self.facts) + len(self.turns)

    # ---- AutoGen Memory protocol ---------------------------------------
    def _render(self) -> str:
        parts = []
        if self.facts:
            parts.append("Known facts about the customer:\n" + "\n".join(f"- {f}" for f in self.facts))
        recent = self.turns[-self.context_turns :]
        if recent:
            def fmt(t: dict[str, str]) -> str:
                q = t.get("resolved") or t["question"]
                return f"Customer asked: {q}\n{t['agent']} replied:\n{t['answer']}"

            earlier = recent[:-1]
            block = []
            if earlier:
                block.append("Earlier exchanges (oldest first):\n" + "\n\n".join(fmt(t) for t in earlier))
            block.append("MOST RECENT exchange - words like 'it', 'this movie', 'there' usually refer to this:\n"
                         + fmt(recent[-1]))
            parts.append(
                "Conversation history with this customer. Use it ONLY to understand what the customer is "
                "referring to. Past replies may contain mistakes: never reuse their facts as sources.\n\n"
                + "\n\n".join(block)
            )
        return "\n\n".join(parts)

    def _as_contents(self) -> list[MemoryContent]:
        items = [MemoryContent(content=f"fact: {f}", mime_type=MemoryMimeType.TEXT) for f in self.facts]
        items += [MemoryContent(content=f"turn: {t['question']}", mime_type=MemoryMimeType.TEXT)
                  for t in self.turns[-self.context_turns :]]
        return items

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        text = self._render()
        if text:
            await model_context.add_message(SystemMessage(content=text))
        return UpdateContextResult(memories=MemoryQueryResult(results=self._as_contents()))

    async def query(self, query: str | MemoryContent, cancellation_token: CancellationToken | None = None,
                    **kwargs: Any) -> MemoryQueryResult:
        return MemoryQueryResult(results=self._as_contents())

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        await self.remember_fact(str(content.content))

    async def clear(self) -> None:
        self.facts.clear()
        self.turns.clear()
        self._save()

    async def close(self) -> None:
        self._save()
