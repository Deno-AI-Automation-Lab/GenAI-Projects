"""The support desk team: SelectorGroupChat with deterministic routing + loop guards.

Flow for every question:
    user → router (LLM picks department) → specialist (searches, writes 5 bullets)
         → quality_observer (APPROVED / REVISE) → [one revision] → done
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    FunctionalTermination,
    MaxMessageTermination,
    TimeoutTermination,
    TokenUsageTermination,
)
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.models import ChatCompletionClient

from agents import DEPARTMENTS, OBSERVER_NAME, ROUTER_NAME, SPECIALIST_NAMES, build_agents
from config import Settings
from memory_store import ConversationMemory
from observability import TraceObserver
from search_tool import EVIDENCE

_ROUTE_RE = re.compile(r"ROUTE:\s*([a-z_]+)", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-•*]\s+")
_QUESTION_RE = re.compile(r"QUESTION:\s*(.+)", re.IGNORECASE)
_CITE_RE = re.compile(r"\s*\[((?:S\d+(?:\s*[-–]\s*S?\d+)?)(?:\s*,\s*S\d+(?:\s*[-–]\s*S?\d+)?)*|none)\]",
                      re.IGNORECASE)


def _expand_cites(group: str) -> list[str]:
    """'S1, S3-S5' -> ['S1', 'S3', 'S4', 'S5']"""
    out: list[str] = []
    for part in group.split(","):
        part = part.strip().upper()
        if not part or part == "NONE":
            continue
        m = re.match(r"S(\d+)\s*[-–]\s*S?(\d+)", part)
        out += [f"S{n}" for n in range(int(m.group(1)), int(m.group(2)) + 1)] if m else [part]
    return out

Message = BaseAgentEvent | BaseChatMessage


@dataclass
class Answer:
    text: str
    agent: str
    approved: bool
    status: str
    resolved_question: str
    sources: list[str]
    revisions: int
    stop_reason: str
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int


def keyword_route(question: str) -> str | None:
    """Fallback classifier if the router's reply can't be parsed."""
    q = question.lower()
    scores = {d.agent_name: sum(k in q for k in d.keywords) for d in DEPARTMENTS}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def _chat_only(messages: Sequence[Message]) -> list[BaseChatMessage]:
    return [m for m in messages if isinstance(m, BaseChatMessage)]


def _first_task(messages: Sequence[BaseChatMessage]) -> str:
    return next((m.to_text() for m in messages if m.source == "user"), "")


def make_selector(messages: Sequence[Message]) -> str | None:
    """Pick the next speaker without an extra LLM call (cheap, predictable, loop-proof)."""
    chat = _chat_only(messages)
    if not chat:
        return ROUTER_NAME
    last = chat[-1]

    if last.source == "user":
        return ROUTER_NAME

    if last.source == ROUTER_NAME:
        match = _ROUTE_RE.search(last.to_text())
        if match and match.group(1).lower() in SPECIALIST_NAMES:
            return match.group(1).lower()
        return keyword_route(_first_task(chat)) or SPECIALIST_NAMES[0]

    if last.source in SPECIALIST_NAMES:
        return OBSERVER_NAME

    if last.source == OBSERVER_NAME:  # REVISE → back to the same specialist
        return next((m.source for m in reversed(chat) if m.source in SPECIALIST_NAMES), ROUTER_NAME)

    return ROUTER_NAME


def _approved(messages: Sequence[Message]) -> bool:
    return any(
        isinstance(m, TextMessage)
        and m.source == OBSERVER_NAME
        and m.content.strip().upper().startswith("APPROVED")
        for m in messages
    )


def _router_said_out_of_scope(messages: Sequence[Message]) -> bool:
    return any(
        isinstance(m, TextMessage) and m.source == ROUTER_NAME and "OUT_OF_SCOPE" in m.content.upper()
        for m in messages
    )


def clean_bullets(text: str, limit: int = 5) -> tuple[str, list[str]]:
    """Normalise the reply to at most 5 '- ' bullets; strip [S#] citations and return them separately."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = [_BULLET_RE.sub("", ln) for ln in lines if _BULLET_RE.match(ln)] or lines
    cited: list[str] = []
    out = []
    for b in bullets[:limit]:
        for m in _CITE_RE.finditer(b):
            cited += _expand_cites(m.group(1))
        out.append(f"- {_CITE_RE.sub('', b).strip()}")
    return "\n".join(out), list(dict.fromkeys(cited))


class SupportDesk:
    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: ConversationMemory,
        settings: Settings,
        observer: TraceObserver,
    ) -> None:
        self.memory = memory
        self.observer = observer
        self._question = ""
        self._revise_count = 0
        self.max_revisions = settings.max_revisions
        router, specialists, quality_observer = build_agents(model_client, memory)

        termination = (
            FunctionalTermination(_approved)                        # observer signed off
            | FunctionalTermination(self._out_of_scope)             # nothing to route
            | FunctionalTermination(self._revision_limit)           # observer can't loop forever
            | MaxMessageTermination(settings.max_messages)          # caps revise loops
            | TimeoutTermination(settings.timeout_seconds)          # caps slow runs
            | TokenUsageTermination(max_total_token=settings.max_tokens_per_query)  # caps cost
        )
        self.team = SelectorGroupChat(
            participants=[router, *specialists, quality_observer],
            model_client=model_client,  # only used if selector_func returned None (it never does)
            selector_func=make_selector,
            termination_condition=termination,
            allow_repeated_speaker=False,
        )

    def _out_of_scope(self, messages: Sequence[Message]) -> bool:
        """Stop only if the router rejected it AND no department keyword matches.

        Otherwise the selector overrides the router and sends it to the matching specialist.
        """
        return _router_said_out_of_scope(messages) and keyword_route(self._question) is None

    def _revision_limit(self, messages: Sequence[Message]) -> bool:
        """Stop once the observer asks for more than `max_revisions` revisions; keep the latest answer."""
        self._revise_count += sum(
            1 for m in messages
            if isinstance(m, TextMessage) and m.source == OBSERVER_NAME
            and m.content.strip().upper().startswith("REVISE")
        )
        return self._revise_count > self.max_revisions

    async def ask(self, question: str) -> Answer:
        run_id = uuid.uuid4().hex[:8]
        self._question = question
        self._revise_count = 0
        EVIDENCE.reset()
        # Date is added per question (not baked into prompts) so "now" is always correct.
        task = f"Today is {datetime.now():%A, %d %B %Y}.\nCustomer question: {question}"
        self.observer.reset()
        await self.team.reset()  # fresh chat each question; continuity comes from memory
        start = time.perf_counter()

        result: TaskResult | None = None
        async for item in self.team.run_stream(task=task):
            if isinstance(item, TaskResult):
                result = item
            else:
                self.observer.on_event(run_id, item)

        assert result is not None
        chat = _chat_only(result.messages)
        revisions = sum(
            1 for m in chat if m.source == OBSERVER_NAME and m.to_text().strip().upper().startswith("REVISE")
        )
        specialist_msgs = [m for m in chat if m.source in SPECIALIST_NAMES and isinstance(m, TextMessage)]

        router_text = next((m.to_text() for m in chat if m.source == ROUTER_NAME), "")
        q_match = _QUESTION_RE.search(router_text)
        resolved = q_match.group(1).strip() if q_match else question

        sources: list[str] = []
        if specialist_msgs:
            agent = specialist_msgs[-1].source
            text, cited = clean_bullets(specialist_msgs[-1].content)
            sources = [u for u in (EVIDENCE.url(c) for c in cited) if u]
        else:
            agent = ROUTER_NAME
            router_msg = next((m.to_text() for m in chat if m.source == ROUTER_NAME), "")
            text = router_msg.split(":", 1)[-1].strip() or "Sorry, I couldn't process that request."
        await self.memory.remember_turn(question, agent, text, resolved=resolved)  # every turn is saved

        approved = _approved(result.messages)
        if approved:
            status = "approved"
        elif agent == ROUTER_NAME:
            status = "out of scope"
        elif revisions > self.max_revisions:
            status = "best effort (revision limit reached)"
        else:
            status = f"stopped: {result.stop_reason}"

        answer = Answer(
            text=text,
            agent=agent,
            approved=approved,
            status=status,
            resolved_question=resolved,
            sources=sources,
            revisions=revisions,
            stop_reason=result.stop_reason or "",
            elapsed_s=round(time.perf_counter() - start, 2),
            prompt_tokens=self.observer.prompt_tokens,
            completion_tokens=self.observer.completion_tokens,
            tool_calls=self.observer.tool_calls,
        )
        self.observer.on_finish(run_id, {"question": question, **answer.__dict__})
        return answer
