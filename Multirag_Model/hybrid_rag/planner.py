"""Query planning with structured output (Pydantic schema)."""
from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from pydantic import BaseModel, Field

from hybrid_rag import config


class QueryPlan(BaseModel):
    """What the planner LLM must return - validated by Pydantic."""
    standalone_question: str = Field(description="Follow-up rewritten as a self-contained question")
    entities: list[str] = Field(default_factory=list, description="Named entities in the question")
    category_filter: Optional[str] = Field(default=None, description="Category to filter on, or null")
    needs_graph: bool = Field(default=False, description="True if relationships between entities matter")
    keywords: list[str] = Field(default_factory=list, description="Important search terms")


def format_history(history: list[dict], turns: int) -> str:
    """Render the last `turns` messages as 'role: content' lines."""
    recent = history[-turns:] if turns else []
    return "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent) or "(none)"


def fallback_plan(question: str) -> QueryPlan:
    """Used when no LLM is available or the planner call fails."""
    return QueryPlan(standalone_question=question, keywords=config.tokenize(question)[:8])


@traceable(name="plan_query", run_type="chain")
def plan_query(question: str, history: list[dict], llm, categories: list[str] | None = None,
               settings: config.Settings | None = None) -> QueryPlan:
    """Rewrite the question using chat history, extract entities and pick a
    metadata filter. The filter is dropped if it is not a known category, so a
    hallucinated filter can never hide all results."""
    settings = settings or config.get_settings()
    categories = sorted({c for c in (categories or []) if c})
    if llm is None:
        return fallback_plan(question)
    system = config.PLANNER_PROMPT.format(category_field=settings.category_field,
                                          categories=", ".join(categories[:50]) or "(none)")
    user = f"CHAT HISTORY:\n{format_history(history, settings.history_turns)}\n\nQUESTION: {question}"
    try:
        planner = llm.with_structured_output(QueryPlan)
        plan: QueryPlan = planner.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception:
        return fallback_plan(question)
    if plan.category_filter and plan.category_filter.lower() not in {c.lower() for c in categories}:
        plan.category_filter = None
    if not plan.standalone_question.strip():
        plan.standalone_question = question
    return plan
