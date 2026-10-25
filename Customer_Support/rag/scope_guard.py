"""Scope guard: decide, from retrieved handbook passages only, whether the question is covered.

Runs before the CrewAI crew so out-of-scope questions (billing, pricing, vendor-specific device
setup, unrelated topics) get an honest "not covered" reply instead of a forced troubleshooting plan.
"""
from typing import List, Literal, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from crew.schemas import SupportResponse


class ScopeDecision(BaseModel):
    verdict: Literal["in_scope", "partial", "out_of_scope"] = Field(
        ..., description="in_scope: excerpts contain the needed guidance. partial: excerpts touch the "
                         "topic only generally, not the specific how-to asked. out_of_scope: not addressed.")
    reason: str = Field(..., description="One sentence explaining the verdict.")
    covered_points: List[str] = Field(
        default_factory=list, description="For partial: what the handbook DOES say that is relevant.")
    suggested_route: str = Field(
        "", description="For partial/out_of_scope: where the customer should go instead, "
                        "e.g. billing team, device vendor support, router manual.")


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict scope checker for a FiberNet Broadband support assistant whose ONLY knowledge "
     "source is the troubleshooting handbook. Judge only from the HANDBOOK EXCERPTS below; do not use "
     "outside knowledge.\n"
     "- in_scope: the excerpts contain troubleshooting/support guidance that addresses the question.\n"
     "- partial: the excerpts mention the topic generally but not the specific procedure or fact asked "
     "(e.g. they list checks for a device type but not its configuration steps).\n"
     "- out_of_scope: the excerpts do not address it (billing, refunds, pricing, plan upgrades, SLA hours, "
     "vendor-specific configuration, unrelated topics).\n"
     "If the new message is clearly a follow-up in an ongoing in-scope conversation (e.g. 'still not "
     "working', 'the LOS light is off now'), treat it as in_scope."),
    ("human",
     "EARLIER CONVERSATION (context only):\n{history}\n\n"
     "NEW CUSTOMER MESSAGE:\n{question}\n\n"
     "HANDBOOK EXCERPTS:\n{context}"),
])


def check_scope(vectorstore: FAISS, question: str, history: str = "None",
                k: int = 4) -> Tuple[ScopeDecision, list]:
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page')} | {d.metadata.get('section')}]\n{d.page_content}" for d in docs)
    llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY, temperature=0)
    decision = (_PROMPT | llm.with_structured_output(ScopeDecision)).invoke(
        {"question": question, "history": history, "context": context})
    return decision, docs


def out_of_scope_response(decision: ScopeDecision) -> SupportResponse:
    route = decision.suggested_route or "the relevant FiberNet team"
    reply = (
        "Thanks for reaching out. I've checked our broadband troubleshooting guide, and it doesn't "
        f"cover this request. {decision.reason}\n\n"
        f"The best next step is to contact **{route}**, who can help with this directly.\n\n"
        "If you're also having a connection problem (no internet, slow speed, Wi-Fi drops, red lights "
        "on the fiber box), tell me what you're seeing and I'll help troubleshoot it."
    )
    return SupportResponse(
        customer_reply=reply,
        fault_domain="Out of scope",
        priority="P4",
        ticket_tags=["OUT-OF-SCOPE"],
        case_note=f"Problem: request not covered by troubleshooting handbook.\n"
                  f"Reason: {decision.reason}\nNext step: route to {route}.",
    )


def scope_note(decision: ScopeDecision) -> str:
    """Instruction passed to the crew for in-scope / partial questions."""
    if decision.verdict == "partial":
        covered = "; ".join(decision.covered_points) or "general related checks only"
        return (
            "PARTIAL COVERAGE: the handbook only covers this generally "
            f"({covered}). Give ONLY what the handbook says, state clearly that detailed "
            f"steps for this request are not in the FiberNet guide, and route the customer to "
            f"{decision.suggested_route or 'the appropriate vendor/team'}. Do NOT invent "
            "configuration steps and do NOT switch to an unrelated troubleshooting workflow."
        )
    if decision.verdict == "out_of_scope":
        return (f"NOT IN HANDBOOK: {decision.reason} Suggested route if the web can't help: "
                f"{decision.suggested_route or 'the relevant FiberNet team'}.")
    return "IN SCOPE: the handbook covers this request."
