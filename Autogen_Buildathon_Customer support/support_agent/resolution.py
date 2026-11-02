"""Resolution + escalation logic used by the Resolution and Log agents' tools."""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .config import Settings
from .guardrails import check_output, strip_citations
from .models import DraftReply, EscalationDecision, QueryAnalysis, RetrievalResult
from .retrieval import ProductIndex, bm25_search, generate_answer, sources_of, vector_search

log = logging.getLogger(__name__)

NO_INFO_EN = ("Thank you for reaching out. I'm sorry – I could not find information about this in our "
              "documentation, so I don't want to guess. A support specialist will review your request "
              "and get back to you shortly.")


def honest_no_info_message(language_name: str, llm=None) -> str:
    """Localised 'we could not find this' message (falls back to English)."""
    if language_name.lower() == "english":
        return NO_INFO_EN
    try:
        from .llm import get_chat_llm

        out = (llm or get_chat_llm()).invoke([
            SystemMessage(content=f"Translate to {language_name}. Output only the translation."),
            HumanMessage(content=NO_INFO_EN)])
        return str(out.content).strip()
    except Exception:
        return NO_INFO_EN


def unsupported_product_message(language_name: str, supported: list[str], mentioned: str = "",
                                llm=None) -> str:
    """Honest reply when the complaint is not about a product we support (no policy facts)."""
    names = ", ".join(supported) if supported else "our listed products"
    about = f" for {mentioned}" if mentioned else ""
    msg = (f"Thank you for contacting us. We provide support only for these products: {names}. "
           f"We could not match your request{about} to one of them, so we cannot share policy or "
           "troubleshooting details for it. If your request is about one of the products above, please reply "
           "with the product name and a short description of the issue. For any other product, please contact "
           "the seller or manufacturer you bought it from.")
    if language_name.lower() == "english":
        return msg
    try:
        from .llm import get_chat_llm

        out = (llm or get_chat_llm()).invoke([
            SystemMessage(content=f"Translate to {language_name}. Keep product names unchanged. "
                                  "Output only the translation."),
            HumanMessage(content=msg)])
        return str(out.content).strip()
    except Exception:
        return msg


def confidence_from_verdict(faithfulness: float, relevance: float, completeness: float) -> float:
    """Weighted confidence, capped by faithfulness: an answer can never be more trusted
    than the share of its claims the documents support (hallucination guard)."""
    weighted = 0.6 * faithfulness + 0.25 * relevance + 0.15 * completeness
    return round(min(weighted, faithfulness), 3)


ORDER_CATEGORIES = ("shipping", "refund", "payment")


@traceable(name="draft_reply")
def draft_reply(analysis: QueryAnalysis, retrieval: RetrievalResult, settings: Settings,
                llm=None, judge_llm=None, order_id: str | None = "n/a",
                supported_products: list[str] | None = None) -> DraftReply:
    """Grounded reply + confidence (LLM judge) + output guardrails."""
    from .evaluation import judge_answer

    sources = sources_of(retrieval)
    if analysis.product == "unknown":
        # Not one of our products (or none named): never answer from another product's documents
        return DraftReply(subject="", body=unsupported_product_message(
            analysis.language_name, supported_products or [], analysis.mentioned_product, llm),
            confidence=0.0, sources=[], found_relevant_info=False)
    if not retrieval.passages:
        return DraftReply(subject="", body=honest_no_info_message(analysis.language_name, llm), confidence=0.0,
                          sources=[], found_relevant_info=False)
    try:
        question = analysis.standalone_question + "\nCustomer summary: " + analysis.summary
        if order_id is None and analysis.category in ORDER_CATEGORIES:
            # Only order-related issues need an order number; never ask for it otherwise
            question += ("\nThe customer did not give an order ID. At the end, politely ask them to reply "
                         "with their order number so we can check this specific order.")
        answer = generate_answer(question,
                                 retrieval, analysis.language_name, analysis.urgency, settings.company_name, llm)
    except Exception as exc:
        log.error("Answer generation failed: %s", exc)
        return DraftReply(subject="", body=honest_no_info_message(analysis.language_name), confidence=0.0,
                          sources=sources, found_relevant_info=False, guardrail_issues=[f"generation_error: {exc}"])

    guard = check_output(answer, sources, retrieval.context)
    try:
        verdict = judge_answer(analysis.standalone_question, retrieval.context, guard.text, judge_llm)
        confidence = confidence_from_verdict(verdict.faithfulness, verdict.relevance, verdict.completeness)
        if verdict.unsupported_claims:
            guard.issues.append(f"unsupported_claims: {len(verdict.unsupported_claims)}")
    except Exception as exc:  # no judge -> be conservative
        log.warning("Judge failed, confidence set to 0.5: %s", exc)
        confidence = 0.5
    if not guard.allowed:
        confidence = min(confidence, 0.3)
    return DraftReply(subject="", body=strip_citations(guard.text), confidence=confidence, sources=sources,
                      found_relevant_info=True, guardrail_issues=guard.issues)


@traceable(name="policy_escalation_check")
def policy_escalation_evidence(idx: ProductIndex | None, complaint: str) -> list[str]:
    """RAG validation for the Log agent: find policy passages about disputes/fraud/legal
    for THIS product and report them when the complaint touches the same topic."""
    if idx is None:
        return []
    from .planner import detect_risk_flags

    hit_terms = detect_risk_flags(complaint)   # same whole-word rules as the Query Agent
    if not hit_terms:
        return []
    query = " ".join(hit_terms) + " dispute fraud chargeback legal escalation"
    ids = list(dict.fromkeys(bm25_search(idx, query, 4) + _safe_vector(idx, query)))
    evidence = []
    for cid in ids[:4]:
        c = idx.chunks[cid]
        if any(t in c.text.lower() for t in ("dispute", "fraud", "chargeback", "legal", "escalat", "investigat")):
            evidence.append(f"{c.source_file} › {c.section_title}")
    return list(dict.fromkeys(evidence))


def _safe_vector(idx: ProductIndex, query: str) -> list[str]:
    try:
        return vector_search(idx, query, 4)
    except Exception:
        return []


def evaluate_escalation(analysis: QueryAnalysis, retrieval: RetrievalResult | None, draft: DraftReply | None,
                        settings: Settings, policy_evidence: list[str] | None = None) -> EscalationDecision:
    """Deterministic escalation rules from the brief (any one => Escalated)."""
    reasons: list[str] = []
    if analysis.urgency == "High":
        reasons.append("Urgency is High")
    if analysis.sentiment == "Very Negative":
        reasons.append("Customer is very angry")
    if analysis.product == "unknown":
        mentioned = getattr(analysis, "mentioned_product", "") or ""
        reasons.append("Product not supported or not identified" + (f" ('{mentioned}')" if mentioned else ""))
    if retrieval is None or not retrieval.passages:
        reasons.append("Knowledge Agent found no matching document")
    elif draft is not None and not draft.found_relevant_info:
        reasons.append("A grounded answer could not be generated from the matching documents")
    if analysis.risk_flags:
        reasons.append("Legal threat / fraud / payment dispute: " + ", ".join(analysis.risk_flags))
    if draft is not None and draft.confidence < settings.confidence_threshold:
        reasons.append(f"Resolution confidence {draft.confidence:.2f} < {settings.confidence_threshold:.2f}")
    if draft is not None and any(i in draft.guardrail_issues for i in ("risky_commitment", "invalid_citation", "prompt_leak")):
        reasons.append("Output guardrail flagged the draft")
    if policy_evidence:
        reasons.append("Policy requires specialist handling: " + "; ".join(policy_evidence[:2]))
    return EscalationDecision(escalate=bool(reasons), reasons=reasons)
