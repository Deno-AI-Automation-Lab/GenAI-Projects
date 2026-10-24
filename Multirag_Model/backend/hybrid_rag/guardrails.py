"""Input and output guardrails (rule-based, no extra LLM calls, fully testable)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from hybrid_rag.config import NO_ANSWER, tokenize

# Prompt-injection / jailbreak phrasings we refuse outright.
_INJECTION = re.compile(
    r"(ignore (all|any|the|previous|prior|above).{0,20}(instructions|rules|prompt)"
    r"|disregard (the|your|all).{0,20}(instructions|rules)"
    r"|system prompt|developer mode|jailbreak|you are now|act as (an? )?unrestricted"
    r"|reveal (your|the) (prompt|instructions))",
    re.IGNORECASE,
)
# Requests that ask the bot to go outside the indexed documents.
_WEB = re.compile(r"\b(search|browse|google|look up) (the )?(web|internet|online)\b", re.I)
# Destructive DB instructions sneaking in via the question.
_DB_WRITE = re.compile(r"\b(drop|delete|detach|merge|create|set|remove)\b\s+\(|;\s*(drop|delete)\b", re.I)
# PII that must not leak back to the user.
_PII = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "AADHAAR": re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"),
    "CARD": re.compile(r"(?<!\d)(?:\d[ -]?){15,16}(?!\d)"),
}
_TOXIC = re.compile(r"\b(kill|bomb|terroris\w*|suicide)\b", re.I)


@dataclass
class GuardResult:
    allowed: bool
    text: str
    reason: str = ""


def check_input(question: str, max_chars: int = 1000) -> GuardResult:
    """Validate a user question before it reaches any LLM or database."""
    q = (question or "").strip()
    if not q:
        return GuardResult(False, q, "Empty question.")
    if len(q) > max_chars:
        return GuardResult(False, q, f"Question is too long (max {max_chars} characters).")
    if not tokenize(q):
        return GuardResult(False, q, "Question contains no searchable words.")
    if _INJECTION.search(q):
        return GuardResult(False, q, "Instruction-override attempts are not allowed.")
    if _WEB.search(q):
        return GuardResult(False, q, "I can only answer from the uploaded documents, not the web.")
    if _DB_WRITE.search(q):
        return GuardResult(False, q, "Database modification requests are not allowed.")
    if _TOXIC.search(q):
        return GuardResult(False, q, "This topic is outside what this assistant can help with.")
    return GuardResult(True, q)


def redact_pii(text: str) -> str:
    """Mask emails, phone, Aadhaar and card numbers."""
    for label, pattern in _PII.items():
        text = pattern.sub(f"[{label} REDACTED]", text)
    return text


def check_output(answer: str, context: str) -> GuardResult:
    """Validate a generated answer: grounded, no prompt leakage, no PII."""
    a = (answer or "").strip()
    if not a:
        return GuardResult(False, NO_ANSWER, "Empty answer.")
    if NO_ANSWER.lower() in a.lower():
        return GuardResult(True, NO_ANSWER)
    if "CONTEXT:" in a or "You are a document assistant" in a:
        return GuardResult(False, NO_ANSWER, "Prompt leakage blocked.")
    # Grounding heuristic: enough of the answer's words must appear in the context.
    overlap = grounding_score(a, context)
    if overlap < 0.35:
        return GuardResult(False, NO_ANSWER, f"Answer not grounded in documents (overlap {overlap:.2f}).")
    return GuardResult(True, redact_pii(a), f"grounding={overlap:.2f}")


def grounding_score(answer: str, context: str) -> float:
    """Share of the answer's content tokens (len>3, no citation tags) found in context."""
    stripped = re.sub(r"\[(S\d+|G)\]", " ", answer)
    words = [t for t in tokenize(stripped) if len(t) > 3]
    if not words:
        return 1.0
    ctx = set(tokenize(context))
    return sum(w in ctx for w in words) / len(words)
