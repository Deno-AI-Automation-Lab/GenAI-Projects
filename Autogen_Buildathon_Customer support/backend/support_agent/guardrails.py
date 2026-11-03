"""Input and output guardrails (deterministic, fast, unit-tested)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import TOKEN_RE

# Prompt-injection / jailbreak phrases (English + common variants)
_INJECTION = re.compile(
    r"(ignore (all|any|the|previous|prior|above) (instructions|rules|prompts?)|"
    r"disregard (the )?(system|previous)|you are now|act as (an?|the) (developer|admin|system)|"
    r"reveal (your|the) (system )?prompt|system prompt|jailbreak|do anything now|DAN mode|"
    r"<\s*/?\s*(system|assistant)\s*>|BEGIN (SYSTEM|ADMIN)|print your instructions)",
    re.IGNORECASE,
)
# Payment card numbers (13-19 digits, spaces/dashes allowed) and CVV mentions
_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_CVV = re.compile(r"\b(cvv|cvc|cvv2)\s*[:#-]?\s*\d{3,4}\b", re.I)
_OTP_PASS = re.compile(r"\b(otp|password|passcode|pin)\s*(is|:|=)\s*\S+", re.I)
_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_URL = re.compile(r"https?://\S+", re.I)
# Commitments the bot must never make unless the policy text says so
_RISKY_PROMISES = re.compile(
    r"(we guarantee|guaranteed refund|full refund (will be|has been) (issued|processed)|"
    r"compensation of|we will pay you|legal(ly)? liable|we admit)", re.I)


@dataclass
class GuardrailResult:
    allowed: bool
    text: str                      # sanitised text to use downstream
    issues: list[str] = field(default_factory=list)


def luhn_ok(number: str) -> bool:
    """Luhn checksum – avoids redacting order IDs that merely look numeric."""
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    if len(digits) < 13:
        return False
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Mask card numbers, CVV, OTP/passwords and Aadhaar-like IDs."""
    issues: list[str] = []

    def _card(m: re.Match) -> str:
        if luhn_ok(m.group(0)):
            issues.append("card_number_redacted")
            return "[REDACTED-CARD]"
        return m.group(0)

    text = _CARD.sub(_card, text)
    if _CVV.search(text):
        issues.append("cvv_redacted")
        text = _CVV.sub("[REDACTED-CVV]", text)
    if _OTP_PASS.search(text):
        issues.append("secret_redacted")
        text = _OTP_PASS.sub(lambda m: f"{m.group(1)} [REDACTED]", text)
    if _AADHAAR.search(text):
        issues.append("national_id_redacted")
        text = _AADHAAR.sub("[REDACTED-ID]", text)
    return text, issues


def check_input(text: str, max_chars: int = 4000) -> GuardrailResult:
    """Input guardrail: size, meaningful content, prompt injection, PII masking.

    Angry or rude language is NOT blocked – it is a sentiment signal for escalation.
    """
    issues: list[str] = []
    raw = (text or "").strip()
    if len(raw) < 5 or len(TOKEN_RE.findall(raw)) < 2:
        return GuardrailResult(False, raw, ["too_short_or_empty"])
    if len(raw) > max_chars:
        raw = raw[:max_chars]
        issues.append("truncated")
    if _INJECTION.search(raw):
        # Neutralise rather than reject: the customer may still have a real issue.
        raw = _INJECTION.sub("[removed]", raw)
        issues.append("prompt_injection_neutralised")
    raw, pii = redact_pii(raw)
    issues.extend(pii)
    return GuardrailResult(True, raw, issues)


def check_output(reply: str, allowed_sources: list[str], context: str) -> GuardrailResult:
    """Output guardrail: no leaked PII, no URLs absent from context, no risky promises,
    citations must refer to passages that exist."""
    issues: list[str] = []
    text, pii = redact_pii(reply or "")
    issues.extend(f"output_{p}" for p in pii)
    for url in _URL.findall(text):
        if url not in context:
            issues.append("unsupported_url")
            text = text.replace(url, "[link removed]")
    if _RISKY_PROMISES.search(text) and not _RISKY_PROMISES.search(context):
        issues.append("risky_commitment")
    cited = {int(n) for n in re.findall(r"\[S(\d+)\]", text)}
    if cited and max(cited) > len(allowed_sources):
        issues.append("invalid_citation")
    if allowed_sources and not cited:
        issues.append("no_citations")
    if _INJECTION.search(text):
        issues.append("prompt_leak")
    blocking = {"risky_commitment", "invalid_citation", "prompt_leak"}
    return GuardrailResult(not (blocking & set(issues)), text, issues)


def strip_citations(text: str) -> str:
    """Remove [S1] markers for the customer-facing email (sources listed separately)."""
    return re.sub(r"\s?\[S\d+\]", "", text)
