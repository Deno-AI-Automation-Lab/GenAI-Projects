"""Email: standard reply templates, SMTP sending and IMAP intake (Gmail).

TLS contexts come from ssl.create_default_context() (OS trust store via truststore)
– certificate verification is never disabled. Use a Gmail *App Password*.
"""
from __future__ import annotations

import email
import imaplib
import logging
import re
import smtplib
import ssl
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import make_msgid, parseaddr

from .config import ORDER_ID_RE, Settings

log = logging.getLogger(__name__)

_GREETING = {"en": "Dear {name},", "ta": "அன்புள்ள {name},", "hi": "प्रिय {name},",
             "fr": "Bonjour {name},", "es": "Estimado/a {name},", "de": "Hallo {name},"}
_CLOSING = {"en": "Warm regards,", "ta": "அன்புடன்,", "hi": "सादर,", "fr": "Cordialement,",
            "es": "Saludos cordiales,", "de": "Mit freundlichen Grüßen,"}
_ESCALATED_NOTE = {
    "en": "Your request has been reviewed by a senior support specialist.",
    "ta": "உங்கள் கோரிக்கையை ஒரு மூத்த ஆதரவு நிபுணர் பரிசீலித்துள்ளார்.",
    "hi": "आपके अनुरोध की समीक्षा एक वरिष्ठ सहायता विशेषज्ञ ने की है।",
}


# Subject tags (customer-facing). Every subject carries the tag and the ticket ID.
TAG_PROPOSED = "Resolved - please confirm"
TAG_ESCALATED = "Escalated"
TAG_RESOLVED = "Resolved"
TAG_REOPENED = "Escalated - High priority"
TAG_CLOSED = "Closed"

_CATEGORY_TITLES = {
    "shipping": "Delivery query", "refund": "Refund request", "account_login": "Account and login help",
    "product_usage": "Product question", "technical_issue": "Technical issue", "payment": "Payment query",
    "warranty": "Warranty claim", "other": "Support request",
}


def subject_title(product_name: str | None, issue_title: str | None, category: str | None) -> str:
    """Neutral subject text, e.g. 'AeroBeat X9 - Headphones not pairing' (max ~60 chars, whole words)."""
    product_short = " ".join((product_name or "").split()[:2])
    title = re.sub(r"\s+", " ", (issue_title or "").strip().strip(".")) or _CATEGORY_TITLES.get(category or "", "Support request")
    text = f"{product_short} - {title}" if product_short else title
    if len(text) > 60:
        text = text[:60].rsplit(" ", 1)[0]
    return text


def build_subject(tag: str, ticket_id: str, title: str) -> str:
    """Standard subject: '[<tag>] Ticket TKT-… | <neutral title>'."""
    return f"[{tag}] Ticket {ticket_id} | {title or 'Your support request'}"


def strip_markdown(text: str) -> str:
    """Plain-text email: remove **bold**, __bold__, `code` and '#' headings produced by the LLM."""
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    return text


def confirmation_block(*, has_email: bool, site_url: str, timeout_hours: int,
                       access_code: str | None = None) -> list[str]:
    """How the customer confirms. Email customers reply YES/NO; everyone can use 'My tickets'."""
    days = max(1, round(timeout_hours / 24))
    lines = ["Did this solve your issue?"]
    if has_email:
        lines += ["- Reply YES to this email if it is solved.",
                  "- Reply NO (with a short description) if you still need help - we will escalate it as high priority."]
    key = f"access code {access_code}" if access_code else "access code"
    lines += [f"- {'Or open' if has_email else 'Open'} 'My tickets' at {site_url} with your Ticket ID and {key} "
              "and choose Yes or No.",
              f"If we do not hear from you within {days} day{'s' if days != 1 else ''}, this ticket will be closed automatically."]
    return lines


def ref_line(ticket_id: str, order_id: str | None, status: str) -> str:
    """'Ticket ID: …    Order ID: …    Status: …' - the order part only when there is one."""
    parts = [f"Ticket ID: {ticket_id}"] + ([f"Order ID: {order_id}"] if order_id else []) + [f"Status: {status}"]
    return "    ".join(parts)


def render_customer_email(*, ticket_id: str, status: str, customer_name: str, order_id: str | None,
                          body: str, sources: list[str], language: str, company: str,
                          escalated: bool = False, confirm_lines: list[str] | None = None,
                          has_email: bool = True) -> str:
    """Standard customer reply template (plain text, localised greeting/closing)."""
    greet = _GREETING.get(language, _GREETING["en"]).format(name=customer_name)
    close = _CLOSING.get(language, _CLOSING["en"])
    lines = [greet, "", ref_line(ticket_id, order_id, status), ""]
    if escalated:
        lines += [_ESCALATED_NOTE.get(language, _ESCALATED_NOTE["en"]), ""]
    lines += [strip_markdown(body).strip(), ""]
    if sources:
        lines += ["Reference: " + "; ".join(sources), ""]
    if confirm_lines:
        lines += confirm_lines + [""]
    else:
        lines += ["Please reply to this email and keep the Ticket ID in the subject if you need anything else."
                  if has_email else "You can follow this ticket and ask questions in 'My tickets' on our support site.", ""]
    lines += [close, f"{company} Team"]
    return "\n".join(lines)


def render_ack_email(*, ticket_id: str, customer_name: str, order_id: str | None, access_code: str | None,
                     site_url: str, company: str) -> tuple[str, str]:
    """Acknowledgement for tickets raised by email: ticket ID + access code for 'My tickets'."""
    subject = f"[Received] Ticket {ticket_id} | We're looking into your request"
    lines = [f"Dear {customer_name},", "",
             f"Thank you for contacting {company}. We have created ticket {ticket_id} for your request."]
    if order_id:
        lines.append(f"Order ID on this ticket: {order_id}")
    lines += ["", "You will receive our reply by email."]
    if access_code:
        lines.append(f"You can also follow the ticket and ask questions under 'My tickets' at {site_url} "
                     f"using your Ticket ID and this access code: {access_code}")
    lines += ["", "Please keep the ticket ID in the subject line if you reply to this email.", "",
              "Warm regards,", f"{company} Team"]
    return subject, "\n".join(lines)


def render_escalation_email(*, ticket_id: str, customer_name: str, customer_email: str | None,
                            order_id: str | None, complaint: str, analysis: dict, reasons: list[str],
                            draft_reply: str, sources: list[str]) -> tuple[str, str]:
    """Internal escalation mail to the human support team."""
    subject = f"[ESCALATION] Ticket {ticket_id} | {analysis.get('urgency', '?')} urgency | {customer_name}"
    body = "\n".join([
        "A ticket requires human review.", "",
        f"Ticket ID      : {ticket_id}", f"Customer       : {customer_name}",
        f"Customer email : {customer_email or 'n/a'}", f"Order ID       : {order_id or 'not provided'}",
        f"Product        : {analysis.get('product_name') or analysis.get('product')}",
        f"Category       : {analysis.get('category')}", f"Urgency        : {analysis.get('urgency')}",
        f"Sentiment      : {analysis.get('sentiment')}", f"Language       : {analysis.get('language_name')}",
        "", "Escalation reasons:", *[f"  - {r}" for r in reasons], "",
        "Customer complaint:", complaint, "",
        "Draft reply (NOT sent – awaiting approval in the admin console):", draft_reply or "(none)", "",
        "KB sources: " + ("; ".join(sources) or "none"),
    ])
    return subject, body


def send_email(settings: Settings, to: str, subject: str, body: str, in_reply_to: str | None = None) -> str:
    """Send via SMTP over implicit TLS (465) or STARTTLS (587). Returns Message-ID."""
    pwd = settings.support_mailbox_password.get_secret_value()
    if not settings.email_enabled or not pwd or not settings.support_mailbox:
        raise RuntimeError("Email disabled, or SUPPORT_MAILBOX / SUPPORT_MAILBOX_APP_PASSWORD missing in .env")
    if not to:
        raise RuntimeError("No recipient address (set ESCALATION_EMAIL in .env for escalations)")
    msg = EmailMessage()
    msg["From"] = f"{settings.company_name} <{settings.support_mailbox}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid(domain=settings.support_mailbox.split("@")[-1])
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    msg.set_content(body, charset="utf-8")
    ctx = ssl.create_default_context()
    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=ctx, timeout=30) as s:
            s.login(settings.support_mailbox, pwd)
            s.send_message(msg)
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as s:
            s.starttls(context=ctx)
            s.login(settings.support_mailbox, pwd)
            s.send_message(msg)
    return msg["Message-ID"]


@dataclass
class InboundMail:
    message_id: str
    from_name: str
    from_email: str
    subject: str
    body: str
    order_id: str | None
    existing_ticket: str | None
    to_addrs: str = ""
    bulk_reason: str | None = None   # why the mail looks automated/marketing (None = personal mail)
    uid: int = 0


_NOREPLY_RE = re.compile(r"(no[-_.]?reply|do[-_.]?not[-_.]?reply|mailer-daemon|postmaster|bounce|"
                         r"newsletter|notifications?@|alerts?@|marketing|promo|offers?@|updates?@|info@)", re.I)


def _decode(value: str | None) -> str:
    return str(make_header(decode_header(value))) if value else ""


def _text_body(msg: email.message.Message) -> str:
    """Prefer text/plain; strip quoted replies so only the new message is used."""
    part_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                part_text = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
                break
    else:
        payload = msg.get_payload(decode=True) or b""
        part_text = payload.decode(msg.get_content_charset() or "utf-8", "replace")
        if msg.get_content_type() == "text/html":
            part_text = re.sub(r"<[^>]+>", " ", part_text)  # crude HTML strip for HTML-only mails
    # Cut the quoted original: Gmail/Outlook markers, possibly wrapped over two lines
    part_text = re.split(r"\n\s*On\s[^\n]{0,200}(?:\n[^\n]{0,200})?wrote:\s*\n|\n-{2,}\s*Original Message|"
                         r"\n_{5,}\s*\nFrom:|\nFrom:\s.*\nSent:", "\n" + part_text, maxsplit=1)[0]
    kept = []
    for line in part_text.splitlines():
        if line.startswith(">"):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def detect_bulk(msg: email.message.Message, from_email: str) -> str | None:
    """Newsletters, promotions, notifications and auto-replies carry tell-tale headers."""
    if msg.get("List-Unsubscribe") or msg.get("List-Id"):
        return "mailing list / newsletter"
    if (msg.get("Precedence") or "").lower() in ("bulk", "list", "junk"):
        return "bulk precedence"
    auto = (msg.get("Auto-Submitted") or "no").lower()
    if auto != "no":
        return "auto-submitted"
    if msg.get("X-Autoreply") or msg.get("X-Autorespond"):
        return "auto-reply"
    if _NOREPLY_RE.search(from_email or ""):
        return "automated sender"
    return None


def parse_inbound(raw: bytes, uid: int = 0) -> InboundMail:
    """Turn a raw RFC822 message into an InboundMail (order id + ticket id detection)."""
    msg = email.message_from_bytes(raw)
    name, addr = parseaddr(_decode(msg.get("From")))
    subject = _decode(msg.get("Subject"))
    body = _text_body(msg)
    oid = ORDER_ID_RE.search(f"{subject}\n{body}")
    tid = re.search(r"TKT-\d{8}-[0-9A-F]{6}", subject)
    to_addrs = " ".join(_decode(msg.get(h)) for h in ("To", "Cc", "Delivered-To") if msg.get(h))
    return InboundMail(
        message_id=msg.get("Message-ID") or f"<no-id-{hash(raw)}>", from_name=name or addr.split("@")[0],
        from_email=addr, subject=subject, body=body, order_id=oid.group(0) if oid else None,
        existing_ticket=tid.group(0) if tid else None, to_addrs=to_addrs,
        bulk_reason=detect_bulk(msg, addr), uid=uid,
    )


_GENERIC = {"wireless", "noise", "robot", "with", "and", "pro", "the", "smart", "cancelling"}


def product_keywords(products: list[dict]) -> set[str]:
    """Distinctive words from product names (e.g. puresweep, aerobeat, smartcrisp, vacuum, fryer)."""
    from .config import tokenize

    words: set[str] = set()
    for p in products:
        words.update(t for t in tokenize(p["name"]) if len(t) >= 5 and t not in _GENERIC)
    return words


def is_support_mail(mail: InboundMail, settings: Settings, products: list[dict]) -> tuple[bool, str]:
    """Decide whether an inbound mail should become a ticket. Returns (accept, reason).

    Order of checks:
      1. our own outgoing mail                       -> skip
      2. newsletter / promo / no-reply / bounce / auto-reply -> skip
      3. reply quoting an existing ticket ID         -> accept (follow-up)
      4. sent to the +support alias                  -> accept
      5. strict mode (MAIL_REQUIRE_SUPPORT_ALIAS)    -> skip everything else
      6. dedicated inbox (MAIL_ACCEPT_ALL)           -> accept every remaining personal mail
      7. mentions a product or a support keyword     -> accept, otherwise skip
    """
    sender = (mail.from_email or "").lower()
    box = settings.support_mailbox.lower()
    if sender == box:
        return False, "own mailbox"
    # Bounces/auto-replies often quote our subject (and ticket ID) - never treat them as replies
    if mail.bulk_reason:
        return False, mail.bulk_reason
    if mail.existing_ticket:
        return True, f"reply to {mail.existing_ticket}"
    local, _, domain = box.partition("@")
    alias = f"{local}+{settings.mail_support_alias}@{domain}".lower()
    if alias in mail.to_addrs.lower():
        return True, "sent to support alias"
    if settings.mail_require_alias:
        return False, "not sent to support alias"
    if settings.mail_accept_all:
        return True, "dedicated support inbox (MAIL_ACCEPT_ALL)"
    text = f"{mail.subject}\n{mail.body}".lower()
    keywords = {k.strip().lower() for k in settings.mail_accept_keywords.split(",") if k.strip()}
    hits = [k for k in keywords | product_keywords(products) if k in text]
    if hits:
        return True, "mentions " + ", ".join(sorted(hits)[:3])
    return False, "no product or support keyword"


def fetch_new_mail(settings: Settings, last_uid: int | None, uidvalidity: str | None
                   ) -> tuple[list[InboundMail], int, str]:
    """Read only mail that arrived AFTER intake started, without changing read/unread flags.

    * First run (last_uid is None): records the current newest UID as the baseline and
      returns nothing, so the existing inbox is never processed (unless
      MAIL_PROCESS_EXISTING=true, which starts from UNSEEN mail).
    * BODY.PEEK[] is used so mail is not marked as read.
    Returns (mails, new_last_uid, uidvalidity).
    """
    pwd = settings.support_mailbox_password.get_secret_value()
    if not pwd:
        raise RuntimeError("SUPPORT_MAILBOX_APP_PASSWORD missing")
    out: list[InboundMail] = []
    with imaplib.IMAP4_SSL(settings.imap_host, ssl_context=ssl.create_default_context()) as imap:
        imap.login(settings.support_mailbox, pwd)
        imap.select("INBOX", readonly=True)            # read-only: flags are never changed
        _, v = imap.response("UIDVALIDITY")
        current_validity = (v[0].decode() if v and v[0] else "")
        _, data = imap.uid("search", None, "ALL")
        uids = [int(u) for u in data[0].split()] if data and data[0] else []
        newest = max(uids) if uids else 0
        if last_uid is None or (uidvalidity and uidvalidity != current_validity):
            if not settings.mail_process_existing:
                log.info("Mail intake baseline set at UID %s – existing mail will be ignored", newest)
                return [], newest, current_validity
            _, data = imap.uid("search", None, "UNSEEN")
            candidates = [int(u) for u in data[0].split()] if data and data[0] else []
        else:
            candidates = [u for u in uids if u > last_uid]
        candidates = sorted(candidates)[: settings.mail_max_per_poll]
        for uid in candidates:
            _, msg_data = imap.uid("fetch", str(uid), "(BODY.PEEK[])")
            try:
                raw = next(part[1] for part in msg_data if isinstance(part, tuple))
                out.append(parse_inbound(raw, uid))
            except Exception as exc:
                log.error("Could not parse mail UID %s: %s", uid, exc)
        new_last = max(candidates) if candidates else (last_uid if last_uid is not None else newest)
    return out, new_last, current_validity
