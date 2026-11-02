"""Read-only diagnostic: shows the newest inbox mails and what the intake would do with each.

    python scripts/check_mail_intake.py          # newest 10 mails
    python scripts/check_mail_intake.py --n 20

Nothing is changed: the inbox is opened read-only and no tickets or emails are created.
"""
from __future__ import annotations

import argparse
import imaplib
import ssl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from support_agent.emailer import is_support_mail, parse_inbound  # noqa: E402
from support_agent.llm import bootstrap_runtime  # noqa: E402
from support_agent.services import build_services  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    bootstrap_runtime()
    svc = build_services()
    s = svc.settings
    print(f"EMAIL_ENABLED={s.email_enabled}  mailbox={s.support_mailbox}  "
          f"require_alias={s.mail_require_alias}  poll={s.mail_poll_seconds}s")
    baseline = svc.memory.get_kv(f"mail_last_uid:{s.support_mailbox.lower()}")
    print(f"Last processed UID (baseline): {baseline}  (only mail with a HIGHER UID is processed)\n")
    pwd = s.support_mailbox_password.get_secret_value()
    if not pwd:
        print("SUPPORT_MAILBOX_APP_PASSWORD is empty in .env")
        return
    with imaplib.IMAP4_SSL(s.imap_host, ssl_context=ssl.create_default_context()) as imap:
        imap.login(s.support_mailbox, pwd)
        imap.select("INBOX", readonly=True)
        _, data = imap.uid("search", None, "ALL")
        uids = sorted(int(u) for u in data[0].split())[-args.n:]
        for uid in reversed(uids):
            _, msg_data = imap.uid("fetch", str(uid), "(BODY.PEEK[])")
            raw = next(part[1] for part in msg_data if isinstance(part, tuple))
            m = parse_inbound(raw, uid)
            ok, reason = is_support_mail(m, s, svc.router.products)
            new = baseline is None or uid > int(baseline)
            status = ("WOULD CREATE TICKET" if ok else "SKIP") if new else "already past (before/at baseline)"
            seen = svc.memory.mail_seen(m.message_id)
            print(f"UID {uid} | from {m.from_email} | to {m.to_addrs[:60]}")
            print(f"   subject: {m.subject[:80]}")
            print(f"   -> {status} ({reason}){'  [already ticketed]' if seen else ''}\n")


if __name__ == "__main__":
    main()
