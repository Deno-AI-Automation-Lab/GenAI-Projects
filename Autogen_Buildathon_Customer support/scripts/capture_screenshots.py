"""Capture README screenshots of the running app with Playwright.

Prereqs: API on :8000 and UI on :8501 running, `pip install playwright` and either
`python -m playwright install chromium` or an installed Chrome (`--chrome`).

    python scripts/capture_screenshots.py --chrome            # UI screenshots + both test cases
    python scripts/capture_screenshots.py --skip-cases        # UI pages only
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
CASES = json.loads((ROOT / "tests" / "test_cases.json").read_text(encoding="utf-8"))


def shot(page: Page, path: Path) -> None:
    """Whole page, not just the visible part: Streamlit scrolls inside its own container,
    so make the window tall, measure where the content ends and clip there."""
    page.set_viewport_size({"width": 1440, "height": 3000})
    page.wait_for_timeout(2000)
    page.evaluate("document.querySelectorAll('*').forEach(e => { if (e.scrollTop) e.scrollTop = 0; })")
    bottom = page.evaluate("""() => {
        const box = document.querySelector('[data-testid="stMainBlockContainer"]') || document.body;
        let b = 0;
        box.querySelectorAll('*').forEach(e => { const r = e.getBoundingClientRect();
            if (r.height > 0 && r.bottom > b) b = r.bottom; });
        return Math.ceil(b);
    }""")
    height = min(3000, max(900, bottom + 40))
    page.screenshot(path=str(path), clip={"x": 0, "y": 0, "width": 1440, "height": height})
    page.set_viewport_size({"width": 1440, "height": 1000})


def go(page: Page, label: str) -> None:
    page.get_by_text(label, exact=True).first.click()
    page.wait_for_timeout(2500)


def submit_case(page: Page, case: dict, out: Path, wait_s: int) -> None:
    go(page, "Get help")
    page.get_by_label("Your name").fill(case["customer_name"])
    page.get_by_label("Order ID (optional)").fill(case["order_id"])
    page.get_by_label("Email for the reply (optional)").fill(case.get("customer_email", ""))
    page.get_by_label("What happened?").fill(case["complaint"])
    page.get_by_role("button", name="Submit ticket").click()
    # wait until the reply panel shows a final badge
    for _ in range(wait_s // 2):
        page.wait_for_timeout(2000)
        if page.locator(".badge.ok, .badge.bad").count():
            break
    page.wait_for_timeout(2500)
    shot(page, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--chrome", action="store_true", help="use the installed Google Chrome")
    ap.add_argument("--skip-cases", action="store_true")
    ap.add_argument("--wait", type=int, default=180, help="seconds to wait per test case")
    ap.add_argument("--admin-password", default=None,
                    help="defaults to ADMIN_PASSWORD from .env (never hard-coded)")
    args = ap.parse_args()
    if not args.admin_password:
        from dotenv import dotenv_values

        args.admin_password = (dotenv_values(ROOT / ".env").get("ADMIN_PASSWORD") or "").strip()
    if not args.admin_password:
        raise SystemExit("Set ADMIN_PASSWORD in .env (or pass --admin-password) to capture the admin screens.")
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        launch = {"headless": True}
        if args.chrome:
            launch["executable_path"] = shutil.which("google-chrome") or shutil.which("chrome") or None
            launch["channel"] = None if launch["executable_path"] else "chrome"
        browser = p.chromium.launch(**{k: v for k, v in launch.items() if v is not None})
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(args.url)
        page.wait_for_selector(".hero", timeout=60000)
        page.wait_for_timeout(2000)
        shot(page, OUT / "01_home.png")
        if not args.skip_cases:
            for i, case in enumerate(CASES, start=1):
                submit_case(page, case, OUT / f"test_case_{i}_{case['id']}.png", args.wait)
        go(page, "My tickets")
        shot(page, OUT / "02_my_tickets.png")
        page.get_by_text("Staff sign-in").click()
        page.get_by_label("User ID").fill("admin")
        page.get_by_role("textbox", name="Password").fill(args.admin_password)
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_timeout(3000)
        go(page, "Admin console")
        page.wait_for_timeout(2000)
        shot(page, OUT / "03_admin_dashboard.png")
        page.get_by_role("tab", name="Ticket log").click()
        page.wait_for_timeout(2500)
        shot(page, OUT / "07_admin_ticket_log.png")
        page.get_by_role("tab", name="Approvals").click()
        page.wait_for_timeout(2500)
        shot(page, OUT / "04_admin_approvals.png")
        page.get_by_role("tab", name="Escalations").click()
        page.wait_for_timeout(2500)
        shot(page, OUT / "08_admin_escalations.png")
        page.get_by_role("tab", name="Vector chunks").click()
        page.wait_for_timeout(2500)
        shot(page, OUT / "05_admin_vector_chunks.png")
        page.get_by_role("tab", name="Knowledge graph").click()
        page.wait_for_timeout(2500)
        shot(page, OUT / "06_admin_knowledge_graph.png")
        browser.close()
    print(f"Screenshots saved to {OUT}")


if __name__ == "__main__":
    main()
