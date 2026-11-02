"""Streamlit frontend (UI only – all logic lives behind the FastAPI backend).

Run:  streamlit run support_agent/ui.py
UI state is kept in st.session_state["store"].
"""
from __future__ import annotations

import base64
import html
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run support_agent/ui.py` without installing the package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support_agent.client import ApiError, SupportApiClient  # noqa: E402
from support_agent.config import ASSETS_DIR, FONTS_DIR, get_settings  # noqa: E402

BRAND = get_settings().company_name  # set COMPANY_NAME in .env
st.set_page_config(page_title=BRAND, page_icon="🛟", layout="wide")

AGENT_LABELS = {
    "Input_Guardrail": "Input check",
    "Customer_Query_Agent": "Customer Query Agent",
    "Knowledge_Base_Agent": "Knowledge Base Agent",
    "Resolution_Agent": "Resolution Agent",
    "Log_Agent": "Log Agent",
    "Support_Approver": "Support approver (human)",
    "Fallback": "Fallback runner",
    "System": "System",
    "Email_Intake": "Email intake",
}
KIND_LABELS = {"tool_call": "calls a tool", "tool_result": "tool result", "message": "says",
               "system": "note", "error": "error"}
FINAL = {"Resolved", "Escalated", "Escalated - Awaiting Approval", "Failed"}


# --------------------------------------------------------------------------- #
# State & assets
# --------------------------------------------------------------------------- #
def store() -> dict:
    """Single session store for the whole UI."""
    if "store" not in st.session_state:
        st.session_state["store"] = {"tickets": [], "current": None, "events": {}, "token": None,
                                     "chat": {}, "last_seq": {}, "orders": {}}
    st.session_state["store"].setdefault("orders", {})
    return st.session_state["store"]


def api() -> SupportApiClient:
    base = os.getenv("API_BASE_URL") or get_settings().api_base_url
    return SupportApiClient(base, token=store()["token"])


@st.cache_data(show_spinner=False)
def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def inject_css() -> None:
    """Bundled fonts (Inter + Noto Sans Tamil fallback), palette and layout."""
    fonts = f"""
    @font-face {{font-family:'Inter';font-weight:400;src:url(data:font/woff2;base64,{b64(FONTS_DIR / 'inter-latin-400-normal.woff2')}) format('woff2');}}
    @font-face {{font-family:'Inter';font-weight:700;src:url(data:font/woff2;base64,{b64(FONTS_DIR / 'inter-latin-700-normal.woff2')}) format('woff2');}}
    @font-face {{font-family:'Noto Sans Tamil';font-weight:400;src:url(data:font/ttf;base64,{b64(FONTS_DIR / 'NotoSansTamil-Regular.ttf')}) format('truetype');}}
    @font-face {{font-family:'Noto Sans Tamil';font-weight:700;src:url(data:font/ttf;base64,{b64(FONTS_DIR / 'NotoSansTamil-Bold.ttf')}) format('truetype');}}
    """
    hero = b64(ASSETS_DIR / "images" / "hero.svg")
    st.markdown(f"""<style>{fonts}
    :root {{ --ink:#13294B; --sea:#2F8F8A; --sea-dark:#226E6A; --fog:#EEF3F6; --paper:#FFFFFF;
             --line:#D5DFE6; --muted:#566B80; --ok:#1E8A4C; --bad:#C2372C; --amber:#F2A33A; }}
    html, body, .stApp, [data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]),
    [data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]) {{
        font-family:'Inter','Noto Sans Tamil',system-ui,sans-serif !important; }}
    .stTextInput input, .stTextArea textarea {{ background:#fff !important; border:1px solid var(--line) !important;
        border-radius:10px !important; color: var(--ink) !important; }}
    .stTextInput input:focus, .stTextArea textarea:focus {{ border-color: var(--sea) !important; }}
    [data-testid="stForm"] {{ background: var(--paper); border:1px solid var(--line); border-radius:14px; }}
    .stApp {{ background: var(--fog); color: var(--ink); }}
    .block-container {{ padding-top: 1.2rem; max-width: 1240px; }}
    h1,h2,h3,h4 {{ color: var(--ink); letter-spacing:-0.01em; }}
    .hero {{ border-radius: 18px; padding: 44px 40px 52px; margin-bottom: 26px; color:#fff;
             background: url(data:image/svg+xml;base64,{hero}) center/cover no-repeat, var(--ink); }}
    .hero h1 {{ color:#fff; font-size: 2.4rem; line-height:1.15; margin:0 0 10px; max-width: 16ch; }}
    .hero p {{ color:#CFE3EA; font-size:1.05rem; max-width: 52ch; margin:0; }}
    .brand {{ font-weight:700; color:#9FD3CF; margin-bottom: 18px; font-size: .95rem; }}
    .panel {{ background: var(--paper); border:1px solid var(--line); border-radius: 14px; padding: 4px 18px 14px; }}
    .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {{
        background: var(--sea); color:#fff; border:0; border-radius: 10px; font-weight:700; padding: .55rem 1.2rem; }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{ background: var(--sea-dark); color:#fff; }}
    .stButton>button:focus-visible, .stFormSubmitButton>button:focus-visible {{ outline: 3px solid var(--amber); }}
    .letter {{ background: var(--paper); border:2px solid var(--ink); border-radius: 14px; padding: 22px 26px;
               box-shadow: 6px 6px 0 var(--ink); margin: 6px 0 18px; }}
    .letter .subject {{ font-weight:700; font-size:1.02rem; margin-bottom: 12px; color: var(--ink); }}
    .letter .body {{ white-space: pre-wrap; line-height: 1.6; color:#1C2B3A; }}
    .badge {{ display:inline-block; padding: 4px 12px; border-radius: 999px; font-weight:700;
              font-size:.85rem; color:#fff; margin-right:8px; }}
    .badge.ok {{ background: var(--ok); }} .badge.bad {{ background: var(--bad); }}
    .badge.wait {{ background: var(--amber); color: var(--ink); }}
    .badge.closed {{ background: #6B7F90; }}
    .confirm {{ background: #FFF7E8; border: 1px solid #F2C36B; border-radius: 12px; padding: 14px 18px 4px;
               margin: 4px 0 16px; }}
    .meta {{ color: var(--muted); font-size: .88rem; }}
    .tid {{ font-weight:700; color: var(--sea-dark); }}
    [data-testid="stSidebar"] {{ background: var(--ink); }}
    [data-testid="stSidebar"] * {{ color: #E7F0F4 !important; }}
    /* Staff sign-in expander: stays dark with light text, open or closed */
    [data-testid="stSidebar"] [data-testid="stExpander"] details,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
        background: transparent !important; border-color: rgba(231,240,244,.35) !important; }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * {{ color: #FFFFFF !important; }}
    /* Sign-in form: white card, dark labels and input text, visible borders */
    [data-testid="stSidebar"] [data-testid="stForm"] {{ background: var(--paper) !important; }}
    [data-testid="stSidebar"] [data-testid="stForm"] label,
    [data-testid="stSidebar"] [data-testid="stForm"] label *,
    [data-testid="stSidebar"] [data-testid="stForm"] input {{ color: var(--ink) !important; }}
    [data-testid="stSidebar"] [data-testid="stForm"] input {{ background: #fff !important;
        border: 1px solid #9FB3C2 !important; border-radius: 10px !important; }}
    [data-testid="stSidebar"] [data-testid="stForm"] [data-baseweb="input"] {{ background: #fff !important;
        border-color: #9FB3C2 !important; }}
    [data-testid="stSidebar"] [data-testid="stForm"] button[aria-label*="password"],
    [data-testid="stSidebar"] [data-testid="stForm"] button[aria-label*="password"] * {{
        color: var(--muted) !important; background: transparent !important; }}
    [data-testid="stSidebar"] [data-testid="stForm"] [data-testid="stFormSubmitButton"] button,
    [data-testid="stSidebar"] [data-testid="stForm"] [data-testid="stFormSubmitButton"] button * {{
        color: #fff !important; }}
    [data-testid="stSidebar"] [data-testid="stAlert"] * {{ color: var(--ink) !important; }}
    [data-testid="stMetricValue"] {{ color: var(--ink); }}
    @media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
    </style>""", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(f"""<div class="hero"><div class="brand">{html.escape(BRAND)}</div>
      <h1>Tell us what went wrong. We'll take it from here.</h1>
      <p>Help for your PureSweep V12 robot vacuum, AeroBeat X9 headphones and SmartCrisp Pro air fryer.
      You get a ticket ID right away and our support team replies here and by email.</p></div>""",
                unsafe_allow_html=True)


def status_badge(status: str) -> str:
    if status == "Resolved":
        return '<span class="badge ok">Resolved</span>'
    if status == "Pending Confirmation":
        return '<span class="badge ok">Solution sent</span><span class="badge wait">Awaiting your confirmation</span>'
    if status == "Closed - No Response":
        return '<span class="badge closed">Closed</span>'
    if status == "Closed - Not Supported":
        return '<span class="badge closed">Closed - product not supported</span>'
    if status == "Escalated - Awaiting Approval":
        return '<span class="badge bad">Escalated</span><span class="badge wait">Awaiting approval</span>'
    if status.startswith("Escalated"):
        return '<span class="badge bad">Escalated</span>'
    if status == "Failed":
        return '<span class="badge bad">Needs more detail</span>'
    return '<span class="badge wait">Working on it</span>'


# --------------------------------------------------------------------------- #
# Ticket views
# --------------------------------------------------------------------------- #
def complaint_form() -> None:
    st.subheader("Raise a support ticket")
    with st.form("complaint", clear_on_submit=False):
        c1, c2 = st.columns(2)
        name = c1.text_input("Your name", max_chars=120)
        order = c2.text_input("Order ID (optional)", placeholder="Only if it's about an order", max_chars=64)
        email = st.text_input("Email for the reply (optional)", placeholder="you@example.com")
        complaint = st.text_area("What happened?", height=170, max_chars=4000,
                                 placeholder="Describe the problem in any language – English, தமிழ், हिन्दी…")
        submitted = st.form_submit_button("Submit ticket")
    if submitted:
        missing = [label for label, v in (("name", name), ("description", complaint)) if not v.strip()]
        if missing:
            st.error("Add your " + ", ".join(missing) + " to submit the ticket.")
            return
        try:
            t = api().submit({"customer_name": name, "order_id": order.strip() or None, "complaint": complaint,
                              "customer_email": email.strip() or None, "channel": "web"})
        except ApiError as exc:
            st.error(str(exc))
            return
        s = store()
        s["current"] = t["ticket_id"]
        s["tickets"].insert(0, t["ticket_id"])
        s["orders"][t["ticket_id"]] = t.get("access_code") or order.strip()   # needed to view the ticket later
        s["events"][t["ticket_id"]] = []
        s["last_seq"][t["ticket_id"]] = 0
        st.success(f"Ticket {t['ticket_id']} created. Your access code is {t.get('access_code')} - keep it with "
                   "the ticket ID to open this ticket later in 'My tickets'. Follow the agents' progress on the right.")


def refresh_events(tid: str) -> dict:
    """Fetch new events incrementally; returns the latest ticket."""
    s = store()
    client = api()
    order = s["orders"].get(tid)
    new = client.events(tid, s["last_seq"].get(tid, 0), order)
    if new:
        s["events"].setdefault(tid, []).extend(new)
        s["last_seq"][tid] = new[-1]["seq"]
    return client.ticket(tid, order)


def render_timeline(events: list[dict]) -> None:
    """Step-by-step timeline: one expandable step per agent message/tool call."""
    if not events:
        st.caption("Waiting for the first agent…")
    for i, ev in enumerate(events, start=1):
        who = AGENT_LABELS.get(ev["agent"], ev["agent"])
        kind = KIND_LABELS.get(ev["kind"], ev["kind"])
        icon = {"error": "⚠️", "tool_call": "🔧", "tool_result": "📄", "system": "ℹ️"}.get(ev["kind"], "💬")
        last = i == len(events)
        with st.expander(f"{icon} Step {i}: {who} {kind}", expanded=last and ev["kind"] == "message"):
            st.markdown(f"<div class='meta'>{html.escape(ev['created_at'])}</div>", unsafe_allow_html=True)
            st.text(ev["content"])


def confirmation_widget(t: dict, order: str | None, key: str) -> None:
    """'Did this solve your issue?' buttons for tickets waiting for the customer."""
    if t["status"] != "Pending Confirmation":
        return
    st.markdown("<div class='confirm'><b>Did this solve your issue?</b><br>"
                "<span class='meta'>Your answer closes the ticket, or sends it to a senior specialist "
                "as high priority.</span></div>", unsafe_allow_html=True)
    comment = st.text_area("What is still not working? (optional)", key=f"fb_{key}_{t['ticket_id']}", height=80)
    c1, c2 = st.columns(2)
    try:
        if c1.button("Yes, it's solved", key=f"yes_{key}_{t['ticket_id']}", width="stretch"):
            api().confirm(t["ticket_id"], True, comment or None, order)
            st.success("Thank you! Your ticket is now resolved.")
            st.rerun()
        if c2.button("No, I still need help", key=f"no_{key}_{t['ticket_id']}", width="stretch"):
            api().confirm(t["ticket_id"], False, comment or None, order)
            st.warning("Sorry about that. We've escalated your ticket to a senior specialist as high priority.")
            st.rerun()
    except ApiError as exc:
        st.error(str(exc))


def render_reply(t: dict) -> None:
    st.markdown(status_badge(t["status"]) + f" <span class='tid'>{html.escape(t['ticket_id'])}</span>",
                unsafe_allow_html=True)
    meta = [f"Category: {t.get('category') or '–'}", f"Urgency: {t.get('urgency') or '–'}",
            f"Sentiment: {t.get('sentiment') or '–'}"]
    if t.get("confidence") is not None:
        meta.append(f"Confidence: {t['confidence']:.0%}")
    st.markdown(f"<div class='meta'>{' | '.join(meta)}</div>", unsafe_allow_html=True)
    if t["status"] == "Escalated - Awaiting Approval":
        st.info("This ticket was escalated to a support specialist. You'll see their approved reply here "
                "and by email once they review it.")
    elif t.get("reply_body"):
        subject = html.escape(t.get("reply_subject") or "Your support reply")
        st.markdown(f"<div class='letter'><div class='subject'>{subject}</div>"
                    f"<div class='body'>{html.escape(t['reply_body'])}</div></div>", unsafe_allow_html=True)
    if t.get("kb_sources"):
        with st.expander("Sources used"):
            for src in t["kb_sources"]:
                st.markdown(f"- {src}")


@st.fragment(run_every=2)
def live_ticket_panel() -> None:
    """Auto-refreshing panel for the current ticket (stops polling once final)."""
    s = store()
    tid = s["current"]
    if not tid:
        st.subheader("Live agent steps")
        st.caption("Submit a ticket to watch the Customer Query, Knowledge Base, Resolution and Log agents work.")
        return
    try:
        t = refresh_events(tid)
    except ApiError as exc:
        st.error(str(exc))
        return
    st.subheader("Live agent steps")
    render_timeline(s["events"].get(tid, []))
    st.subheader("Reply")
    render_reply(t)
    confirmation_widget(t, s["orders"].get(tid), "live")


def support_page() -> None:
    hero()
    left, right = st.columns([5, 6], gap="large")
    with left:
        complaint_form()
    with right:
        live_ticket_panel()


def my_tickets_page() -> None:
    st.header("My tickets")
    s = store()
    st.caption("Open a ticket you raised here or by email. Use the ticket ID and the access code from your confirmation (or the order ID, if you gave one).")
    with st.form("lookup"):
        c1, c2 = st.columns(2)
        lookup_id = c1.text_input("Ticket ID", placeholder="TKT-20261102-ABC123")
        lookup_order = c2.text_input("Access code or Order ID", placeholder="e.g. 7F3A9C")
        if st.form_submit_button("Open ticket"):
            tid, order = lookup_id.strip(), lookup_order.strip()
            if not tid or not order:
                st.error("Enter the Ticket ID and the access code (or the order ID).")
                st.stop()
            try:
                api().ticket(tid, order)   # API returns 404 unless ticket ID and order ID match
                if tid not in s["tickets"]:
                    s["tickets"].insert(0, tid)
                s["orders"][tid] = order
            except ApiError:
                st.error("No ticket matches this ticket ID and access code / order ID. Check both and try again.")
    if not s["tickets"]:
        return
    tid = st.selectbox("Ticket", s["tickets"], index=0)
    order = s["orders"].get(tid)
    try:
        t = api().ticket(tid, order)
    except ApiError as exc:
        st.error(str(exc))
        return
    render_reply(t)
    confirmation_widget(t, order, "mine")
    st.subheader("Chat about this ticket")
    try:
        history = api().chat_history(tid, order)
    except ApiError:
        history = s["chat"].get(tid, [])
    for m in history:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])
    if prompt := st.chat_input("Ask a follow-up question"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking our documentation…"):
                try:
                    r = api().chat(tid, prompt, order)
                    st.markdown(r["answer"])
                    if r["sources"]:
                        st.caption("Sources: " + "; ".join(r["sources"]))
                except ApiError as exc:
                    st.error(str(exc))
        s["chat"][tid] = history + [{"role": "user", "content": prompt}]


def ticket_log_tab(client: SupportApiClient) -> None:
    """Admin-only: every ticket as a table, filterable, downloadable as CSV."""
    try:
        rows = client.log()
    except ApiError as exc:
        st.error(str(exc))
        return
    if not rows:
        st.caption("No tickets logged yet.")
        return
    df = pd.DataFrame(rows)
    status_filter = st.multiselect("Status", sorted(df["status"].unique()), default=None)
    if status_filter:
        df = df[df["status"].isin(status_filter)]
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download log (CSV)", data=client.log_csv(), file_name="ticket_log.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #
ADMIN_COOKIE = "support_admin_token"
COOKIE_MAX_AGE = 8 * 3600   # same lifetime as the API token


def _write_cookie(value: str | None) -> None:
    """Set (or delete) the admin cookie in the browser.
    st.html with JavaScript runs in the page itself (no iframe, no deprecated components API)."""
    if value:
        js = f"document.cookie = '{ADMIN_COOKIE}={value}; path=/; max-age={COOKIE_MAX_AGE}; SameSite=Strict';"
    else:
        js = f"document.cookie = '{ADMIN_COOKIE}=; path=/; max-age=0; SameSite=Strict';"
    st.html(f"<script>{js}</script>", unsafe_allow_javascript=True)


def restore_admin_session() -> None:
    """After a browser refresh the session is new: sign the admin back in from the cookie,
    but only if the API still accepts the token (expired/forged tokens are discarded)."""
    s = store()
    if s["token"] or s.get("cookie_checked"):
        return
    s["cookie_checked"] = True
    token = st.context.cookies.get(ADMIN_COOKIE)
    if not token:
        return
    try:
        SupportApiClient(api().base, token=token).me()
        s["token"] = token
    except ApiError:
        s["pending_cookie"] = ""          # invalid or expired -> delete it


def sync_admin_cookie() -> None:
    """Write any pending cookie change (after sign-in / sign-out)."""
    s = store()
    if "pending_cookie" in s:
        _write_cookie(s.pop("pending_cookie") or None)


def admin_login_sidebar() -> None:
    s = store()
    with st.sidebar:
        st.markdown(f"### {BRAND}")
        if s["token"]:
            st.markdown("Signed in as **admin**")
            if st.button("Sign out"):
                s["token"] = None
                s["pending_cookie"] = ""   # delete the browser cookie
                st.rerun()
            return
        with st.expander("Staff sign-in"):
            with st.form("admin_login"):
                u = st.text_input("User ID")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in"):
                    try:
                        s["token"] = SupportApiClient(api().base).login(u, p)
                        s["pending_cookie"] = s["token"]   # keep the sign-in across page refreshes
                        st.rerun()
                    except ApiError:
                        st.error("User ID or password is incorrect.")


def _show_decision(result: dict) -> None:
    """Tell the approver exactly what happened (emailed / no email on file / email failed)."""
    detail = result.get("detail") or ""
    if "email failed" in detail:
        st.error(f"Decision saved, but the email could not be sent: {detail}")
    elif "emailed to" in detail:
        st.success(f"Done. {detail}")
    elif "no customer email" in detail:
        st.warning("Decision saved. This customer gave no email address, so the reply is shown in "
                   "'My tickets' only.")
    else:
        st.info(detail or "Decision saved.")


def escalations_tab(client: SupportApiClient) -> None:
    """Tickets support is handling directly - finish them here."""
    try:
        open_items = client.escalations()
    except ApiError as exc:
        st.error(str(exc))
        return
    if not open_items:
        st.caption("No escalated tickets are being handled manually right now.")
        return
    st.caption("When you have solved the problem, either send the resolution so the customer can confirm it, "
               "or close the ticket directly (for example after fixing it on a phone call).")
    for t in open_items:
        tid = t["ticket_id"]
        with st.expander(f"{tid} | {t['customer_name']} | {t.get('urgency') or '-'} urgency", expanded=True):
            st.markdown(f"**Complaint**")
            st.text(t["complaint"])
            if t.get("escalation_reasons"):
                st.markdown("**Escalation reasons**")
                for r in t["escalation_reasons"]:
                    st.markdown(f"- {r}")
            if t.get("customer_feedback"):
                st.markdown(f"**Customer feedback:** {t['customer_feedback']}")
            note = st.text_area("Resolution (sent to the customer)", key=f"res_{tid}", height=140,
                                placeholder="e.g. We replaced the charging cable and the headphones now pair normally.")
            notify = st.checkbox("Email the customer when closing", value=True, key=f"notify_{tid}")
            c1, c2 = st.columns(2)
            try:
                if c1.button("Send resolution - ask customer to confirm", key=f"conf_{tid}", width="stretch"):
                    if not note.strip():
                        st.error("Write the resolution first.")
                    else:
                        _show_decision(client.resolve(tid, "confirm", note))
                if c2.button("Mark resolved and close", key=f"close_{tid}", width="stretch"):
                    _show_decision(client.resolve(tid, "close", note, notify))
            except ApiError as exc:
                st.error(str(exc))


def admin_page() -> None:
    import altair as alt

    st.header("Admin console")
    client = api()
    if not store()["token"]:   # defence in depth: the API also rejects calls without a token
        st.error("Sign in as admin to view this page.")
        return
    tab_dash, tab_log, tab_appr, tab_esc, tab_kb, tab_graph = st.tabs(
        ["Dashboard", "Ticket log", "Approvals", "Escalations", "Vector chunks", "Knowledge graph"])
    with tab_log:
        ticket_log_tab(client)
    with tab_esc:
        escalations_tab(client)
    with tab_dash:
        try:
            d = client.dashboard()
        except ApiError as exc:
            st.error(str(exc))
            d = None
        if d:
            esc = d["status"].get("Escalated", 0)
            res = d["status"].get("Resolved", 0)
            pend = d["status"].get("Pending Confirmation", 0)
            closed = d["status"].get("Closed - No Response", 0) + d["status"].get("Closed - Not Supported", 0)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Tickets", d["total"])
            m2.metric("Resolved (confirmed)", res)
            m3.metric("Escalated", esc)
            m4.metric("Awaiting customer", pend)
            m5.metric("Average urgency (1 = Low, 3 = High)", d["avg_urgency"])
            c1, c2 = st.columns(2)
            cat = pd.DataFrame(d["per_category"].items(), columns=["category", "tickets"])
            c1.markdown("**Tickets per category**")
            c1.altair_chart(alt.Chart(cat).mark_bar(color="#2F8F8A").encode(
                x=alt.X("tickets:Q", axis=alt.Axis(tickMinStep=1)), y=alt.Y("category:N", sort="-x", title=None)
            ).properties(height=240), width="stretch")
            order_ = ["Resolved", "Escalated", "Awaiting customer", "Closed (no reply / not supported)"]
            rv = pd.DataFrame({"status": order_, "tickets": [res, esc, pend, closed]})
            c2.markdown("**Resolved vs escalated**")
            c2.altair_chart(alt.Chart(rv).mark_bar().encode(
                x=alt.X("status:N", title=None, sort=order_, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("tickets:Q", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("status:N", scale=alt.Scale(domain=order_,
                                                            range=["#1E8A4C", "#C2372C", "#F2A33A", "#6B7F90"]),
                                legend=None)
            ).properties(height=240), width="stretch")
    with tab_appr:
        try:
            pending = client.approvals()
        except ApiError as exc:
            st.error(str(exc))
            pending = []
        if not pending:
            st.caption("No escalated replies are waiting for approval.")
        for t in pending:
            with st.expander(f"{t['ticket_id']} | {t['customer_name']} | {t.get('urgency')} urgency", expanded=True):
                st.markdown("**Complaint**")
                st.text(t["complaint"])
                st.markdown("**Escalation reasons**")
                for r in t.get("escalation_reasons", []):
                    st.markdown(f"- {r}")
                edited = st.text_area("Reply to send (edit if needed)", value=t.get("reply_body") or "",
                                      key=f"edit_{t['ticket_id']}", height=200)
                note = st.text_input("Note (for rejections)", key=f"note_{t['ticket_id']}")
                b1, b2, b3 = st.columns(3)
                try:
                    if b1.button("Approve and send", key=f"a_{t['ticket_id']}"):
                        r = client.decide(t["ticket_id"], "approve")
                        _show_decision(r)
                    if b2.button("Send my edited reply", key=f"e_{t['ticket_id']}"):
                        r = client.decide(t["ticket_id"], "edit", edited_reply=edited)
                        _show_decision(r)
                    if b3.button("Handle it myself", key=f"r_{t['ticket_id']}",
                                 help="No automatic reply. The ticket moves to the Escalations tab, "
                                      "where you close it once you have solved it."):
                        r = client.decide(t["ticket_id"], "reject", note=note or "handled manually")
                        _show_decision(r)
                except ApiError as exc:
                    st.error(str(exc))
    products = []
    try:
        products = client.products()
    except ApiError:
        pass
    names = {p["slug"]: p["name"] for p in products}
    with tab_kb:
        if not names:
            st.caption("No product indexes found. Run the ingestion step first.")
        else:
            slug = st.selectbox("Product", list(names), format_func=names.get, key="kb_prod")
            q = st.text_input("Search chunks (leave empty to browse)", key="kb_q")
            try:
                rows = client.chunks(slug, q or None, 15)
                st.dataframe(pd.DataFrame(rows)[["section_title", "text", "source_file", "kind", "language",
                                                 "chunk_id", "prev_id", "next_id"]],
                             width="stretch", hide_index=True)
            except ApiError as exc:
                st.error(str(exc))
            tid = st.text_input("Inspect what a ticket retrieved (ticket ID)", key="kb_tid")
            if tid.strip():
                try:
                    st.json(client.ticket_retrieval(tid.strip()))
                except ApiError as exc:
                    st.error(str(exc))
    with tab_graph:
        if names:
            slug = st.selectbox("Product", list(names), format_func=names.get, key="g_prod")
            try:
                st.dataframe(pd.DataFrame(client.graph(slug, 100)), width="stretch", hide_index=True)
            except ApiError as exc:
                st.error(str(exc))


# --------------------------------------------------------------------------- #
def main() -> None:
    store()
    inject_css()
    restore_admin_session()
    sync_admin_cookie()
    admin_login_sidebar()
    pages = ["Get help", "My tickets"]
    if store()["token"]:
        pages.append("Admin console")  # dashboard, ticket log and KB views are admin-only
    with st.sidebar:
        page = st.radio("Go to", pages, label_visibility="collapsed")
    {"Get help": support_page, "My tickets": my_tickets_page,
     "Admin console": admin_page}[page]()


main()
