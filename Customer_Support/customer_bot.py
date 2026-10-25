"""FiberNet Broadband – AI Customer Support Agent (Streamlit UI).

Run:  streamlit run app.py
"""
import time
import uuid

import streamlit as st

import config

st.set_page_config(page_title="FiberNet Support Agent", page_icon="📶", layout="wide")

# ---------- Cached resources ----------

@st.cache_resource(show_spinner="Loading the handbook index (FAISS)...")
def get_vectorstore():
    from rag.vectorstore import load_vectorstore
    return load_vectorstore()


def get_tool():
    from rag.rag_tool import FiberNetHandbookTool
    return FiberNetHandbookTool(vectorstore=get_vectorstore())


@st.cache_resource(show_spinner="Connecting to Google Sheets...")
def get_sheet_logger():
    from sheet_logger import get_logger
    return get_logger()   # None if not configured


def log_to_sheet(**kwargs) -> None:
    """Write one row; never let a Sheets problem break the chat."""
    try:
        logger = get_sheet_logger()
        if logger:
            logger.log(case_id=st.session_state.case_id, turn=st.session_state.turn, **kwargs)
    except Exception as e:
        st.toast(f"⚠️ Couldn't log to Google Sheet: {e}")


def get_web_tool():
    from rag.web_search_tool import WebSearchTool
    return WebSearchTool()


# ---------- Session state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role", "content", "result"?, "sources"?}
if "case_id" not in st.session_state:
    st.session_state.case_id = f"FN-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    st.session_state.turn = 0

# ---------- Sidebar ----------

with st.sidebar:
    st.title("📶 FiberNet Support")
    st.caption(f"LLM: `{config.LLM_MODEL}` · Embeddings: `{config.EMBEDDING_MODEL}`")

    if not config.OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found in .env")
        st.stop()

    st.subheader("Customer details (optional)")
    st.caption("Fill what you know — it helps triage.")
    affected = st.selectbox("Who is affected?", ["Unknown", "All devices", "One device only",
                                                 "Wi-Fi devices only", "Wired only", "One app / website"])
    connection = st.selectbox("Connection used", ["Unknown", "Wi-Fi", "Ethernet", "Both"])
    ont_power = st.selectbox("ONT power LED", ["Unknown", "On", "Off"])
    ont_pon = st.selectbox("ONT PON LED", ["Unknown", "Solid", "Blinking", "Off"])
    ont_los = st.selectbox("ONT LOS LED", ["Unknown", "Off", "Red solid", "Red blinking"])
    router_wan = st.selectbox("Router WAN / Internet LED", ["Unknown", "On", "Off / Red", "Blinking"])
    pattern = st.selectbox("Pattern", ["Unknown", "Continuous", "Intermittent", "Peak hours only"])
    plan = st.text_input("Subscribed plan (e.g. 300 Mbps)")
    notes = st.text_area("Recent changes / notes", placeholder="Power cut, router moved, renovation...")

    st.divider()
    use_web = st.toggle("🌐 Web search for questions not in the handbook",
                        value=config.ENABLE_WEB_SEARCH,
                        help="Adds the Web Research Agent. Uses Tavily if TAVILY_API_KEY is in .env, "
                             "otherwise DuckDuckGo.")
    if use_web:
        st.caption(f"Web provider: {'Tavily' if config.TAVILY_API_KEY else 'DuckDuckGo'}")
    verbose = st.toggle("Show agent reasoning in terminal", value=False)

    if config.GSHEET_ENABLED and config.GSHEET_ID:
        try:
            ok = get_sheet_logger() is not None
            st.caption(f"📊 Logging queries to Google Sheet · Case `{st.session_state.case_id}`"
                       if ok else "📊 Google Sheet logging not configured")
        except Exception as e:
            st.warning(f"Google Sheet logging unavailable: {e}")
    else:
        st.caption("📊 Google Sheet logging off (set GSHEET_ID in .env)")
    c1, c2 = st.columns(2)
    if c1.button("🆕 New case", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("case_id", None)
        st.rerun()
    if c2.button("🔄 Rebuild index", use_container_width=True):
        from rag.vectorstore import build_vectorstore
        with st.spinner("Re-indexing the handbook..."):
            build_vectorstore()
        get_vectorstore.clear()
        st.success("Index rebuilt.")


def customer_context() -> str:
    fields = {
        "Affected": affected, "Connection": connection, "ONT power LED": ont_power,
        "ONT PON LED": ont_pon, "ONT LOS LED": ont_los, "Router WAN LED": router_wan,
        "Pattern": pattern, "Plan": plan or "Unknown", "Notes": notes or "None",
    }
    known = {k: v for k, v in fields.items() if v not in ("Unknown", "None", "")}
    return "\n".join(f"- {k}: {v}" for k, v in known.items()) or "Not provided"


def chat_history(max_turns: int = 6) -> str:
    lines = []
    for m in st.session_state.messages[-max_turns * 2:]:
        who = "Customer" if m["role"] == "user" else "Support"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines) or "None"


# ---------- Rendering helpers ----------

PRIORITY_COLOR = {"P1": "red", "P2": "orange", "P3": "blue", "P4": "green"}  # P4 also used for out-of-scope


SOURCE_BADGE = {"FiberNet handbook": ":blue[📖 FiberNet handbook]",
                "Handbook + web": ":violet[📖 Handbook + 🌐 web]",
                "Web search": ":orange[🌐 Web search – general guidance, not FiberNet policy]",
                "Not answered": ":gray[⛔ Not answered]"}


def render_details(result, sources):
    st.markdown(f"**Answer source:** {SOURCE_BADGE.get(result.answer_source, result.answer_source)}")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Fault domain**  \n{result.fault_domain}")
    color = PRIORITY_COLOR.get(result.priority, "gray")
    c2.markdown(f"**Priority**  \n:{color}[**{result.priority}**]")
    esc = f"Yes → {result.escalation_team}" if result.escalation_required else "No"
    c3.markdown(f"**Escalation**  \n{esc}")
    if result.ticket_tags:
        st.markdown(" ".join(f"`{t}`" for t in result.ticket_tags))

    if result.follow_up_questions:
        with st.expander("❓ Information still needed"):
            for q in result.follow_up_questions:
                st.markdown(f"- {q}")
    if result.escalation_required and result.escalation_note:
        with st.expander(f"🚨 Escalation note ({result.escalation_team})", expanded=True):
            st.code(result.escalation_note, language=None)
    if result.case_note:
        with st.expander("📝 Internal case note"):
            st.code(result.case_note, language=None)
    if result.web_sources:
        with st.expander(f"🌐 Web sources ({len(result.web_sources)})", expanded=True):
            for url in result.web_sources:
                st.markdown(f"- {url}")
    if sources:
        with st.expander(f"📖 Handbook passages used ({len(sources)})"):
            for s in sorted(sources, key=lambda x: (x["page"] or 0)):
                st.markdown(f"**Page {s['page']} — {s['section']}**")
                st.caption(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))


# ---------- Main chat ----------

st.header("FiberNet Broadband – AI Support Agent")
st.caption("CrewAI agents (Triage → Troubleshooting → Web Research → Customer reply) grounded in the "
           "FiberNet Troubleshooting Handbook via LangChain + FAISS, with web search for uncovered questions.")

if not st.session_state.messages:
    st.info("Describe the customer's issue. Examples:\n"
            "- *My internet is down on all devices and the box has a blinking red light.*\n"
            "- *Speed is only 90 Mbps on my 300 Mbps plan even on the laptop cable.*\n"
            "- *Wi-Fi shows connected but nothing loads on my phone.*")

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "📶"):
        st.markdown(m["content"])
        if m.get("result"):
            render_details(m["result"], m.get("sources", []))

if prompt := st.chat_input("Describe the broadband issue..."):
    history = chat_history()
    st.session_state.turn += 1
    started = time.time()
    decision, mode, result, sources = None, None, None, []
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📶"):
        try:
            from crew.support_crew import run_support_crew
            from rag.scope_guard import check_scope, out_of_scope_response, scope_note
            tool = get_tool()
            with st.status("Checking the question against the handbook...", expanded=False) as status:
                decision, guard_docs = check_scope(get_vectorstore(), prompt, history)
                st.write(f"🧭 Scope check: **{decision.verdict.replace('_', ' ')}** – {decision.reason}")
                mode = {"in_scope": "handbook", "partial": "hybrid", "out_of_scope": "web"}[decision.verdict]
                if mode != "handbook" and not use_web:
                    mode = "handbook" if mode == "hybrid" else None   # web disabled

                if mode is None:
                    result = out_of_scope_response(decision)
                    sources = []
                    status.update(label="Not covered by the handbook", state="complete")
                else:
                    status.update(label="Agents are working on the case...")
                    if mode in ("handbook", "hybrid"):
                        st.write("🔎 Triage Analyst – classifying impact and fault domain")
                        st.write("🛠️ Troubleshooting Engineer – building the step plan from the handbook")
                    if mode in ("hybrid", "web"):
                        st.write("🌐 Web Research Agent – searching the web for what the handbook lacks")
                    st.write("💬 Support Specialist – writing the customer reply")
                    web_tool = get_web_tool() if mode != "handbook" else None
                    result = run_support_crew(tool, prompt, customer_context(), history,
                                              scope_note=scope_note(decision), mode=mode,
                                              web_tool=web_tool, verbose=verbose)
                    sources = list(tool.retrieved)
                    status.update(label=f"Done ({decision.verdict.replace('_', ' ')} → {mode} agents)",
                                  state="complete")
            st.markdown(result.customer_reply)
            render_details(result, sources)
            st.session_state.messages.append({"role": "assistant", "content": result.customer_reply,
                                              "result": result, "sources": sources})
            log_to_sheet(query=prompt, customer_details=customer_context(), decision=decision,
                         mode=mode, result=result, sources=sources,
                         elapsed_s=time.time() - started)
        except Exception as e:  # show a readable error instead of a stack trace
            st.error(f"Something went wrong: {e}")
            log_to_sheet(query=prompt, customer_details=customer_context(), decision=decision,
                         mode=mode, result=result, sources=sources,
                         elapsed_s=time.time() - started, error=str(e))
