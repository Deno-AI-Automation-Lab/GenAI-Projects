"""CrewAI crew for FiberNet support.

Agents
  1. Broadband Triage Analyst       (handbook tool)
  2. FTTH Troubleshooting Engineer  (handbook tool)
  3. Web Research Agent             (web search tool)  <- for questions the handbook doesn't cover
  4. Customer Support Specialist    (handbook tool)    -> final SupportResponse

Routing (decided by the scope check in rag/scope_guard.py)
  mode="handbook" (in scope)     : 1 -> 2 -> 4
  mode="hybrid"   (partial)      : 1 -> 2 -> 3 -> 4
  mode="web"      (out of scope) : 3 -> 4
"""
from typing import Literal

from crewai import LLM, Agent, Crew, Process, Task

import config
from crew.schemas import SupportResponse
from rag.rag_tool import FiberNetHandbookTool
from rag.web_search_tool import WebSearchTool

Mode = Literal["handbook", "hybrid", "web"]

GROUNDING_RULE = (
    "Base every technical statement on the FiberNet handbook (use the fibernet_handbook_search tool). "
    "If the handbook does not cover something, say so instead of inventing values. "
    "Never invent optical dBm thresholds, PPPoE/VLAN values or passwords. "
    "Never recommend a factory reset as an early step. "
    "If the handbook has no guidance for the request, say so plainly - never substitute an "
    "unrelated troubleshooting workflow to fill the gap."
)

WEB_RULES = (
    "Only answer questions related to internet/broadband, home networking, Wi-Fi, routers, connected "
    "devices (cameras, smart TVs, IoT), VPN or general tech usage. Politely decline anything unrelated "
    "(sports, news, politics, etc.). "
    "FiberNet's own prices, plans, billing, refunds, SLAs, relocation charges and policies are NOT on "
    "the public web: never guess them from other ISPs - say the customer must contact FiberNet. "
    "Prefer official manufacturer documentation and reputable sources; cite the URLs you used. "
    "If sources disagree or are unclear, say so."
)


def _llm() -> LLM:
    config.require_api_key()
    return LLM(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY,
               temperature=config.LLM_TEMPERATURE)


def build_crew(handbook_tool: FiberNetHandbookTool, web_tool: WebSearchTool | None = None,
               mode: Mode = "handbook", verbose: bool = False) -> Crew:
    if mode != "handbook" and web_tool is None:
        raise ValueError("web_tool is required for hybrid/web mode")
    llm = _llm()

    # ---------------- Agents ----------------
    triage_agent = Agent(
        role="Broadband Triage Analyst",
        goal="Classify the customer's broadband problem by impact, fault domain and priority "
             "using the handbook's layered troubleshooting model.",
        backstory="A senior L1/L2 service-desk analyst at FiberNet who always checks impact and "
                  "outage status first, then reads ONT/router state before touching settings. "
                  + GROUNDING_RULE,
        tools=[handbook_tool], llm=llm, allow_delegation=False, verbose=verbose, max_iter=6,
    )

    troubleshooter_agent = Agent(
        role="FTTH Troubleshooting Engineer",
        goal="Produce a safe, ordered troubleshooting plan for the identified fault domain and "
             "decide whether NOC or field escalation is needed.",
        backstory="An L2 FTTH engineer who knows GPON/XGS-PON, ONT LEDs, PPPoE/DHCP, Wi-Fi and DNS. "
                  "Isolates one layer at a time, preserves configuration and documents evidence. "
                  + GROUNDING_RULE,
        tools=[handbook_tool], llm=llm, allow_delegation=False, verbose=verbose, max_iter=8,
    )

    web_research_agent = Agent(
        role="Web Research Agent",
        goal="Find accurate, up-to-date answers on the public web for customer questions that the "
             "FiberNet handbook does not cover, and report them with source URLs.",
        backstory="A meticulous tech-support researcher who reads manufacturer documentation and "
                  "reputable guides, summarises only what sources actually say, and clearly separates "
                  "general web guidance from FiberNet's official policy. " + WEB_RULES,
        tools=[web_tool] if web_tool else [], llm=llm, allow_delegation=False, verbose=verbose,
        max_iter=6,
    )

    communicator_agent = Agent(
        role="Customer Support Specialist",
        goal="Turn the findings into a clear, empathetic reply the customer can follow, "
             "plus accurate internal case notes and escalation details.",
        backstory="A FiberNet support specialist known for plain-language explanations. "
                  "Avoids jargon (or explains it), never blames customer equipment before isolation, "
                  "never asks for or exposes passwords, and never presents web information as FiberNet "
                  "policy. " + GROUNDING_RULE,
        tools=[handbook_tool], llm=llm, allow_delegation=False, verbose=verbose, max_iter=4,
    )

    # ---------------- Tasks ----------------
    triage_task = Task(
        description=(
            "CURRENT customer message (this is what you must answer):\n\"{customer_message}\"\n\n"
            "Scope check result: {scope_note}\n\n"
            "Known details from the agent's form:\n{customer_context}\n\n"
            "Earlier conversation (context only - use it ONLY if the current message is a follow-up; "
            "if the current message is a new topic, ignore it):\n{chat_history}\n\n"
            "1. Determine impact: all devices / one device / Wi-Fi only / wired only / one app.\n"
            "2. Map the symptoms to the most likely fault domain using the handbook's fault-domain "
            "quick map and LED diagnosis table.\n"
            "3. Assign priority P1-P4 using the handbook's severity classification.\n"
            "4. Pick ticket tags from the handbook's recommended tag list.\n"
            "5. List the key facts still missing (e.g. ONT LED colours, wired vs Wi-Fi behaviour)."
        ),
        expected_output="Impact, fault domain with reasoning, priority, ticket tags, missing information, "
                        "and the handbook pages consulted.",
        agent=triage_agent,
    )

    troubleshoot_task = Task(
        description=(
            "Scope check result: {scope_note}\n"
            "If coverage is partial, list only the handbook's relevant general checks and skip unrelated "
            "workflows. Otherwise, using the triage result, search the handbook for the matching workflow "
            "and write an ordered troubleshooting plan for this customer (most likely cause first, safe "
            "customer-side checks before disruptive ones). Include what each result would mean. "
            "Decide whether escalation is needed (NOC vs Field technician) using the handbook's "
            "escalation criteria; if yes, fill the handbook's escalation template fields. "
            "Respect safety rules: no looking into fibre connectors, no random power adapters, "
            "no factory reset without authorisation."
        ),
        expected_output="Numbered troubleshooting steps with expected outcomes, escalation decision with "
                        "justification, filled escalation template if needed, and handbook pages used.",
        agent=troubleshooter_agent,
        context=[triage_task],
    )

    web_task = Task(
        description=(
            "CURRENT customer message:\n\"{customer_message}\"\n\n"
            "Scope check result: {scope_note}\n\n"
            "Earlier conversation (context only, use only for follow-ups):\n{chat_history}\n\n"
            "The FiberNet handbook does not (fully) answer this. Use the web_search tool (1-3 focused "
            "queries) to find the missing information.\n"
            "- If the question is unrelated to broadband/networking/devices, do NOT search: report "
            "'UNRELATED TOPIC'.\n"
            "- If it asks about FiberNet-specific prices, plans, billing, refunds, SLA or policies, do NOT "
            "search: report 'FIBERNET-SPECIFIC - route to FiberNet team'.\n"
            "- Otherwise summarise the answer as clear steps or facts, note any caveats (model/firmware "
            "differences, security risks such as exposing cameras to the internet), and list the URLs used."
        ),
        expected_output="Either 'UNRELATED TOPIC', 'FIBERNET-SPECIFIC - route to FiberNet team', or a "
                        "concise answer with caveats and a list of source URLs.",
        agent=web_research_agent,
        context=[triage_task, troubleshoot_task] if mode == "hybrid" else [],
    )

    source_rules = {
        "handbook": "Set answer_source='FiberNet handbook' and web_sources=[].",
        "hybrid": (
            "Combine both: first what the FiberNet handbook says, then a section titled "
            "'From the web (general guidance, not FiberNet policy)' with the web findings. "
            "Set answer_source='Handbook + web' and copy the web URLs into web_sources."
        ),
        "web": (
            "The handbook does not cover this. Start the reply by saying the FiberNet guide doesn't cover "
            "it, then give the web findings under 'General guidance from the web (not FiberNet policy)'. "
            "Set answer_source='Web search', fault_domain='Out of scope', priority='P4', "
            "ticket_tags=['OUT-OF-SCOPE'], copy the web URLs into web_sources. "
            "If the researcher reported UNRELATED TOPIC, politely say you can only help with broadband "
            "and connected-device questions and set answer_source='Not answered'. "
            "If it reported FIBERNET-SPECIFIC, tell the customer to contact the relevant FiberNet team "
            "(billing/sales/customer care) and set answer_source='Not answered'."
        ),
    }[mode]

    respond_task = Task(
        description=(
            "Write the final response for the support agent's console. Answer the CURRENT customer "
            "message: \"{customer_message}\" - not an earlier topic.\n"
            "Scope check result: {scope_note}\n"
            f"{source_rules}\n"
            "- customer_reply: warm, concise, plain language; acknowledge the specific situation; give "
            "the next 3-6 steps the customer can do now, each with a one-line reason; if information is "
            "missing, ask for it; if escalating, set realistic expectations without promising times "
            "that no source gives. Never invent steps that neither the handbook nor the web sources state.\n"
            "- case_note: Problem / Environment / Observations / Actions / Next step (no passwords).\n"
            "- Carry over fault_domain, priority, ticket_tags, escalation fields and handbook references "
            "from the previous tasks where they exist."
        ),
        expected_output="A SupportResponse object.",
        agent=communicator_agent,
        context={"handbook": [triage_task, troubleshoot_task],
                 "hybrid": [triage_task, troubleshoot_task, web_task],
                 "web": [web_task]}[mode],
        output_pydantic=SupportResponse,
    )

    plan = {
        "handbook": ([triage_agent, troubleshooter_agent, communicator_agent],
                     [triage_task, troubleshoot_task, respond_task]),
        "hybrid": ([triage_agent, troubleshooter_agent, web_research_agent, communicator_agent],
                   [triage_task, troubleshoot_task, web_task, respond_task]),
        "web": ([web_research_agent, communicator_agent], [web_task, respond_task]),
    }[mode]

    return Crew(agents=plan[0], tasks=plan[1], process=Process.sequential, verbose=verbose)


def run_support_crew(handbook_tool: FiberNetHandbookTool, customer_message: str,
                     customer_context: str = "Not provided", chat_history: str = "None",
                     scope_note: str = "IN SCOPE: the handbook covers this request.",
                     mode: Mode = "handbook", web_tool: WebSearchTool | None = None,
                     verbose: bool = False) -> SupportResponse:
    handbook_tool.reset()
    if web_tool:
        web_tool.reset()
    crew = build_crew(handbook_tool, web_tool, mode=mode, verbose=verbose)
    result = crew.kickoff(inputs={
        "customer_message": customer_message,
        "customer_context": customer_context or "Not provided",
        "chat_history": chat_history or "None",
        "scope_note": scope_note,
    })
    if result.pydantic is not None:
        resp = result.pydantic
    else:  # fallback if the model didn't return valid structured output
        resp = SupportResponse(customer_reply=result.raw, fault_domain="Unclear - more info needed",
                               priority="P3")
    # Use the URLs the tool actually returned, not ones the model may have typed
    if web_tool and web_tool.results and resp.answer_source in ("Handbook + web", "Web search"):
        seen = {r["url"] for r in web_tool.results}
        resp.web_sources = [u for u in resp.web_sources if u in seen] or [r["url"] for r in web_tool.results]
    elif not web_tool or not web_tool.results:
        resp.web_sources = []
    return resp
