from typing import List, Literal

from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    """Structured output of the final task — rendered by the Streamlit UI."""

    customer_reply: str = Field(
        ..., description="Friendly, plain-language reply to the customer in markdown, "
                         "with numbered steps where relevant.")
    fault_domain: Literal[
        "Provider outage", "Optical access / fiber", "WAN / session", "Home LAN / Wi-Fi",
        "End device / application", "Unclear - more info needed", "Out of scope",
    ]
    priority: Literal["P1", "P2", "P3", "P4"]
    ticket_tags: List[str] = Field(
        default_factory=list,
        description="Tags from the handbook list, e.g. NO-INTERNET, LOS/OPTICAL, SLOW-SPEED.")
    follow_up_questions: List[str] = Field(
        default_factory=list, description="Questions still needed to isolate the fault.")
    escalation_required: bool = False
    escalation_team: Literal["None", "NOC", "Field technician"] = "None"
    escalation_note: str = Field(
        "", description="If escalating: filled handbook escalation template; else empty.")
    case_note: str = Field(
        "", description="Internal case note: Problem / Observations / Actions / Next step.")
    answer_source: Literal["FiberNet handbook", "Handbook + web", "Web search", "Not answered"] = Field(
        "FiberNet handbook", description="Where the answer came from.")
    web_sources: List[str] = Field(
        default_factory=list, description="URLs of web pages used, if any.")
    handbook_references: List[str] = Field(
        default_factory=list, description="Handbook pages/sections used, e.g. 'p.5 - 4. ONT/ONU LED Diagnosis'.")
