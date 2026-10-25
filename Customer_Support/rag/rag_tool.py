"""CrewAI tool that searches the FiberNet handbook through a LangChain FAISS retriever."""
from typing import List, Type

from crewai.tools import BaseTool
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, ConfigDict, Field

import config


class HandbookSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="A focused search query, e.g. 'LOS red light blinking what to do' "
                    "or 'speed test method wired vs Wi-Fi'.",
    )


class FiberNetHandbookTool(BaseTool):
    name: str = "fibernet_handbook_search"
    description: str = (
        "Searches the official FiberNet Broadband Customer Support Troubleshooting Handbook. "
        "Use it for ONT/ONU LED meanings, LOS/optical faults, no-internet workflows, slow speed, "
        "latency/packet loss, Wi-Fi coverage, DNS, router reboot/reset safety, hardware/power issues, "
        "escalation rules and case-note templates. Returns passages with page and section references. "
        "Call it several times with different queries if the issue spans multiple areas."
    )
    args_schema: Type[BaseModel] = HandbookSearchInput

    model_config = ConfigDict(arbitrary_types_allowed=True)
    vectorstore: FAISS
    k: int = config.TOP_K
    # Every passage retrieved during a run, so the UI can show citations
    retrieved: List[dict] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        if not results:
            return "No relevant passages found in the handbook."

        blocks = []
        for doc, score in results:
            meta = doc.metadata
            ref = {
                "page": meta.get("page"),
                "section": meta.get("section"),
                "chunk_id": meta.get("chunk_id"),
                "score": round(float(score), 4),  # L2 distance: lower = closer
                "text": doc.page_content,
            }
            if not any(r["chunk_id"] == ref["chunk_id"] for r in self.retrieved):
                self.retrieved.append(ref)
            blocks.append(f"[Page {ref['page']} | {ref['section']}]\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)

    def reset(self) -> None:
        self.retrieved = []
