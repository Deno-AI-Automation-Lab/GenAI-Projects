"""CrewAI web-search tool for questions the FiberNet handbook doesn't cover.

Provider:
- Tavily  (if TAVILY_API_KEY is set in .env) - better quality, built for LLM agents
- DuckDuckGo (default, no API key needed) - via the `ddgs` package
"""
from typing import List, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import config


class WebSearchInput(BaseModel):
    query: str = Field(..., description="A focused web search query, e.g. "
                                        "'Hikvision NVR port forwarding steps'.")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Searches the public web for information NOT covered by the FiberNet handbook, such as "
        "device-specific setup (routers, cameras, mesh systems), general networking how-tos and "
        "manufacturer documentation. Returns titles, URLs and snippets. Prefer official vendor or "
        "reputable sources. Do NOT use it for FiberNet's own prices, plans, billing or SLA - those "
        "are not on the public web."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    max_results: int = config.WEB_MAX_RESULTS
    # Every result seen during a run, so the UI can list web sources
    results: List[dict] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        try:
            hits = self._tavily(query) if config.TAVILY_API_KEY else self._duckduckgo(query)
        except Exception as e:  # never crash the crew because search failed
            return f"Web search failed ({e}). Answer without web results and say so."
        if not hits:
            return "No web results found."

        lines = []
        for h in hits:
            if not any(r["url"] == h["url"] for r in self.results):
                self.results.append(h)
            lines.append(f"Title: {h['title']}\nURL: {h['url']}\nSnippet: {h['snippet']}")
        return "\n\n---\n\n".join(lines)

    def _tavily(self, query: str) -> List[dict]:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            json={"query": query, "max_results": self.max_results, "search_depth": "basic"},
            timeout=20,
        )
        resp.raise_for_status()
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": (r.get("content") or "")[:500]}
                for r in resp.json().get("results", [])]

    def _duckduckgo(self, query: str) -> List[dict]:
        from ddgs import DDGS
        raw = DDGS().text(query, region=config.WEB_REGION, max_results=self.max_results)
        return [{"title": r.get("title", ""), "url": r.get("href", ""),
                 "snippet": (r.get("body") or "")[:500]} for r in raw]

    def reset(self) -> None:
        self.results = []
