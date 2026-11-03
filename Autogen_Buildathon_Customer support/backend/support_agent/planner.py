"""Query planning: product routing (RAG), structured QueryPlan, triage signals.

Pure functions – no Streamlit. Indexes/LLMs are passed in so tests can inject fakes.
"""
from __future__ import annotations

import json
import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .config import (CATEGORIES, CATEGORY_TO_DOCTYPE, DOC_TYPES, LEGAL_FRAUD_PAYMENT_TERMS,
                     PLANNER_SYSTEM, Settings, tokenize)
from .language import detect_language, language_name
from .models import QueryAnalysis, QueryPlan

log = logging.getLogger(__name__)


@dataclass
class RouterIndex:
    """Product catalogue: FAISS over product profiles + product list."""

    store: Any | None            # langchain FAISS (None if not built yet)
    products: list[dict]         # [{"name":..., "slug":...}]


def load_router(settings: Settings, embeddings=None) -> RouterIndex:
    """Load the router index written by ingest.build_router_index."""
    from langchain_community.vectorstores import FAISS

    from .ingest import ROUTER_DIR_NAME
    from .llm import get_embeddings

    d = settings.index_dir / ROUTER_DIR_NAME
    products = json.loads((d / "products.json").read_text(encoding="utf-8")) if (d / "products.json").exists() else []
    store = None
    if (d / "index.faiss").exists():
        try:
            # allow_dangerous_deserialization: the pickle is produced by our own ingest step only
            store = FAISS.load_local(str(d), embeddings or get_embeddings(), allow_dangerous_deserialization=True)
        except Exception as exc:  # routing then relies on exact product-name matching + planner
            log.error("Router FAISS unavailable: %s", exc)
    _add_store_aliases(settings, products)
    return RouterIndex(store=store, products=products)


_STORE_RE = (re.compile(r"Store:\s*([A-Z][A-Za-z]+)"), re.compile(r"\b([A-Z][a-z]+[A-Z][A-Za-z]+) Online\b"))


def _add_store_aliases(settings: Settings, products: list[dict]) -> None:
    """Store/brand names from each product's own documents ("Store: SoundSphere Online",
    "HomePilot Online | ...") become aliases, so "my SoundSphere account" routes to AeroBeat.
    A brand used by more than one product is ignored (it would not identify the product)."""
    found: dict[str, set[str]] = {}
    for p in products:
        try:
            recs = json.loads((settings.index_dir / p["slug"] / "records.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        text = " ".join(f"{r.get('title', '')} {r.get('text', '')}" for r in recs[:60])
        found[p["slug"]] = {m.lower() for rx in _STORE_RE for m in rx.findall(text)}
    for p in products:
        mine = found.get(p["slug"], set())
        others = set().union(*(v for k, v in found.items() if k != p["slug"])) if len(found) > 1 else set()
        p["aliases"] = sorted(mine - others)


# Words that describe a product TYPE, not a specific product: they must never decide routing on
# their own ("Philips air purifier" is not the SmartCrisp air fryer). Type words are handled by
# product_evidence together with the planner instead.
_GENERIC_NAME_WORDS = {"pro", "and", "the", "wi", "fi", "wireless", "air", "fryer", "robot", "vacuum", "mop",
                       "noise", "cancelling", "canceling", "headphones", "headphone", "smart", "max", "mini"}


def exact_product_mention(text: str, products: list[dict]) -> str | None:
    """Deterministic routing when the customer names the product (e.g. 'AeroBeat', 'X9')."""
    toks = set(tokenize(text))
    best, best_hits = None, 0
    for p in products:
        name_toks = [t for t in tokenize(p["name"]) if len(t) > 2 or any(ch.isdigit() for ch in t)]
        distinctive = [t for t in name_toks if t not in _GENERIC_NAME_WORDS]   # brand / model only
        distinctive += [a for a in p.get("aliases", []) if a not in _GENERIC_NAME_WORDS]
        hits = sum(1 for t in distinctive if t in toks)
        if hits > best_hits:
            best, best_hits = p["slug"], hits
    return best if best_hits >= 1 else None


@traceable(name="route_product")
def route_product(router: RouterIndex, text: str, k: int = 8) -> tuple[dict[str, float], list[str]]:
    """RAG routing: similarity search over product profiles, votes summed per product."""
    votes: dict[str, float] = defaultdict(float)
    evidence: list[str] = []
    if router.store is None:
        return {}, []
    for rank, doc in enumerate(router.store.similarity_search(text, k=k)):
        slug = doc.metadata["product"]
        votes[slug] += 1.0 / (rank + 1)          # rank-weighted vote
        if len(evidence) < 6:
            evidence.append(f"- [{slug}] {doc.page_content[:200]}")
    return dict(votes), evidence


# Words that describe each product TYPE. A product is only selected when its name or one of
# its type words appears in the complaint. Keys are matched against words in product names.
PRODUCT_TYPE_WORDS: dict[str, tuple[str, ...]] = {
    "headphones": ("headphone", "headset", "earphone", "earbud", "ear cup", "earcup", "anc",
                   "noise cancel", "noise-cancel", "ஹெட்ஃபோன்", "ஹெட்போன்", "हेडफ़ोन", "हेडफोन"),
    "vacuum": ("vacuum", "robot", "mop", "sweeper", "cleaner", "dock", "suction",
               "வாக்யூம்", "ரோபோ", "वैक्यूम", "रोबोट"),
    "fryer": ("fryer", "air fryer", "airfryer", "basket", "ஃப்ரையர்", "பிரையர்", "फ्रायर"),
}


def product_evidence(text: str, product_name: str) -> bool:
    """True if the complaint names this product or describes its product type."""
    low = (text or "").lower()
    name_tokens = [t for t in tokenize(product_name) if len(t) >= 4 or any(c.isdigit() for c in t)]
    distinctive = [t for t in name_tokens if t not in {"wireless", "noise", "cancelling", "robot", "smart", "with"}]
    if any(t in set(tokenize(low)) for t in distinctive):
        return True
    name_low = product_name.lower()
    for type_key, words in PRODUCT_TYPE_WORDS.items():
        if type_key in name_low and any(_word_in(w, low) for w in words):
            return True
    return False


def _word_in(word: str, text: str) -> bool:
    """English words must start at a word boundary ('anc' must not match 'cancel');
    Tamil/Hindi words are matched as substrings (inflected suffixes)."""
    if word.isascii():
        return re.search(r"(?<![a-z0-9])" + re.escape(word), text) is not None
    return word in text


def detect_risk_flags(text: str) -> list[str]:
    """Keyword layer for legal threats / fraud / payment disputes (multilingual).

    English terms must match whole words, so 'sue' does not fire on 'issue' and 'court'
    not on 'courtesy'. Tamil/Hindi terms are matched as substrings because they take
    suffixes (e.g. மோசடியான)."""
    low = (text or "").lower()
    hits = set()
    for term in LEGAL_FRAUD_PAYMENT_TERMS:
        if term.isascii():
            if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", low):
                hits.add(term)
        elif term in low:
            hits.add(term)
    return sorted(hits)


@traceable(name="plan_query")
def plan_query(question: str, history: list[dict[str, str]], products: list[dict],
               router_evidence: list[str], settings: Settings, llm=None) -> QueryPlan:
    """Structured-output planning step (standalone rewrite, entities, filter, fusion queries)."""
    from .llm import get_chat_llm

    llm = (llm or get_chat_llm()).with_structured_output(QueryPlan)
    hist = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-8:]) or "(none)"
    system = PLANNER_SYSTEM.format(
        products=", ".join(f"{p['slug']} ({p['name']})" for p in products) or "none",
        router_evidence="\n".join(router_evidence) or "(no evidence)",
        categories=", ".join(CATEGORIES), n_queries=settings.fusion_queries,
    )
    plan: QueryPlan = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Chat history:\n{hist}\n\nLatest customer message:\n{question}"),
    ])
    return sanitize_plan(plan, products, settings)


def sanitize_plan(plan: QueryPlan, products: list[dict], settings: Settings) -> QueryPlan:
    """Clamp LLM output to allowed values so downstream code never sees surprises."""
    slugs = {p["slug"] for p in products}
    if plan.product not in slugs:
        plan.product = "unknown"
    if plan.category not in CATEGORIES:
        plan.category = "other"
    if plan.doc_type_filter not in DOC_TYPES:
        plan.doc_type_filter = CATEGORY_TO_DOCTYPE.get(plan.category) if plan.category in ("shipping", "refund", "account_login") else None
    qs = [q.strip() for q in plan.fusion_queries if q and q.strip()]
    plan.fusion_queries = list(dict.fromkeys([plan.standalone_question] + qs))[: settings.fusion_queries + 1]
    plan.entities = list(dict.fromkeys(e.strip() for e in plan.entities if e.strip()))[:10]
    return plan


def heuristic_plan(text: str, product: str) -> QueryPlan:
    """Fallback when the LLM is unavailable: keyword triage, no fusion variants."""
    low = text.lower()
    category = next((c for c, kws in {
        "refund": ("refund", "return", "money back"), "shipping": ("deliver", "shipping", "courier", "track"),
        "account_login": ("login", "password", "account", "otp"), "payment": ("payment", "charged", "card"),
        "technical_issue": ("not working", "error", "broken", "stopped"),
    }.items() if any(k in low for k in kws)), "other")
    negative = any(w in low for w in ("angry", "worst", "terrible", "useless", "furious", "disappointed"))
    return QueryPlan(
        standalone_question=text[:500], product=product, category=category,
        urgency="Medium", sentiment="Negative" if negative else "Neutral",
        summary=text[:200], entities=[], doc_type_filter=CATEGORY_TO_DOCTYPE.get(category),
        fusion_queries=[text[:500]],
    )


@traceable(name="analyze_complaint")
def analyze_complaint(text: str, router: RouterIndex, settings: Settings,
                      history: list[dict[str, str]] | None = None, llm=None) -> QueryAnalysis:
    """Customer Query Agent core: route product, plan, sentiment/urgency + risk rules."""
    lang = detect_language(text)
    exact = exact_product_mention(text, router.products)
    try:
        votes, evidence = route_product(router, text)
    except Exception as exc:
        log.warning("Router search failed: %s", exc)
        votes, evidence = {}, []
    try:
        plan = plan_query(text, history or [], router.products, evidence, settings, llm=llm)
    except Exception as exc:
        log.error("Planner LLM failed, using heuristic plan: %s", exc)
        # Without the LLM: exact brand/model, else the ONE product whose type is described
        typed = [p["slug"] for p in router.products if product_evidence(text, p["name"])]
        plan = heuristic_plan(text, exact or (typed[0] if len(typed) == 1 else "unknown"))

    # Product decision (safety-critical: a wrong product means a wrong policy in the reply).
    # 1) explicit product name/model -> that product
    # 2) the planner's choice is kept only if the complaint contains evidence for it
    #    (name or product type, e.g. "headphones", "robot vacuum", "air fryer")
    # 3) otherwise unknown - the router's nearest neighbour is NEVER used on its own,
    #    because similarity search always returns *some* product
    if exact:
        plan.product = exact
    elif plan.product != "unknown":
        chosen = next((p for p in router.products if p["slug"] == plan.product), None)
        if not chosen or not product_evidence(text, chosen["name"]):
            log.info("Planner chose %s without evidence in the complaint -> unknown", plan.product)
            plan.product = "unknown"

    flags = detect_risk_flags(text)
    if flags:
        plan.urgency = "High"
        if any(f in flags for f in ("chargeback", "double charged", "charged twice", "payment dispute",
                                    "dispute the charge", "unauthorized", "unauthorised")):
            plan.category = "payment"
    name = next((p["name"] for p in router.products if p["slug"] == plan.product), "")
    return QueryAnalysis(**plan.model_dump(), language=lang, language_name=language_name(lang),
                         risk_flags=flags, product_name=name)
