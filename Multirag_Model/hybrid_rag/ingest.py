"""Ingestion: crawling, PDF parsing, OCR, summarising, chunking and indexing.

No Streamlit here. Every step is a plain function so it can be run from
scripts (`python -m hybrid_rag.ingest file.pdf`) and tests.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import deque
from datetime import date
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from hybrid_rag import config
from hybrid_rag.config import Settings, TAMIL_CHAR, TOKEN_PATTERN

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def content_hash(data: bytes | str) -> str:
    """SHA-256 of raw content; used as the cache key and document id."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def detect_language(text: str) -> str:
    """'ta' when Tamil script dominates the letters, else 'en' (script-ratio check)."""
    letters = TOKEN_PATTERN.findall(text or "")
    total = sum(len(t) for t in letters)
    if not total:
        return "en"
    tamil = len(TAMIL_CHAR.findall(text))
    return "ta" if tamil / total > 0.3 else "en"


def _cache_get(kind: str, key: str, cache_dir: Path) -> str | None:
    path = cache_dir / f"{kind}_{key}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _cache_put(kind: str, key: str, value: str, cache_dir: Path) -> None:
    (cache_dir / f"{kind}_{key}.txt").write_text(value, encoding="utf-8")


# ---------------------------------------------------------------------------
# OCR + summaries (LLM calls are cached by content hash -> paid for once)
# ---------------------------------------------------------------------------

@traceable(name="ocr_page", run_type="llm")
def ocr_page_with_vision(png_bytes: bytes, vision_llm, cache_dir: Path = config.CACHE_DIR) -> str:
    """OCR a rendered page image with a vision LLM (GPT-4o), cached by image hash."""
    key = content_hash(png_bytes)
    cached = _cache_get("ocr", key, cache_dir)
    if cached is not None:
        return cached
    b64 = base64.b64encode(png_bytes).decode("ascii")
    msg = HumanMessage(content=[
        {"type": "text", "text": config.OCR_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])
    text = vision_llm.invoke([msg]).content.strip()
    _cache_put("ocr", key, text, cache_dir)
    return text


@traceable(name="key_facts_summary")
def summarise_key_facts(text: str, llm, cache_dir: Path = config.CACHE_DIR) -> str:
    """English KEY FACTS summary for a non-English document, cached by text hash."""
    key = content_hash(text)
    cached = _cache_get("facts", key, cache_dir)
    if cached is not None:
        return cached
    summary = llm.invoke(config.KEY_FACTS_PROMPT.format(text=text[:12000])).content.strip()
    summary = "KEY FACTS (English summary)\n" + summary
    _cache_put("facts", key, summary, cache_dir)
    return summary


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def extract_pdf_pages(pdf_path: Path, vision_llm=None, min_chars: int = 40,
                      cache_dir: Path = config.CACHE_DIR) -> list[dict]:
    """Return [{'page', 'text', 'ocr'}]. Pages with almost no text layer are
    treated as scanned and OCR'd with the vision LLM (if one is supplied)."""
    import pymupdf

    pages = []
    with pymupdf.open(Path(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            used_ocr = False
            if len(text) < min_chars and vision_llm is not None:
                png = page.get_pixmap(dpi=160).tobytes("png")
                text, used_ocr = ocr_page_with_vision(png, vision_llm, cache_dir), True
            pages.append({"page": i, "text": text, "ocr": used_ocr})
    return pages


def rows_to_records(header: list[str], rows: Iterable[list], base_meta: dict,
                    category_field: str = "department") -> list[Document]:
    """One labelled record per table row ("Header: value" lines). The first
    column is the entity name; a column matching `category_field` becomes the
    category metadata. Metadata fields stay separate from the free text."""
    header = [(h or f"col{i}").strip().replace("\n", " ") for i, h in enumerate(header)]
    cat_idx = next((i for i, h in enumerate(header) if category_field.lower() in h.lower()), None)
    records = []
    for row in rows:
        cells = [("" if c is None else str(c)).strip().replace("\n", " ") for c in row]
        if not any(cells) or cells == header:
            continue
        entity = cells[0] or next((c for c in cells if c), "")
        lines = [f"{h}: {v}" for h, v in zip(header, cells) if v]
        meta = dict(base_meta, kind="record", entity=entity,
                    category=cells[cat_idx] if cat_idx is not None and cat_idx < len(cells) else "")
        records.append(Document(page_content=f"RECORD: {entity}\n" + "\n".join(lines), metadata=meta))
    return records


def table_cells_text(page, tab) -> list[list[str]]:
    """Cell text rebuilt from the page's words inside each cell's bbox.
    (tab.extract() can drop narrow spaces; word boxes keep them intact.)"""
    words = page.get_text("words")
    grid = []
    for row in tab.rows:
        cells = []
        for bbox in row.cells:
            if bbox is None:
                cells.append("")
                continue
            x0, y0, x1, y1 = bbox
            inside = [w for w in words
                      if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
            inside.sort(key=lambda w: (w[5], w[6], w[7]))  # block, line, word order
            cells.append(" ".join(w[4] for w in inside))
        grid.append(cells)
    return grid


def extract_pdf_tables(pdf_path: Path, base_meta: dict, category_field: str) -> list[Document]:
    """Turn every table in the PDF into labelled per-row records."""
    import pymupdf

    records: list[Document] = []
    with pymupdf.open(Path(pdf_path)) as doc:
        for pno, page in enumerate(doc, start=1):
            if not hasattr(page, "find_tables"):
                break
            try:
                tables = page.find_tables().tables
            except Exception:
                continue
            for tab in tables:
                data = table_cells_text(page, tab)
                if len(data) < 2:
                    continue
                records += rows_to_records(data[0], data[1:], dict(base_meta, page=pno), category_field)
    return records


def chunk_pages(pages: list[dict], base_meta: dict, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split page text into overlapping chunks. Each chunk gets a stable
    `chunk_id` and a `position` so neighbours (prev/next) can be found later."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "। ", ". ", " ", ""])
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            chunks.append(Document(page_content=piece, metadata=dict(
                base_meta, page=p["page"], ocr=p["ocr"], kind="text",
                lang=detect_language(piece))))
    return chunks


def assign_ids(docs: list[Document], doc_id: str) -> list[Document]:
    """Give every document a sequential position and an id '<doc>:<n>'."""
    for pos, d in enumerate(docs):
        d.metadata["position"] = pos
        d.metadata["chunk_id"] = f"{doc_id[:10]}:{pos}"
    return docs


@traceable(name="ingest_pdf")
def ingest_pdf(pdf_path: Path, settings: Settings, llm=None, vision_llm=None,
               source_url: str = "") -> list[Document]:
    """Full PDF pipeline -> list of Documents ready for indexing:
    text chunks + table records + (for non-English docs) a KEY FACTS summary."""
    pdf_path = Path(pdf_path)
    raw = pdf_path.read_bytes()
    doc_id = content_hash(raw)
    base = {"doc_id": doc_id, "source": pdf_path.name, "source_url": source_url,
            "doc_urls": [source_url] if source_url else [], "date": date.today().isoformat(),
            "entity": "", "category": ""}
    pages = extract_pdf_pages(pdf_path, vision_llm, settings.min_scanned_chars)
    docs = chunk_pages(pages, base, settings.chunk_size, settings.chunk_overlap)
    docs += extract_pdf_tables(pdf_path, base, settings.category_field)
    full_text = "\n".join(p["text"] for p in pages)
    lang = detect_language(full_text)
    if lang != "en" and llm is not None and full_text.strip():
        facts = summarise_key_facts(full_text, llm)
        docs.append(Document(page_content=facts, metadata=dict(
            base, kind="key_facts", lang="en", page=0, entity=pdf_path.stem)))
    for d in docs:
        d.metadata.setdefault("lang", lang)
    return assign_ids(docs, doc_id)


# ---------------------------------------------------------------------------
# Website crawling (optional source; set SOURCE_WEBSITE in .env)
# ---------------------------------------------------------------------------

def load_robots(start_url: str, user_agent: str) -> RobotFileParser:
    """Fetch robots.txt with verified TLS; fall back to allow-all if missing."""
    import requests

    rp = RobotFileParser()
    robots_url = urljoin(start_url, "/robots.txt")
    try:
        r = requests.get(robots_url, timeout=15, verify=config.requests_verify(),
                         headers={"User-Agent": user_agent})
        rp.parse(r.text.splitlines() if r.ok else [])
    except requests.RequestException:
        rp.parse([])
    return rp


def fetch_html(url: str, user_agent: str) -> tuple[str, str]:
    """GET a page with requests (TLS verified). Returns (html, content_type)."""
    import requests

    r = requests.get(url, timeout=30, verify=config.requests_verify(),
                     headers={"User-Agent": user_agent})
    r.raise_for_status()
    return r.text, r.headers.get("content-type", "")


def needs_javascript(html: str) -> bool:
    """Heuristic: visible text is tiny or the page asks for JavaScript."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    text = soup.get_text(" ", strip=True)
    return len(text) < 200 or "enable javascript" in html.lower()


def fetch_with_playwright(url: str) -> str:
    """Render a JavaScript page with headless Chromium (only used as fallback)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=45000)
        html = page.content()
        browser.close()
    return html


def html_to_records(html: str, url: str, category_field: str) -> tuple[list[Document], list[str], list[str]]:
    """Parse a page into (records, page_links, pdf_links).
    Tables -> one labelled record per row; list items with links -> one record each."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    base = {"source": url, "source_url": url, "date": date.today().isoformat(), "doc_urls": []}
    records: list[Document] = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
        for tr in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = (cells + [""] * len(header))[:len(header)]  # align to header width
            links = [urljoin(url, a["href"]) for a in tr.find_all("a", href=True)]
            for rec in rows_to_records(header, [cells], base, category_field):
                rec.metadata["doc_urls"] = links  # linked documents kept as a field
                records.append(rec)
    heading = ""
    for el in soup.find_all(["h1", "h2", "h3", "li"]):
        if el.name != "li":
            heading = el.get_text(" ", strip=True)
            continue
        a = el.find("a", href=True)
        text = el.get_text(" ", strip=True)
        if a and len(text) > 15:
            records.append(Document(
                page_content=f"RECORD: {text}\nSection: {heading}",
                metadata=dict(base, kind="record", entity=text[:120], category=heading,
                              doc_urls=[urljoin(url, a["href"])])))
    links = [urldefrag(urljoin(url, a["href"]))[0] for a in soup.find_all("a", href=True)]
    pdfs = [l for l in links if l.lower().endswith(".pdf")]
    pages = [l for l in links if not l.lower().endswith(".pdf")]
    for t in soup(["script", "style", "noscript", "nav", "footer"]):
        t.decompose()
    text = soup.get_text("\n", strip=True)
    if text:
        records.insert(0, Document(page_content=text, metadata=dict(base, kind="page", entity="", category="")))
    return records, pages, pdfs


@traceable(name="crawl_site")
def crawl_site(start_url: str, settings: Settings,
               fetch: Callable[[str, str], tuple[str, str]] = fetch_html) -> tuple[list[Document], list[str]]:
    """Breadth-first crawl within the start domain. Respects robots.txt, waits
    `crawl_delay` between requests, stops at `crawl_max_pages`, never revisits.
    Returns (page/record documents, discovered PDF URLs)."""
    domain = urlparse(start_url).netloc
    robots = load_robots(start_url, settings.user_agent)
    queue, visited = deque([start_url]), set()
    docs: list[Document] = []
    pdf_urls: list[str] = []
    while queue and len(visited) < settings.crawl_max_pages:
        url = queue.popleft()
        if url in visited or urlparse(url).netloc != domain:
            continue
        if not robots.can_fetch(settings.user_agent, url):
            continue
        visited.add(url)
        try:
            html, ctype = fetch(url, settings.user_agent)
            if "html" in ctype and needs_javascript(html):
                html = fetch_with_playwright(url)
        except Exception:
            continue
        recs, links, pdfs = html_to_records(html, url, settings.category_field)
        docs += recs
        pdf_urls += [p for p in pdfs if p not in pdf_urls]
        queue.extend(l for l in links if l not in visited)
        time.sleep(settings.crawl_delay)
    return docs, pdf_urls


def chunk_web_docs(docs: list[Document], settings: Settings) -> list[Document]:
    """Split long page texts; keep records whole. Assign ids per source URL."""
    out: list[Document] = []
    by_source: dict[str, list[Document]] = {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
                                              chunk_overlap=settings.chunk_overlap)
    for d in docs:
        pieces = splitter.split_documents([d]) if d.metadata.get("kind") == "page" else [d]
        for p in pieces:
            p.metadata.setdefault("lang", detect_language(p.page_content))
            p.metadata["doc_id"] = content_hash(p.metadata["source_url"])
            by_source.setdefault(p.metadata["source_url"], []).append(p)
    for src, items in by_source.items():
        out += assign_ids(items, content_hash(src))
    return out


# ---------------------------------------------------------------------------
# CLI: python -m hybrid_rag.ingest path/to/file.pdf [--graph]
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    from hybrid_rag.retrieval import build_store, load_store, make_embeddings, make_llm, save_store

    ap = argparse.ArgumentParser(description="Ingest PDFs (and optionally a website) into the index")
    ap.add_argument("pdfs", nargs="*", type=Path)
    ap.add_argument("--crawl", action="store_true", help="also crawl SOURCE_WEBSITE")
    ap.add_argument("--graph", action="store_true", help="also build the Neo4j knowledge graph")
    args = ap.parse_args()
    s = config.get_settings()
    llm, vision = make_llm(s), make_llm(s, model=s.vision_model)
    docs: list[Document] = []
    for pdf in args.pdfs:
        docs += ingest_pdf(pdf, s, llm, vision)
    if args.crawl and s.source_website:
        web, _ = crawl_site(s.source_website, s)
        docs += chunk_web_docs(web, s)
    store = load_store(make_embeddings(s))
    store = build_store(docs, make_embeddings(s), existing=store)
    save_store(store)
    print(f"Indexed {len(docs)} chunks; store now has {len(store.chunks)}.")
    if args.graph and s.graph_enabled:
        from hybrid_rag.graph import build_graph, connect

        with connect(s) as driver:
            print(build_graph(driver, docs, llm, s.neo4j_database))


if __name__ == "__main__":  # pragma: no cover
    main()
