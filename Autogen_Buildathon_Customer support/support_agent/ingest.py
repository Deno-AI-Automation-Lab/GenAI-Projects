"""Ingestion: parse product .docx files, OCR embedded images, summarise, chunk and index.

Layout expected under DATA_DIR (one folder per product, folder name = product name):

    data/
      PureSweep V12 Robot Vacuum & Mop/
        shipping_policy.docx  refund_policy.docx  account_and_login.docx  product_faq.docx
      AeroBeat X9 Wireless Noise-Cancelling Headphones/ ...
      SmartCrisp Pro 8L Wi-Fi Air Fryer/ ...

Output under INDEX_DIR/<product-slug>/ :
    chunks.json          every chunk with prev/next links and metadata fields
    records.json         full section ("entity") text + LLM summary
    faiss_chunks/        FAISS index over chunks         (semantic retriever)
    faiss_summaries/     FAISS index over record summaries (entity-summary retriever)
    manifest.json        file hashes -> skip unchanged products on re-run
and INDEX_DIR/_router/ : FAISS over product profiles used to route a complaint.

Each product gets its OWN indexes so answers never mix products.

Note: the template's web-crawling step (robots.txt, BFS, Playwright) does not apply –
the sources here are local .docx files. Tables become one labelled record per row,
embedded images are OCR'd with a vision model and cached by content hash, and
non-English documents get an English KEY FACTS summary indexed alongside them.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from docx import Document as load_docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from .config import DOC_TYPES, Settings, get_settings, product_slug
from .language import detect_language, language_name
from .models import Chunk

log = logging.getLogger(__name__)

ROUTER_DIR_NAME = "_router"


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@dataclass
class ProductSource:
    name: str
    slug: str
    folder: Path
    files: list[Path] = field(default_factory=list)


def discover_products(data_dir: Path) -> list[ProductSource]:
    """Every sub-folder of DATA_DIR containing .docx files is one product."""
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")
    products: list[ProductSource] = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in folder.glob("*.docx") if not f.name.startswith("~$"))
        if files:
            products.append(ProductSource(folder.name, product_slug(folder.name), folder, files))
    return products


def doc_type_from_filename(path: Path) -> str:
    """Map 'refund_policy.docx' / 'Refund Policy - X9.docx' to a canonical doc type."""
    stem = path.stem.lower().replace("-", "_").replace(" ", "_")
    for dt in DOC_TYPES:
        if dt in stem:
            return dt
    for dt, hint in (("shipping_policy", "ship"), ("refund_policy", "refund"),
                     ("account_and_login", "login"), ("product_faq", "faq")):
        if hint in stem:
            return dt
    return "other"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# DOCX parsing
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    """A heading-delimited record (the 'entity' unit of this corpus)."""

    title: str
    paragraphs: list[str] = field(default_factory=list)
    table_rows: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


def iter_block_items(doc: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in true document order (python-docx splits them)."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _is_heading(p: Paragraph) -> bool:
    """Headings = Heading/Title styles, or short fully-bold lines, or FAQ questions."""
    style = (p.style.name if p.style is not None else "").lower()
    text = p.text.strip()
    if not text:
        return False
    if style.startswith(("heading", "title")):
        return True
    runs = [r for r in p.runs if r.text.strip()]
    if runs and len(text) < 120 and all(r.bold for r in runs):
        return True
    # FAQ documents: a short standalone question starts a new Q&A record
    core = re.sub(r"^(?:Q\s*\d*\s*[.:)]\s*|\d+\s*[.)]\s*)", "", text, flags=re.I)
    is_question = core.endswith("?") and len(core.split()) <= 25 and "." not in core.rstrip("?")
    return is_question


def table_to_records(table: Table, context_title: str) -> list[str]:
    """One labelled record per table row: 'Section | Col: value; Col: value'."""
    rows = [[c.text.strip() for c in r.cells] for r in table.rows]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body:  # single-row table: treat as plain text
        return [" | ".join(header)]
    records = []
    for r in body:
        pairs = [f"{h or f'col{i + 1}'}: {v}" for i, (h, v) in enumerate(zip(header, r)) if v]
        if pairs:
            records.append(f"[{context_title}] " + "; ".join(pairs))
    return records


def parse_docx_sections(path: Path) -> list[Section]:
    """Split a .docx into sections by headings; tables become labelled row records."""
    doc = load_docx(str(path))
    sections: list[Section] = [Section(title=path.stem.replace("_", " ").title())]
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            if _is_heading(block):
                sections.append(Section(title=text))
            else:
                sections[-1].paragraphs.append(text)
        else:
            sections[-1].table_rows.extend(table_to_records(block, sections[-1].title))
    return [s for s in sections if s.paragraphs or s.table_rows]


def extract_docx_images(path: Path) -> list[tuple[str, bytes, str]]:
    """Return (sha256, bytes, mime) for every image embedded in the .docx."""
    doc = load_docx(str(path))
    out = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            blob = rel.target_part.blob
            out.append((hashlib.sha256(blob).hexdigest(), blob, rel.target_part.content_type))
    return out


# --------------------------------------------------------------------------- #
# Hash-keyed LLM cache (OCR, summaries, KEY FACTS, graph extraction)
# --------------------------------------------------------------------------- #
def cached_json(namespace: str, key: str, producer: Callable[[], dict], settings: Settings) -> dict:
    """Return cached JSON for sha256(key) or call `producer` once and store it."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    path = settings.cache_dir / namespace / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


@traceable(name="ocr_image")
def ocr_image(image_sha: str, blob: bytes, mime: str, settings: Settings) -> str:
    """OCR an embedded image with the vision model (GPT-4o); cached by content hash."""
    from .llm import get_chat_llm

    def _run() -> dict:
        llm = get_chat_llm(settings.vision_model)
        b64 = base64.b64encode(blob).decode()
        msg = HumanMessage(content=[
            {"type": "text", "text": "Transcribe all text in this image exactly. "
                                     "If it contains no text, reply with EMPTY."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ])
        text = llm.invoke([msg]).content
        return {"text": "" if str(text).strip() == "EMPTY" else str(text)}

    return cached_json("ocr", image_sha, _run, settings)["text"]


@traceable(name="summarise_section")
def summarise_section(product_name: str, title: str, text: str, settings: Settings) -> str:
    """2-3 sentence English summary of a section, used by the entity-summary retriever."""
    from .llm import get_chat_llm

    def _run() -> dict:
        llm = get_chat_llm()
        out = llm.invoke([
            SystemMessage(content="Summarise this support-document section in 2-3 English "
                                  "sentences. Keep exact numbers, time limits and conditions. "
                                  "Do not add information."),
            HumanMessage(content=f"Product: {product_name}\nSection: {title}\n\n{text[:6000]}"),
        ]).content
        return {"summary": str(out).strip()}

    return cached_json("summary", f"{product_name}|{title}|{text}", _run, settings)["summary"]


@traceable(name="key_facts_summary")
def key_facts_summary(product_name: str, text: str, lang: str, settings: Settings) -> str:
    """English KEY FACTS for non-English documents so English queries can find them."""
    from .llm import get_chat_llm

    def _run() -> dict:
        llm = get_chat_llm()
        out = llm.invoke([
            SystemMessage(content="Write an English 'KEY FACTS' list for this "
                                  f"{language_name(lang)} support document: eligibility, "
                                  "conditions, amounts, fees, time limits, dates, contacts and "
                                  "procedures. Only facts stated in the text."),
            HumanMessage(content=text[:12000]),
        ]).content
        return {"facts": "KEY FACTS:\n" + str(out).strip()}

    return cached_json("keyfacts", f"{product_name}|{text}", _run, settings)["facts"]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def build_chunks(
    product: ProductSource,
    file_path: Path,
    sections: list[Section],
    settings: Settings,
    extra_texts: list[tuple[str, str, str]] | None = None,
) -> list[Chunk]:
    """Chunk each section; table rows stay one-record-per-chunk. Links prev/next in order.

    extra_texts: (title, text, kind) for OCR output and KEY FACTS summaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "। ", "。", " ", ""],
    )
    doc_type = doc_type_from_filename(file_path)
    base = f"{product.slug}:{file_path.stem}"
    chunks: list[Chunk] = []

    def add(record_id: str, title: str, text: str, kind: str) -> None:
        cid = f"{base}:{len(chunks):04d}"
        chunks.append(Chunk(
            chunk_id=cid, product=product.slug, product_name=product.name, doc_type=doc_type,
            source_file=file_path.name, section_title=title, record_id=record_id,
            text=text, kind=kind, language=detect_language(text),
        ))

    for idx, sec in enumerate(sections):
        record_id = f"{base}:sec{idx:03d}"
        if sec.text:
            for piece in splitter.split_text(sec.text):
                # Prefix the heading so each chunk is self-describing for BM25/embeddings
                add(record_id, sec.title, f"{sec.title}\n{piece}", "text")
        for row in sec.table_rows:
            add(record_id, sec.title, row, "table_row")
    for j, (title, text, kind) in enumerate(extra_texts or []):
        for piece in splitter.split_text(text):
            add(f"{base}:{kind}{j:03d}", title, piece, kind)

    # Neighbour links let context building pull previous/next chunks
    for i, c in enumerate(chunks):
        c.prev_id = chunks[i - 1].chunk_id if i > 0 else None
        c.next_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
    return chunks


# --------------------------------------------------------------------------- #
# Index building
# --------------------------------------------------------------------------- #
def product_index_dir(settings: Settings, slug: str) -> Path:
    return settings.index_dir / slug


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def compute_manifest(product: ProductSource) -> dict[str, str]:
    return {f.name: file_sha256(f) for f in product.files}


@traceable(name="ingest_product")
def ingest_product(product: ProductSource, settings: Settings, embeddings=None, force: bool = False,
                   build_graph: bool = True) -> dict:
    """Parse, enrich, chunk and index one product into its own FAISS stores."""
    from langchain_community.vectorstores import FAISS

    from .llm import get_embeddings

    out_dir = product_index_dir(settings, product.slug)
    manifest = compute_manifest(product)
    manifest_path = out_dir / "manifest.json"
    if not force and manifest_path.exists() and json.loads(manifest_path.read_text()) == manifest:
        log.info("Skipping %s (unchanged)", product.name)
        return {"product": product.slug, "skipped": True}

    embeddings = embeddings or get_embeddings()
    all_chunks: list[Chunk] = []
    records: dict[str, dict] = {}
    for f in product.files:
        sections = parse_docx_sections(f)
        extras: list[tuple[str, str, str]] = []
        if settings.ocr_enabled:
            for sha, blob, mime in extract_docx_images(f):
                try:
                    text = ocr_image(sha, blob, mime, settings)
                    if text.strip():
                        extras.append((f"Image text ({f.stem})", text, "ocr"))
                except Exception as exc:  # OCR must never block ingestion
                    log.warning("OCR failed for image in %s: %s", f.name, exc)
        full_text = "\n".join(s.text + "\n" + "\n".join(s.table_rows) for s in sections)
        lang = detect_language(full_text)
        if lang != "en" and full_text.strip():
            extras.append(("KEY FACTS (English)", key_facts_summary(product.name, full_text, lang, settings), "key_facts"))
        chunks = build_chunks(product, f, sections, settings, extras)
        all_chunks.extend(chunks)
        for c in chunks:
            rec = records.setdefault(c.record_id, {
                "record_id": c.record_id, "title": c.section_title, "source_file": c.source_file,
                "doc_type": c.doc_type, "chunk_ids": [], "text": "",
            })
            rec["chunk_ids"].append(c.chunk_id)
            rec["text"] += ("\n" if rec["text"] else "") + c.text

    if not all_chunks:
        raise ValueError(f"No text extracted for {product.name}")

    for rec in records.values():
        try:
            rec["summary"] = summarise_section(product.name, rec["title"], rec["text"], settings)
        except Exception as exc:
            log.warning("Summary failed (%s); using heading+lead text", exc)
            rec["summary"] = f"{rec['title']}: {rec['text'][:300]}"

    chunk_docs = [LCDocument(page_content=c.text, metadata=c.model_dump()) for c in all_chunks]
    FAISS.from_documents(chunk_docs, embeddings).save_local(str(out_dir / "faiss_chunks"))
    summary_docs = [
        LCDocument(page_content=f"{r['title']}\n{r['summary']}",
                   metadata={"record_id": r["record_id"], "source_file": r["source_file"],
                             "doc_type": r["doc_type"], "title": r["title"]})
        for r in records.values()
    ]
    FAISS.from_documents(summary_docs, embeddings).save_local(str(out_dir / "faiss_summaries"))
    _write_json(out_dir / "chunks.json", [c.model_dump() for c in all_chunks])
    _write_json(out_dir / "records.json", list(records.values()))
    _write_json(out_dir / "product.json", {"name": product.name, "slug": product.slug,
                                           "files": [f.name for f in product.files]})

    graph_status = "disabled"
    if build_graph and settings.graph_enabled:
        try:
            from .graph import build_product_graph, get_driver

            with get_driver(settings) as driver:
                build_product_graph(driver, product.name, product.slug, all_chunks, records, settings)
            graph_status = "ok"
        except Exception as exc:  # graph is optional: retrieval falls back to hybrid
            log.error("Graph build failed for %s: %s", product.name, exc)
            graph_status = f"failed: {exc}"

    _write_json(manifest_path, manifest)  # written last => partial runs are retried
    return {"product": product.slug, "chunks": len(all_chunks), "records": len(records), "graph": graph_status}


def build_router_index(products: list[ProductSource], settings: Settings, embeddings=None) -> None:
    """FAISS index of product profiles (name + FAQ summaries) for complaint routing."""
    from langchain_community.vectorstores import FAISS

    from .llm import get_embeddings

    embeddings = embeddings or get_embeddings()
    docs = []
    for p in products:
        rec_path = product_index_dir(settings, p.slug) / "records.json"
        recs = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else []
        faq = [r for r in recs if r["doc_type"] == "product_faq"] or recs
        # several small profile docs per product -> better recall for short complaints
        docs.append(LCDocument(page_content=p.name, metadata={"product": p.slug, "name": p.name}))
        for r in faq[:40]:
            docs.append(LCDocument(page_content=f"{p.name}: {r['title']}. {r.get('summary', '')}",
                                   metadata={"product": p.slug, "name": p.name}))
    FAISS.from_documents(docs, embeddings).save_local(str(settings.index_dir / ROUTER_DIR_NAME))
    _write_json(settings.index_dir / ROUTER_DIR_NAME / "products.json",
                [{"name": p.name, "slug": p.slug} for p in products])


def run_ingestion(force: bool = False, build_graph: bool = True, settings: Settings | None = None) -> list[dict]:
    """Ingest every product folder and rebuild the router. Safe to re-run."""
    s = settings or get_settings()
    products = discover_products(s.data_dir)
    if not products:
        raise FileNotFoundError(f"No product folders with .docx files in {s.data_dir}")
    results = []
    for p in products:
        try:
            results.append(ingest_product(p, s, force=force, build_graph=build_graph))
        except Exception as exc:
            log.exception("Ingestion failed for %s", p.name)
            results.append({"product": p.slug, "error": str(exc)})
    build_router_index(products, s)
    return results


def main() -> None:
    from .llm import bootstrap_runtime

    bootstrap_runtime()
    ap = argparse.ArgumentParser(description="Index product .docx documents")
    ap.add_argument("--force", action="store_true", help="re-index even if files are unchanged")
    ap.add_argument("--no-graph", action="store_true", help="skip Neo4j graph build")
    args = ap.parse_args()
    for r in run_ingestion(force=args.force, build_graph=not args.no_graph):
        print(json.dumps(r))


if __name__ == "__main__":
    main()
