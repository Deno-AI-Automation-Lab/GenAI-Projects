"""Tamil Nadu Agriculture Scheme Assistant - LangChain + FAISS + OpenAI + Streamlit (single file).

What it does
  1. Fetches the scheme list page and its scheme-detail sub-links (WebBaseLoader first, Playwright
     for JavaScript-built pages), plus linked PDFs and images. Images and scanned PDF pages are
     turned into text with an OpenAI vision model. robots.txt is checked before every fetch.
  2. Adds anything you saved yourself into the  manual_sources/  folder (PDF, HTML, images, TXT).
  3. Writes everything into ONE PDF (tn_agri_schemes.pdf), then chunks + embeds that PDF into
     ONE local FAISS index. The number of chunks is reported when building finishes.
  4. Answers questions ONLY from that PDF (no web search) using hybrid RAG
     (FAISS semantic search + BM25 keyword search), with chat memory and streaming.
  5. Re-fetches automatically when the data is older than 7 days (and can be scheduled weekly).

Setup
  pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters ^
              faiss-cpu pypdf pymupdf rank_bm25 beautifulsoup4 lxml reportlab requests python-dotenv playwright
  python -m playwright install chromium

.env (next to this file)
  OPENAI_API_KEY=sk-...
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=tn-agri-scheme-rag

Run
  streamlit run tamilnaduagrischemes_app.py         # the app
  python tamilnaduagrischemes_app.py --refresh      # fetch + rebuild only (use this for the weekly schedule)
"""
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse
from typing import Literal
from xml.sax.saxutils import escape

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
load_dotenv()
os.environ.setdefault("USER_AGENT", "TN-Agri-Scheme-RAG/1.0 (personal research; contact: you@example.com)")
warnings.filterwarnings("ignore", category=DeprecationWarning)   # hide the harmless langchain-community notice

import pymupdf  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

# ============================== settings ==============================
START_URL = "https://www.tnagrisnet.tn.gov.in/people_app/GoScheme"
LINK_KEYWORDS = ("goscheme", "scheme")   # follow only same-site links containing one of these words (+ PDFs)
MAX_DEPTH = 2                            # 0 = start page only, 1 = + its links, 2 = one level deeper
MAX_PAGES = 150                          # safety limit per refresh
REQUEST_DELAY = 1.0                      # seconds between requests, to be polite to the server
PAGE_TIMEOUT = 30                        # seconds to wait for one page before giving up on it
MAX_REFRESH_MINUTES = 90                 # stop an in-app refresh that runs longer than this
MAX_DOWNLOAD_MB = 50                     # skip PDFs/images bigger than this
LLM_TIMEOUT = 60                         # seconds before a chat call is retried / abandoned
VISION_TIMEOUT = 180                     # dense scanned Tamil pages need longer
USE_VISION = True                        # convert images / scanned PDF pages to text with the vision model
MAX_IMAGES_PER_PAGE = 5
MAX_OCR_PAGES_PER_PDF = 150              # high enough to read whole scheme booklets
OCR_WORKERS = 4                          # scanned pages read in parallel (lower it if you hit OpenAI rate limits)
OCR_CACHE_DIR = APP_DIR / "ocr_cache"    # vision results are saved here, so weekly refreshes don't pay twice
SKIP_IMAGE_WORDS = ("logo", "icon", "banner", "emblem", "flag", "arrow", "sprite", "loader", "social")
JUNK_WORDS = {"tester", "test", "testing", "dummy"}   # table rows containing ONLY these words are skipped
IGNORE_WORDS = {"view", "not", "uploaded"}            # link/placeholder text ignored when judging a row


def clean_repeats(text):
    """Collapse runaway repetition from OCR, e.g. the same word 200 times in a row -> once."""
    text = re.sub(r"(\S+)(?:\s+\1){3,}", r"\1", text)
    return re.sub(r"((?:\S+\s+){1,6}?\S+)(?:\s+\1){3,}", r"\1", text)   # repeated short phrases


def bm25_tokens(text):
    """Tokenizer for keyword search: lowercase words, numbers and Tamil, punctuation removed,
    so 'Rotavator', 'rotavator' and '(Rotavator)' all match."""
    return re.findall(r"[a-z0-9\u0B80-\u0BFF]+(?:\.[0-9]+)?", text.lower())


def norm(text):
    """Normalise whitespace so the same scheme name matches in the table and in the popup link."""
    return " ".join(str(text).split())


def is_junk_row(text):
    """True for placeholder rows like '29 | | tester | -tester | tester | tester | View'.
    Numbers, quotes and punctuation are ignored; a row is junk only if every real word is a junk word."""
    words = {w.lower() for w in re.findall(r"[A-Za-z஀-௿]+", text)} - IGNORE_WORDS
    return bool(words) and words <= JUNK_WORDS

MANUAL_DIR = APP_DIR / "manual_sources"  # drop your own saved PDFs / HTML / screenshots here
PDF_PATH = APP_DIR / "tn_agri_schemes.pdf"
INDEX_DIR = APP_DIR / "faiss_index"
META_PATH = APP_DIR / "index_meta.json"
REFRESH_DAYS = 7
INDEX_VERSION = "4"                      # bump when chunking/cleaning changes, to force one rebuild

CHAT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o"                   # reads Tamil scans far better than gpt-4o-mini
OCR_DPI = 220                              # sharper page images for small Tamil print
OCR_VERSION = "v2"                         # change to force every page to be re-read
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 200
TOP_K = 12                                # passages given to the model per question
MEMORY_TURNS = 4                          # how many previous Q&A pairs the bot remembers

SYSTEM_PROMPT = """You are an assistant for Tamil Nadu Government agriculture schemes.
Rules:
- Answer ONLY from the numbered context passages taken from the scheme PDF. Never use outside
  knowledge or the internet, and never guess amounts, dates or eligibility.
- If the answer is not in the context, reply exactly: "I could not find this in the scheme documents."
- Cite the passages you used like [1] or [2][3].
- Each scheme record has labelled fields:
    "Scheme Description"  = what the scheme is / its purpose (official text from the scheme list)
    "Eligibility Criteria", "Document's Required", "GO & Guidelines"
    "Scheme Details View" = components and pattern of subsidy (amounts)
  Answer the field that was asked:
  * Description / "what is <scheme>": reproduce the COMPLETE "Scheme Description" field - every
    sentence and paragraph, in order, nothing shortened, merged or moved into other bullets (you may
    show each sentence as its own bullet). Then add one line offering the subsidy components.
  * Eligibility / documents: give the COMPLETE field (every item of a numbered list).
  * Subsidy / components / amounts / "all details": give the Scheme Details View rows in full.
- GO & Guidelines passages are government orders (translated from Tamil). Each starts with KEY FACTS
  in English, followed by the full transcription. Use them for GO numbers, dates, budgets, component
  rates and detailed rules. If a passage contains [unclear], say that part could not be read.
- If the context starts with a [Scheme index], it is the exact, complete set for the question: include
  every scheme in it (state the count, e.g. "There are 10 Horticulture schemes"), in index order, and
  add no other schemes. For other questions across schemes, check EVERY scheme record in the context.
- Reply in the same language as the question (Tamil or English).
Format (crisp and clear):
- Start with one short sentence that answers directly. No preamble, no repeating the question.
- Then short bullet points with bold labels, e.g. "- **Eligibility:** All district farmers [2]".
  One fact per bullet. Use sub-bullets only for lists such as documents.
- For comparisons or lists of schemes, use a compact markdown table (Scheme | the asked columns).
- Amount / budget / outlay / sanction / release questions: report EVERY relevant figure in the context,
  each with what it is - e.g. approved outlay, Government of India share, State share, amount released /
  sanctioned (installment), administrative sanction, and any category-wise (General/SC/ST) or
  component-wise breakdown (show breakdowns as a table). Name the G.O. (number, date, year) each figure
  comes from. Never stop at the first amount you find.
- Subsidy components / Scheme Details View: ALWAYS show them as a markdown table on its own lines
  (never inside a bullet), with columns | Component | Pattern of subsidy |, one row per component, in
  the order given. Copy each row's value exactly from that same row; if it says "subsidy pattern not
  given on the website", write "Not specified" - never borrow a value from another row.
- Copy amounts, percentages, limits and dates exactly as written (e.g. Rs.15,000/-, 50%).
- Keep your own wording short (about 150 words). This limit never applies to official text you were
  asked for - a complete description or document list is always given in full. No closing remarks."""

VISION_PROMPT = """You are transcribing a page of a Tamil Nadu Government order (G.O.) or scheme guideline.
1. TAMIL TEXT: transcribe the page exactly as printed, line by line. Copy every number, G.O. number,
   date, amount (Rs./ரூ.), percentage and table row exactly. Write tables as 'cell | cell'.
   If a word is unreadable write [unclear] - never guess or invent words.
2. ENGLISH TRANSLATION: a faithful, literal translation of what you transcribed. Keep official names
   (e.g. "Kalaignarin All Village Integrated Agriculture Development Programme"), G.O. numbers, dates
   and amounts exactly. Translate [unclear] as [unclear]. Do not add, summarise or interpret.
If the page is already in English, give it once under ENGLISH TEXT. Ignore stamps, signatures and logos.
If the image has no meaningful text, reply with: NO_TEXT"""


# ============================== fetching ==============================
_robots = {}


def allowed_by_robots(url):
    """Respect the site's robots.txt."""
    root = "{0.scheme}://{0.netloc}".format(urlparse(url))
    if root not in _robots:
        rp = robotparser.RobotFileParser()
        try:   # fetched with a timeout so a slow server cannot hang the refresh
            r = requests.get(root + "/robots.txt", headers={"User-Agent": os.environ["USER_AGENT"]}, timeout=15)
            if r.status_code in (401, 403):
                rp.disallow_all = True
            elif r.status_code >= 400:
                rp = None                  # no robots.txt -> allowed
            else:
                rp.parse(r.text.splitlines())
        except Exception:
            rp = None                      # robots.txt unreachable -> treat as allowed
        _robots[root] = rp
    rp = _robots[root]
    return True if rp is None else rp.can_fetch(os.environ["USER_AGENT"], url)


def download(url):
    """Download with a total time limit and a size limit, so one big/slow file cannot stall the refresh."""
    start, data = time.time(), bytearray()
    with requests.get(url, headers={"User-Agent": os.environ["USER_AGENT"]}, timeout=PAGE_TIMEOUT, stream=True) as r:
        r.raise_for_status()
        for block in r.iter_content(64 * 1024):
            data += block
            if len(data) > MAX_DOWNLOAD_MB * 1024 * 1024:
                raise RuntimeError(f"file larger than {MAX_DOWNLOAD_MB} MB - skipped")
            if time.time() - start > 120:
                raise RuntimeError("download took longer than 2 minutes - skipped")
        return bytes(data), r.headers.get("content-type", "")


def fetch_static(url):
    """Plain HTTP fetch with LangChain's WebBaseLoader (fast; no JavaScript)."""
    try:
        return WebBaseLoader(url, requests_kwargs={"timeout": PAGE_TIMEOUT}).scrape()
    except Exception:
        return None


POPUP_SELECTOR = ".modal:visible, [role='dialog']:visible, .ui-dialog:visible, .popup:visible"
CLOSE_SELECTOR = (".close, [data-dismiss='modal'], [data-bs-dismiss='modal'], .ui-dialog-titlebar-close, "
                  "button:has-text('×'), button:has-text('Close')")


NEXT_SELECTOR = ".paginate_button.next:not(.disabled), li.next:not(.disabled) > a"   # DataTables / Bootstrap "Next"


def show_all_rows(page, log):
    """Make a paginated table show every row: DataTables API first, then the 'Show N entries' dropdown."""
    page.wait_for_timeout(1_500)                  # let the table finish its first draw(s)
    how = page.evaluate("""() => {
        try {
            if (window.jQuery && jQuery.fn.dataTable) {
                const tables = jQuery.fn.dataTable.tables();
                tables.forEach(t => jQuery(t).DataTable().page.len(-1).draw());
                if (tables.length) return 'DataTables API';
            }
        } catch (e) {}
        return null;
    }""")
    if not how:
        try:
            sel = page.locator("select[name$='_length'], .dataTables_length select, select:has(option[value='100'])").first
            best = sel.evaluate("s => { const v = [...s.options].map(o => o.value);"
                                "return v.includes('-1') ? '-1' : v.sort((a, b) => b - a)[0]; }")
            sel.select_option(best, timeout=3_000)
            how = f"'Show {best}' dropdown"
        except Exception:
            pass
    page.wait_for_timeout(1_500)                  # let the table redraw before we read or click anything
    log(f"    table rows on page: {page.locator('table tbody tr').count()}" + (f" (via {how})" if how else ""))


MISSING = {"", "-", "null", "none", "nil", "na", "n/a"}


def popup_to_text(html, fallback_text):
    """Turn the popup's table into one clear line per component, read from the real table cells
    (so an empty cell stays empty and can't shift a value onto the wrong row):
        - Oil palm Maintenance: 50% subsidy limited to a maximum of Rs.5,250/ Ha
        - Aluminium ladders: subsidy pattern not given on the website"""
    table = BeautifulSoup(html, "lxml").find("table")
    if table is None:                                  # no table: keep plain text, minus title bar / close
        lines = [ln.replace("\t", " | ").strip() for ln in fallback_text.splitlines()]
        return "\n".join(ln for ln in lines if ln.replace("×", "").replace("Close", "").strip(" |")
                         not in ("", "Scheme Details View"))
    rows = [[norm(td.get_text(" ")) for td in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    header = [h.lower() for h in rows[0]]
    has_header = any("subsidy" in h or "input" in h or "component" in h or "componenet" in h for h in header)
    data = rows[1:] if has_header else rows
    sub_col = next((i for i, h in enumerate(header) if "subsidy" in h or "pattern" in h), None) if has_header else None
    lines = []
    for r in data:
        if sub_col is None:
            sub_col_r = len(r) - 1
        else:
            sub_col_r = sub_col
        value = r[sub_col_r] if sub_col_r < len(r) else ""
        names = [c for i, c in enumerate(r) if i != sub_col_r and c.lower() not in MISSING]
        label = names[0] if names else "Component"
        for extra in names[1:]:                        # add the component name only if it says something new
            if extra.lower() not in label.lower():
                label += f" ({extra})"
        lines.append(f"- {label}: {value if value.lower() not in MISSING else 'subsidy pattern not given on the website'}")
    return "\n".join(lines)


def close_popup(page):
    """Close an open popup (its close button, or Escape) and any tab a click may have opened."""
    popup = page.locator(POPUP_SELECTOR).first
    if popup.count():
        try:
            popup.locator(CLOSE_SELECTOR).first.click(timeout=2_000)
        except Exception:
            page.keyboard.press("Escape")
        try:
            popup.wait_for(state="hidden", timeout=3_000)
        except Exception:
            pass
    for extra in page.context.pages[1:]:
        extra.close()


def read_popups(page, url, log, start_no=1):
    """Click every in-page scheme link that opens a popup (no real URL), read the popup, close it.
    This captures the 'Scheme Details View' boxes (components, subsidy pattern, etc.)."""
    sources, previous, junk = {}, "", 0               # scheme name -> popup text
    links = page.locator("table tbody a")
    for n in range(links.count()):
        link = links.nth(n)
        try:
            href = (link.get_attribute("href") or "").strip()
            onclick = link.get_attribute("onclick")
            name = " ".join(link.inner_text().split())
            row_text = link.locator("xpath=ancestor::tr[1]").inner_text(timeout=2_000)
        except Exception:
            continue
        if not (href in ("", "#") or href.startswith("javascript") or onclick):
            continue                                   # real links (e.g. GO PDFs) are followed by the crawler
        if is_junk_row(name) or is_junk_row(row_text):
            junk += 1
            continue                                   # 'tester' placeholder row - nothing to read
        for attempt in (1, 2):                         # retry once: tables can redraw, popups can be slow
            try:
                link = links.nth(n)                    # re-find the element in case the table redrew
                link.scroll_into_view_if_needed(timeout=5_000)
                link.click(timeout=5_000)
                popup = page.locator(POPUP_SELECTOR).first
                popup.wait_for(state="visible", timeout=10_000)
                text = ""
                for _ in range(40):                    # up to 20 s for THIS scheme's data to load
                    text = popup.inner_text().strip()
                    if text and text != previous and "loading" not in text.lower():
                        break
                    page.wait_for_timeout(500)
                previous = text
                sources[norm(name)] = popup_to_text(popup.inner_html(), text)
                log(f"    popup {start_no + len(sources) - 1}: {name[:70]}")
                break
            except Exception as e:
                if attempt == 2:
                    log(f"    popup not read for '{name[:50]}': {str(e).splitlines()[0]}")
                else:
                    page.wait_for_timeout(1_500)
            finally:
                close_popup(page)
    if junk:
        log(f"    skipped {junk} test/dummy row(s)")
    return sources


def fetch_rendered(browser, url, log, popups=False):
    """Real browser fetch with Playwright, for pages whose content is loaded by JavaScript/AJAX.
    Returns (soup, {scheme name: popup text})."""
    page = browser.new_page(user_agent=os.environ["USER_AGENT"])
    page.set_default_timeout(PAGE_TIMEOUT * 1000)     # no single browser step can wait longer than this
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT * 1000)
        try:   # give AJAX content a short chance to arrive; busy sites may never become fully idle
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
        show_all_rows(page, log)

        # read the table page by page (only one page if 'show all' worked)
        htmls, popup_texts = [], {}
        for table_page in range(1, 51):
            html = page.inner_html("body")
            if htmls and html == htmls[-1]:
                break                                   # "Next" did nothing - last page reached
            htmls.append(html)
            if popups:
                popup_texts.update(read_popups(page, url, log, start_no=len(popup_texts) + 1))
            nxt = page.locator(NEXT_SELECTOR).first
            if not nxt.count() or not nxt.is_visible():
                break
            log(f"    table page {table_page + 1} ...")
            nxt.click()
            page.wait_for_timeout(1_500)
        title = escape(page.title() or url)
        soup = BeautifulSoup(f"<html><head><title>{title}</title></head><body>{''.join(htmls)}</body></html>", "lxml")
        return soup, popup_texts
    finally:
        page.close()


def page_text(soup):
    """Readable text of a page; tables become 'cell | cell' lines so amounts stay next to their labels."""
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    for table in soup.find_all("table"):
        rows = [" | ".join(c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"]))
                for tr in table.find_all("tr")]
        rows = [r for r in rows if r.strip() and not is_junk_row(r)]      # drop 'tester' placeholder rows
        table.replace_with("\n" + "\n".join(rows) + "\n")
    main = soup.find("main") or soup.find(id="content") or soup.body or soup
    lines = (" ".join(line.split()) for line in main.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def extract_scheme_records(soup, url):
    """Turn the scheme list table into one labelled record per scheme, e.g.
         Scheme: ...  /  Department: ...  /  Scheme Description: ...  /  Eligibility Criteria: ...
    so retrieval and the LLM can tell the description apart from eligibility, documents and subsidy."""
    records, seen = [], set()
    for table in soup.find_all("table"):
        header_row = table.find("tr")
        headers = [norm(th.get_text(" ")) for th in header_row.find_all(["th", "td"])] if header_row else []
        if not any(h.lower() == "scheme" for h in headers):
            continue                                   # not the scheme list table
        scheme_col = [h.lower() for h in headers].index("scheme")
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) != len(headers) or is_junk_row(tr.get_text(" ")):
                continue
            name = norm(cells[scheme_col].get_text(" "))
            if not name or name in seen:
                continue
            seen.add(name)
            fields, go_links = [], []
            for header, cell in zip(headers, cells):
                if header.lower() in ("s.no", "sno", "scheme"):
                    continue
                value = "\n".join(norm(t) for t in cell.get_text("\n").splitlines() if t.strip())
                for a in cell.find_all("a", href=True):
                    link = urljoin(url, a["href"])
                    if link.lower().endswith(".pdf"):
                        go_links.append(link)
                        value += f" (PDF: {link})"
                fields.append((header, value or "-"))
            records.append({"name": name, "fields": fields, "go_links": go_links})
    return records


def image_to_text(img_bytes, mime="image/png"):
    """Multimodal step: let the vision model read text (incl. Tamil) from an image.
    Results are cached by image content, so an unchanged image is never sent twice."""
    OCR_CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.sha256(img_bytes + f"{VISION_MODEL}|{OCR_VERSION}".encode()).hexdigest()
    cache_file = OCR_CACHE_DIR / (key + ".txt")    # new model/prompt version -> page is read again
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    b64 = base64.b64encode(img_bytes).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
    ])
    llm = ChatOpenAI(model=VISION_MODEL, temperature=0, timeout=VISION_TIMEOUT, max_retries=2)
    text = llm.invoke([msg], config={"run_name": "image_to_text", "tags": ["ingest"]}).content.strip()
    text = "" if text.startswith("NO_TEXT") else clean_repeats(text)
    cache_file.write_text(text, encoding="utf-8")
    return text


GO_SUMMARY_PROMPT = """Below is the transcription (Tamil + English translation) of a Tamil Nadu Government
order / scheme guideline titled "{title}". Write an English KEY FACTS list using ONLY this text:
- G.O. number and date; issuing department
- Scheme name and year; purpose / objective
- Total budget / sanction amount and funding share (State / Central)
- Components with subsidy pattern, unit cost, rates and limits (one bullet each)
- Eligibility / beneficiaries / priority groups; districts or area covered
- Implementing agencies, how to apply, conditions, timelines
Copy numbers exactly. Skip any item that is not in the text - never guess.

TRANSCRIPTION:
{text}"""


def summarize_go(text, title, log):
    """English key facts for a GO (cached). Makes GO details easy to find with English questions."""
    OCR_CACHE_DIR.mkdir(exist_ok=True)
    cache_file = OCR_CACHE_DIR / ("summary_" + hashlib.sha256((title + text).encode()).hexdigest() + ".txt")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    log("    writing English key-facts summary ...")
    try:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, timeout=VISION_TIMEOUT, max_retries=2)
        summary = llm.invoke(GO_SUMMARY_PROMPT.format(title=title, text=text[:120_000]),
                             config={"run_name": "go_summary", "tags": ["ingest"]}).content.strip()
    except Exception as e:
        log(f"    summary failed: {e}")
        return ""
    cache_file.write_text(summary, encoding="utf-8")
    return summary


def pdf_to_text(data, log):
    """Text of a PDF; scanned pages (no text layer) go through the vision model."""
    texts, scanned = {}, {}                       # page number -> text / page image
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        total = doc.page_count
        log(f"    PDF has {total} page(s)")
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) >= 50 or not USE_VISION:
                texts[i] = text
            elif len(scanned) < MAX_OCR_PAGES_PER_PDF:
                scanned[i] = page.get_pixmap(dpi=OCR_DPI).tobytes("png")   # render now; read in parallel below

    if scanned:
        log(f"    {len(scanned)} scanned page(s) -> vision model ({OCR_WORKERS} at a time) ...")

        def read_page(i):
            for attempt in (1, 2):                   # one extra try for slow/timed-out pages
                try:
                    return i, image_to_text(scanned[i])
                except Exception as e:
                    if attempt == 2:
                        log(f"    vision failed on page {i}: {e}")
            return i, ""

        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
            for done, future in enumerate(as_completed([pool.submit(read_page, i) for i in scanned]), start=1):
                i, text = future.result()
                texts[i] = text
                log(f"    page {i} done ({done}/{len(scanned)})")

    return "\n\n".join(f"[PDF page {i}]\n{texts[i]}" for i in sorted(texts) if texts[i])


def crawl_website(log):
    """Breadth-first crawl of START_URL and its scheme sub-links. Returns a list of {title, url, text}."""
    sources, seen, queue = [], set(), [(START_URL, 0)]
    domain = urlparse(START_URL).netloc
    go_titles = {}                                     # GO PDF url -> scheme name(s), for clear titles
    pw = browser = None
    try:
        while queue and len(seen) < MAX_PAGES:
            url, depth = queue.pop(0)
            url = url.split("#")[0]
            if url in seen:
                continue
            seen.add(url)
            if not allowed_by_robots(url):
                log(f"  SKIPPED (robots.txt disallows): {url}")
                continue
            log(f"  fetching ({len(seen)}/{MAX_PAGES}): {url}")
            try:
                if url.lower().endswith(".pdf"):
                    log("    downloading PDF ...")
                    data, _ = download(url)
                    text = pdf_to_text(data, log)
                    if text:
                        title = (f"GO & Guidelines - {go_titles[url]}" if url in go_titles
                                 else url.rsplit("/", 1)[-1])
                        summary = summarize_go(text, title, log)
                        if summary:
                            text = (f"KEY FACTS (English summary of this document):\n{summary}\n\n"
                                    f"FULL TEXT (transcription and translation):\n{text}")
                        sources.append({"title": title, "url": url, "text": text})
                        log(f"  pdf : {url}")
                    continue

                # static first; Playwright for the start page (AJAX list) or when static text is thin
                soup = None
                if depth > 0:
                    log("    static fetch (WebBaseLoader) ...")
                    soup = fetch_static(url)
                if soup is None or len(soup.get_text(" ", strip=True)) < 300:
                    log("    browser fetch (Playwright) ...")
                    if browser is None:
                        from playwright.sync_api import sync_playwright
                        pw = sync_playwright().start()
                        browser = pw.chromium.launch()
                    soup, popup_texts = fetch_rendered(browser, url, log, popups=(depth == 0))
                else:
                    popup_texts = {}

                title = soup.title.get_text(strip=True) if soup.title else url

                # scheme list table -> one labelled record per scheme (table fields + its popup)
                records = extract_scheme_records(soup, url)
                for rec in records:
                    body = [f"Scheme: {rec['name']}"] + [f"{h}: {v}" for h, v in rec["fields"]]
                    popup = popup_texts.pop(rec["name"], "")
                    if popup:
                        body.append("Scheme Details View (components and pattern of subsidy):\n" + popup)
                    sources.append({"title": f"Scheme - {rec['name']}", "url": url, "text": "\n".join(body)})
                    for link in rec["go_links"]:
                        go_titles[link] = f"{go_titles[link]}; {rec['name']}" if link in go_titles else rec["name"]
                for name, popup in popup_texts.items():     # popups whose name didn't match a table row
                    sources.append({"title": f"Scheme - {name}", "url": url,
                                    "text": f"Scheme: {name}\nScheme Details View (components and pattern of subsidy):\n{popup}"})
                if records:
                    log(f"    {len(records)} scheme record(s) built from the table")

                # collect links and images BEFORE page_text() strips the page
                if depth < MAX_DEPTH:
                    for a in soup.find_all("a", href=True):
                        row = a.find_parent("tr")
                        if row and is_junk_row(row.get_text(" ")):
                            continue                         # don't follow 'View' links of tester rows
                        link = urljoin(url, a["href"]).split("#")[0]
                        if urlparse(link).netloc == domain and link not in seen and (
                                link.lower().endswith(".pdf") or any(k in link.lower() for k in LINK_KEYWORDS)):
                            queue.append((link, depth + 1))
                images = []
                if USE_VISION:
                    for img in soup.find_all("img", src=True):
                        src = urljoin(url, img["src"])
                        if not any(w in src.lower() for w in SKIP_IMAGE_WORDS) and src not in images:
                            images.append(src)

                text = page_text(soup)
                todo = images[:MAX_IMAGES_PER_PAGE]
                for n, src in enumerate(todo, start=1):
                    if not allowed_by_robots(src):
                        continue
                    try:
                        data, ctype = download(src)
                        if len(data) < 5_000:        # tiny images are icons/spacers
                            continue
                        log(f"    image {n}/{len(todo)} -> vision model ...")
                        img_text = image_to_text(data, ctype.split(";")[0] or "image/png")
                    except Exception as e:
                        log(f"    image skipped ({e})")
                        continue
                    if img_text:
                        text += f"\n\n[Text from image {src.rsplit('/', 1)[-1]}]\n{img_text}"

                if records:
                    log(f"  page: {url}")            # the table is already stored as scheme records
                elif text.strip():
                    sources.append({"title": title, "url": url, "text": text})
                    log(f"  page: {url}")
            except Exception as e:
                log(f"  ERROR {url}: {e}")
            time.sleep(REQUEST_DELAY)
    finally:
        if browser:
            browser.close()
        if pw:
            pw.stop()
    return sources


def load_manual_sources(log):
    """Files you saved yourself (browser 'Save as PDF', saved HTML, screenshots, text)."""
    sources = []
    MANUAL_DIR.mkdir(exist_ok=True)
    for path in sorted(MANUAL_DIR.iterdir()):
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                text = pdf_to_text(path.read_bytes(), log)
            elif ext in (".html", ".htm"):
                text = page_text(BeautifulSoup(path.read_bytes(), "lxml"))
            elif ext in (".png", ".jpg", ".jpeg", ".webp") and USE_VISION:
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext[1:]}"
                text = image_to_text(path.read_bytes(), mime)
            elif ext == ".txt":
                text = path.read_text(encoding="utf-8", errors="ignore")
            else:
                continue
            if text.strip():
                sources.append({"title": path.stem, "url": f"manual_sources/{path.name}", "text": text})
                log(f"  file: {path.name}")
        except Exception as e:
            log(f"  ERROR {path.name}: {e}")
    return sources


# ============================== PDF + index ==============================
def pick_font():
    """Nirmala UI (Windows) can display Tamil; fall back to Arial, then Helvetica."""
    for name, path in (("Nirmala", r"C:\Windows\Fonts\Nirmala.ttc"), ("Nirmala", r"C:\Windows\Fonts\Nirmala.ttf"),
                       ("Arial", r"C:\Windows\Fonts\arial.ttf")):
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=0))
                return name
            except Exception:
                continue
    return "Helvetica"


def build_pdf(sources):
    """All sources in ONE PDF; each source starts on a new page with its URL (used for citations)."""
    font = pick_font()
    styles = getSampleStyleSheet()
    for s in ("Title", "Normal", "BodyText"):
        styles[s].fontName = font
    story = []
    for i, src in enumerate(sources):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(escape(src["title"]), styles["Title"]))
        story.append(Paragraph("Source URL: " + escape(src["url"]), styles["Normal"]))
        story.append(Spacer(1, 8))
        for line in src["text"].splitlines():
            story.append(Paragraph(escape(line), styles["BodyText"]))
    SimpleDocTemplate(str(PDF_PATH), pagesize=A4, title="Tamil Nadu Agriculture Schemes").build(story)


def build_index():
    """Read the PDF back, tag every page with its section title and web URL, chunk, embed into ONE FAISS index.
    Every chunk starts with its section title (e.g. '[Scheme - KAVIADP ...]'), so a chunk from the
    middle of a long GO still says which scheme it belongs to."""
    pages, url, title = [], "", ""
    for doc in PyPDFLoader(str(PDF_PATH)).load():
        m = re.search(r"Source URL:\s*(\S+)", doc.page_content)
        if m:                                          # first page of a new section
            url = m.group(1)
            title = norm(doc.page_content[:m.start()])
        if doc.page_content.strip():
            doc.metadata = {"source": PDF_PATH.name, "page": doc.metadata.get("page", 0) + 1,
                            "url": url, "title": title}
            pages.append(doc)
    for doc in pages:
        doc.page_content = clean_repeats(doc.page_content)
    chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(pages)
    chunks = [c for c in chunks if len(set(c.page_content.split())) >= 8]   # drop near-empty / junk chunks
    for seq, c in enumerate(chunks):
        c.page_content = f"[{c.metadata['title']}]\n{c.page_content}"
        c.metadata["seq"] = seq                        # reading order, used to fetch neighbouring chunks
    store = FAISS.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))
    store.save_local(str(INDEX_DIR))
    return len(chunks)


def read_meta():
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else {}


def refresh_data(log=print):
    """Fetch -> PDF -> chunk + embed. Skips re-embedding when nothing changed since last time."""
    log(f"Refresh started {datetime.now():%Y-%m-%d %H:%M}")
    sources = []
    try:
        sources += crawl_website(log)
    except Exception as e:
        log(f"Website fetch failed: {e}")
    sources += load_manual_sources(log)
    if not sources:
        raise SystemExit("No content collected. The website may block automated access - "
                         "save the pages into manual_sources/ and refresh again.")

    digest = hashlib.sha256(json.dumps([INDEX_VERSION] + [s["text"] for s in sources]).encode()).hexdigest()
    meta, now = read_meta(), datetime.now().isoformat(timespec="seconds")
    if meta.get("hash") == digest and INDEX_DIR.exists():
        meta["checked"] = now
        log("No changes since the last refresh - existing index kept.")
    else:
        log(f"Writing {len(sources)} source(s) into {PDF_PATH.name} ...")
        build_pdf(sources)
        log("Chunking and embedding the PDF (OpenAI embeddings) ...")
        chunks = build_index()
        meta = {"hash": digest, "updated": now, "checked": now, "sources": len(sources), "chunks": chunks}
        log(f"Index built: {chunks} chunks from {len(sources)} source(s) -> {PDF_PATH.name}")
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


# ============================== retrieval + answer ==============================
class SearchPlan(BaseModel):
    """How to search for one user question."""
    question: str = Field(description="The question rewritten to stand alone: resolve 'it', 'that scheme', "
                                      "'its' from the conversation. Keep scheme names, numbers and language.")
    schemes: list[str] = Field(default_factory=list,
                               description="Names or short names of the specific schemes the question is "
                                           "about, as written (e.g. 'TNIAMP 2023-24'). Empty if none.")
    departments: list[str] = Field(default_factory=list,
                                   description="If the question is limited to one or more departments "
                                               "(e.g. 'horticulture schemes'), those department names copied "
                                               "exactly from the list given. Otherwise empty.")
    scope: Literal["specific", "across_schemes"] = Field(
        description="'specific' = about one or two named schemes, or one detail/GO. "
                    "'across_schemes' = needs ALL schemes: which/list/compare/how many/best for a farmer "
                    "group/overview of schemes.")


ACROSS_WORDS = ("which scheme", "which schemes", "list", "compare", "comparison", "how many", "all scheme",
                "schemes for", "schemes available", "overview", "best scheme", "எந்த", "அனைத்து")


def plan_question(question, history, cfg, departments=()):
    """Memory + routing in one small LLM call: rewrite follow-ups, decide the search scope,
    pick out named schemes and any department filter."""
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history) or "(no earlier messages)"
    prompt = (f"Departments in the scheme list: {', '.join(departments) or '(unknown)'}\n\n"
              f"Conversation so far:\n{convo}\n\nNew user question: {question}")
    try:
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, timeout=LLM_TIMEOUT)
        return llm.with_structured_output(SearchPlan).invoke(prompt, config={**cfg, "run_name": "plan_question"})
    except Exception:   # fall back to simple keyword rules if the planner call fails
        scope = "across_schemes" if any(w in question.lower() for w in ACROSS_WORDS) else "specific"
        depts = [d for d in departments if d.lower() in question.lower()]
        return SearchPlan(question=question, scope=scope, departments=depts)


def hybrid_search(question, cfg, k=TOP_K):
    """Hybrid RAG: semantic (FAISS) + keyword (BM25), merged by taking turns:
    semantic #1, keyword #1, semantic #2, keyword #2, ... so the best hits of EACH method always
    reach the model. BM25 catches exact words (scheme names, GO numbers, 'fruit crops');
    FAISS catches meaning ('help for buying a tractor')."""
    vec = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question, config=cfg)
    kw = st.session_state.bm25.invoke(question, config=cfg)
    merged, seen = [], set()
    for pair in zip(vec + [None] * (k - len(vec)), kw + [None] * (k - len(kw))):
        for d in pair:
            key = d and (d.metadata.get("page"), d.page_content[:100])
            if d and key not in seen:
                seen.add(key)
                merged.append(d)
    return merged[:k]


@st.cache_resource(show_spinner=False)
def load_scheme_catalogue(pdf_mtime):
    """Every 'Scheme - ...' record, read back from the PDF (cached until the PDF changes).
    Used for questions that must look at all schemes, and for the sidebar scheme browser."""
    sections, title, url = {}, "", ""
    for doc in PyPDFLoader(str(PDF_PATH)).load():
        text = doc.page_content
        m = re.search(r"Source URL:\s*(\S+)", text)
        if m:
            url, title, text = m.group(1), norm(text[:m.start()]), text[m.end():]
        if title.startswith(("Scheme - ", "GO & Guidelines - ")):
            sec = sections.setdefault(title, {"page": doc.metadata.get("page", 0) + 1, "url": url, "parts": []})
            sec["parts"].append(text.strip())
    catalogue, go_facts = [], []
    for title, sec in sections.items():
        body = "\n".join(sec["parts"])
        if title.startswith("GO & Guidelines - "):          # keep only the KEY FACTS part of each GO
            facts = body.split("FULL TEXT (transcription and translation):")[0].strip()
            if facts.startswith("KEY FACTS"):
                go_facts.append(Document(page_content=f"[{title}]\n{facts}",
                                         metadata={"source": PDF_PATH.name, "page": sec["page"],
                                                   "url": sec["url"], "title": title}))
            continue
        dept = re.search(r"Department:\s*(.+)", body)
        catalogue.append(Document(
            page_content=f"[{title}]\n{body}",
            metadata={"source": PDF_PATH.name, "page": sec["page"], "url": sec["url"], "title": title,
                      "department": dept.group(1).strip() if dept else "Other"}))
    return catalogue, go_facts


SYNONYMS = {"farm": "agricu", "farmer": "agricu", "farmers": "agricu", "agri": "agricu", "machinery": "mechan",
            "machine": "mechan", "machines": "mechan", "horti": "hortic", "organic": "organi", "solar": "solar"}
STOP_WORDS = {"scheme", "schemes", "the", "of", "for", "and", "under", "in", "on", "to", "a", "an", "project",
              "programme", "program", "mission", "sub"}


def words(text):
    """Comparable word stems: lowercase, first 6 letters (mechanisation ~ mechanization), synonyms mapped."""
    out = set()
    for w in re.findall(r"[a-z0-9\u0B80-\u0BFF]+", text.lower()):
        if w in STOP_WORDS:
            continue
        out.add(SYNONYMS.get(w, w[:6]))
    return out


def match_schemes(names, catalogue):
    """Find the full scheme records a question names, e.g. 'TNIAMP 2023-24' -> the TNIAMP -2023-24 record.
    Score = share of the asked words found in the scheme title; keeps the best match(es) per name."""
    found = []
    for name in names:
        asked = words(name)
        if not asked:
            continue
        scored = [(len(asked & words(d.metadata["title"])) / len(asked), d) for d in catalogue]
        best = max((score for score, _ in scored), default=0)
        if best >= (1.0 if len(asked) == 1 else 0.6):
            found += [d for score, d in scored if score == best and d not in found][:2]
    return found[:3]


def gather_context(plan, cfg):
    """'across_schemes' -> every scheme record + a few GO passages.
    'specific'        -> the FULL record of each named scheme (so no field is cut off by chunking)
                         + the best-matching chunks for anything else (GO details etc.)."""
    catalogue = st.session_state.catalogue
    if plan.scope == "across_schemes" and catalogue:
        # the PROGRAM picks the schemes (exact department filter), the LLM only formats them
        wanted = {d.lower() for d in plan.departments}
        selected = [d for d in catalogue if d.metadata["department"].lower() in wanted] if wanted else catalogue
        selected = selected or catalogue
        scope_text = f"department(s): {', '.join(sorted({d.metadata['department'] for d in selected}))}" if wanted \
            else "all departments"
        index_lines = [f"{n}. {d.metadata['title'].removeprefix('Scheme - ')} | Department: {d.metadata['department']}"
                       for n, d in enumerate(selected, start=1)]
        index = Document(
            page_content=(f"[Scheme index]\nThe scheme list has EXACTLY {len(selected)} scheme(s) for {scope_text}. "
                          f"Every one of them must appear in the answer, and no others:\n" + "\n".join(index_lines)),
            metadata={"source": PDF_PATH.name, "page": "index", "url": START_URL,
                      "title": f"Scheme index ({len(selected)} schemes, {scope_text})"})
        extra = [d for d in hybrid_search(plan.question, cfg, k=4) if not d.metadata.get("title", "").startswith("Scheme - ")]
        return [index] + selected + (extra if not wanted else [])
    records = match_schemes(plan.schemes, catalogue)
    for rec in list(records):                    # bring each named scheme's GO key facts along
        name = rec.metadata["title"].removeprefix("Scheme - ")
        records += [g for g in st.session_state.go_facts if name in g.metadata["title"]]
    titles = {d.metadata["title"] for d in records}
    chunks = [d for d in hybrid_search(plan.question, cfg) if d.metadata.get("title") not in titles]
    return records + with_neighbours(chunks[:max(6, TOP_K - len(records))])


def with_neighbours(docs, limit=6):
    """For the top GO passages, also include the chunk just before and just after it (same document),
    so a sentence like '...as detailed below:' arrives together with the table that follows it."""
    by_seq = st.session_state.get("by_seq", {})
    out, seen = [], set()

    def add(d):
        key = (d.metadata.get("page"), d.page_content[:100])
        if key not in seen:
            seen.add(key)
            out.append(d)

    for i, d in enumerate(docs):
        seq, title = d.metadata.get("seq"), d.metadata.get("title", "")
        if i < limit and seq is not None and title.startswith("GO & Guidelines"):
            for n in (seq - 1, seq, seq + 1):
                nb = by_seq.get(n)
                if nb is not None and nb.metadata.get("title") == title:
                    add(nb)
        else:
            add(d)
    return out


def to_messages(history):
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in history]


def stream_answer(question, history, docs, cfg, usage):
    """Streaming step: yields text pieces as the model writes; collects token usage at the end."""
    context = "\n\n".join(
        f"[{i}] (PDF page {d.metadata['page']}, source: {d.metadata.get('url', '')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1))
    messages = [SystemMessage(SYSTEM_PROMPT), *to_messages(history),
                HumanMessage(f"Context:\n{context}\n\nQuestion: {question}")]
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, stream_usage=True, timeout=LLM_TIMEOUT)
    for chunk in llm.stream(messages, config={**cfg, "run_name": "answer"}):
        if chunk.usage_metadata:
            usage.update(chunk.usage_metadata)
        if chunk.content:
            yield chunk.content


def show_sources(docs):
    with st.expander(f"📄 Sources ({len(docs)})"):
        for i, d in enumerate(docs, start=1):
            url = d.metadata.get("url", "")
            link = f"[open]({url})" if url.startswith("http") else url
            st.markdown(f"**[{i}] {d.metadata.get('title', '')}** · {d.metadata['source']}, "
                        f"page {d.metadata['page']} · {link}")
            st.caption(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))


# ============================== look & feel ==============================
def hero_svg():
    """Hand-drawn farm scene (sun, fields, furrows, paddy) - no external image needed."""
    stalks = []
    for i, x in enumerate(range(770, 1200, 16)):
        h = 55 + (i * 37) % 30
        stalks.append(f'<path d="M{x} 240 Q{x + 6} {240 - h / 2} {x + 14} {240 - h}" stroke="#4f8a2b" '
                      f'stroke-width="3" fill="none"/>')
        for g in range(4):
            gy = 240 - h + 6 + g * 7
            stalks.append(f'<ellipse cx="{x + 14 - g * 1.5}" cy="{gy}" rx="3.2" ry="5.5" fill="#e0b23c" '
                          f'transform="rotate(25 {x + 14 - g * 1.5} {gy})"/>')
    furrows = "".join(f'<path d="M{-50 + i * 90} 240 Q {250 + i * 60} 222 {600 + i * 45} 216" '
                      f'stroke="#2f6b22" stroke-width="2" fill="none" opacity=".35"/>' for i in range(9))
    rays = "".join(f'<line x1="{1060 + 62 * cos}" y1="{62 + 62 * sin}" x2="{1060 + 80 * cos}" y2="{62 + 80 * sin}" '
                   f'stroke="#f2b134" stroke-width="4" stroke-linecap="round" opacity=".7"/>'
                   for cos, sin in [(1, 0), (.7, .7), (0, 1), (-.7, .7), (-1, 0), (-.7, -.7), (0, -1), (.7, -.7)])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 240" preserveAspectRatio="xMidYMid slice">
      <defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#fdf7e4"/><stop offset="1" stop-color="#e6f2d3"/></linearGradient></defs>
      <rect width="1200" height="240" fill="url(#sky)"/>
      <circle cx="1060" cy="62" r="44" fill="#f6c453"/>{rays}
      <path d="M0 188 Q 300 160 620 180 T 1200 166 V240 H0Z" fill="#b6da8a"/>
      <path d="M0 206 Q 360 184 720 200 T 1200 190 V240 H0Z" fill="#7cb552"/>
      <path d="M0 222 Q 420 206 820 220 T 1200 214 V240 H0Z" fill="#4a8f34"/>
      {furrows}{''.join(stalks)}</svg>"""
    return base64.b64encode(svg.encode()).decode()


def inject_style():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&family=Noto+Sans+Tamil:wght@400;600&display=swap');
:root {{ --leaf:#2e7d32; --leaf-dark:#1b4d1f; --paddy:#e0b23c; --soil:#7a5230; --cream:#fbf8ef; }}
html, body, [class*="css"], .stMarkdown, .stChatMessage {{ font-family:'Noto Sans','Noto Sans Tamil',sans-serif; }}
.stApp {{ background: var(--cream); }}
.block-container {{ padding-top: 3.2rem; max-width: 980px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.hero {{ position:relative; border-radius:18px; overflow:hidden; min-height:250px; margin-bottom:1rem;
        background:url("data:image/svg+xml;base64,{hero_svg()}") center/cover no-repeat;
        box-shadow:0 6px 24px rgba(27,77,31,.15); }}
.hero-text {{ position:relative; padding:24px 30px 64px; max-width:64%; }}
.hero .eyebrow {{ color:var(--soil); font-weight:600; font-size:.95rem; letter-spacing:.3px; }}
.hero h1 {{ color:var(--leaf-dark); font-size:2.0rem; line-height:1.15; margin:.25rem 0 .4rem; font-weight:700; }}
.hero p {{ color:#33502f; font-size:.98rem; margin:0; }}
.pill {{ display:inline-block; background:rgba(255,255,255,.75); border:1px solid #cfe3bd; color:var(--leaf-dark);
        border-radius:999px; padding:3px 10px; font-size:.78rem; margin:10px 6px 0 0; }}
[data-testid="stChatMessage"] {{ background:#ffffff; border:1px solid #e3ead8; border-radius:14px;
        padding:.8rem 1rem; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
[data-testid="stChatMessage"] table {{ font-size:.9rem; }}
[data-testid="stChatMessage"] th {{ background:#eef6e6; }}
.stButton > button {{ border-radius:12px; border:1px solid #bcd9a5; background:#ffffff; color:var(--leaf-dark);
        text-align:left; width:100%; }}
.stButton > button:hover {{ border-color:var(--leaf); color:var(--leaf); background:#f3f9ec; }}
[data-testid="stSidebar"] {{ background:#eef5e6; }}
.stat {{ background:#ffffff; border:1px solid #d7e7c8; border-radius:12px; padding:8px 12px; margin-bottom:8px; }}
.stat b {{ font-size:1.35rem; color:var(--leaf-dark); display:block; }}
.stat span {{ font-size:.8rem; color:#5d6f55; }}
.muted {{ color:#6b7a63; font-size:.85rem; }}
</style>""", unsafe_allow_html=True)


def hero(scheme_count):
    st.markdown(f"""
<div class="hero"><div class="hero-text">
  <div class="eyebrow">🌾 தமிழ்நாடு வேளாண்மைத் திட்டங்கள்</div>
  <h1>Tamil Nadu Agriculture<br>Scheme Assistant</h1>
  <p>Ask about any scheme - purpose, eligibility, subsidy, documents or GO - in English or Tamil.</p>
  <span class="pill">✔ Official scheme pages only</span><span class="pill">📚 {scheme_count} schemes</span>
  <span class="pill">🔄 Refreshed weekly</span>
</div></div>""", unsafe_allow_html=True)


EXAMPLES = [
    "Which schemes are available for small and marginal farmers?",
    "What subsidy is available for the Chief Minister's Scheme of Solar Powered Pump-set?",
    "What documents are needed for TNIAMP 2023-24?",
    "Who is eligible for the Rainfed Area Development - IFS 2024-25 scheme?",
]


# ============================== Streamlit app ==============================
def run_refresh(log_box):
    """Run the fetch in a separate process (Playwright is most reliable outside Streamlit's thread)
    and show its progress live in the app."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--refresh"], cwd=APP_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    lines, start = [], time.time()
    for line in proc.stdout:                       # one line per fetched page, shown as it happens
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-25:]), language="text")
        if time.time() - start > MAX_REFRESH_MINUTES * 60:
            proc.kill()
            lines.append(f"Stopped: refresh took longer than {MAX_REFRESH_MINUTES} minutes.")
            break
    proc.wait()
    log_box.code("\n".join(lines[-25:]), language="text")
    return proc.returncode == 0


def days_since(iso):
    return (datetime.now() - datetime.fromisoformat(iso)).days if iso else 999


def main():
    st.set_page_config(page_title="TN Agri Scheme Assistant", page_icon="🌾", layout="wide")
    inject_style()

    if not os.environ.get("OPENAI_API_KEY"):
        hero("?")
        st.error("OPENAI_API_KEY is not set. Add it to the .env file next to this script and restart.")
        st.stop()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", str(uuid.uuid4()))

    # ---- build or weekly refresh ----
    meta = read_meta()
    stale = not INDEX_DIR.exists() or days_since(meta.get("checked")) >= REFRESH_DAYS
    if (stale and "refresh_tried" not in st.session_state) or st.session_state.pop("force_refresh", False):
        st.session_state.refresh_tried = True
        hero("…")
        with st.status("Fetching scheme pages and building the index...", expanded=True) as status:
            ok = run_refresh(st.empty())
            status.update(label="Refresh finished" if ok else "Refresh failed", state="complete" if ok else "error")
        st.session_state.pop("vectorstore", None)
        meta = read_meta()
        if ok:
            st.success(f"Index built: {meta.get('chunks', 0)} chunks from {meta.get('sources', 0)} source(s).")
        elif not INDEX_DIR.exists():
            st.error("Could not collect any scheme data. Save the scheme pages (Ctrl+P -> Save as PDF) "
                     f"into {MANUAL_DIR} and click 'Refresh data now'.")
        if not ok or not INDEX_DIR.exists():
            st.stop()
        st.rerun()

    # ---- load the store + scheme catalogue into session_state ----
    if "vectorstore" not in st.session_state and INDEX_DIR.exists():
        with st.spinner("Loading the scheme knowledge base..."):
            store = FAISS.load_local(str(INDEX_DIR), OpenAIEmbeddings(model=EMBED_MODEL),
                                     allow_dangerous_deserialization=True)   # safe: we created this file
            chunks = list(store.docstore._dict.values())
            bm25 = BM25Retriever.from_documents(chunks, preprocess_func=bm25_tokens)
            bm25.k = TOP_K
            catalogue, go_facts = load_scheme_catalogue(PDF_PATH.stat().st_mtime) if PDF_PATH.exists() else ([], [])
        st.session_state.update(vectorstore=store, bm25=bm25, chunk_count=len(chunks),
                                catalogue=catalogue, go_facts=go_facts,
                                by_seq={c.metadata["seq"]: c for c in chunks if "seq" in c.metadata})
    if "vectorstore" not in st.session_state:
        hero("?")
        st.stop()

    catalogue = st.session_state.catalogue

    # ---- sidebar ----
    with st.sidebar:
        st.markdown("### 🌾 Knowledge base")
        st.markdown(f"""<div class="stat"><b>{len(catalogue)}</b><span>schemes</span></div>
<div class="stat"><b>{st.session_state.chunk_count}</b><span>chunks indexed (FAISS + BM25)</span></div>
<div class="stat"><b>{(meta.get('updated') or '?')[:10]}</b><span>data last changed · checked {(meta.get('checked') or '?')[:10]}</span></div>""",
                    unsafe_allow_html=True)

        st.markdown("### 🔎 Browse schemes")
        departments = sorted({d.metadata["department"] for d in catalogue})
        dept = st.selectbox("Department", ["All departments"] + departments, label_visibility="collapsed")
        names = [d.metadata["title"].removeprefix("Scheme - ") for d in catalogue
                 if dept == "All departments" or d.metadata["department"] == dept]
        scheme = st.selectbox("Scheme", names, label_visibility="collapsed") if names else None
        if scheme and st.button("📋 Summarise this scheme"):
            st.session_state.pending = (f"Give me a summary of the scheme '{scheme}': purpose, eligibility, "
                                        "subsidy pattern, documents required and GO details.")
            st.rerun()

        st.markdown("### ⚙️ Tools")
        if PDF_PATH.exists():
            st.download_button("⬇️ Download scheme PDF", PDF_PATH.read_bytes(), PDF_PATH.name, "application/pdf")
        if st.button("🔄 Refresh data now"):
            st.session_state.force_refresh = True
            st.rerun()
        if st.button("🧹 Clear chat"):
            st.session_state.messages = []
            st.rerun()
        tracing = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        st.markdown(f"<div class='muted'>LangSmith tracing: "
                    f"{'ON · ' + os.environ.get('LANGSMITH_PROJECT', 'default') if tracing else 'OFF'}</div>",
                    unsafe_allow_html=True)

    # ---- main area ----
    hero(len(catalogue))


    for m in st.session_state.messages:            # chat history (memory)
        with st.chat_message(m["role"], avatar="🧑‍🌾" if m["role"] == "user" else "🌾"):
            st.markdown(m["content"])
            if m.get("docs"):
                show_sources(m["docs"])
            if m.get("note"):
                st.caption(m["note"])

    question = st.chat_input("Ask about a scheme - eligibility, subsidy, documents, GO... (English / தமிழ்)")
    question = question or st.session_state.pop("pending", None)
    if not question:
        if not st.session_state.messages:          # first visit: one-click example questions
            st.markdown("**Try asking:**")
            cols = st.columns(2)
            for i, example in enumerate(EXAMPLES):
                if cols[i % 2].button(example, key=f"ex{i}"):
                    st.session_state.pending = example
                    st.rerun()
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-MEMORY_TURNS * 2:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(question)

    cfg = {"tags": ["tn-agri-rag"], "metadata": {"session_id": st.session_state.session_id}}
    with st.chat_message("assistant", avatar="🌾"):
        with st.spinner("Searching the scheme documents..."):
            plan = plan_question(question, history, cfg, sorted({d.metadata["department"] for d in catalogue}))
            docs = gather_context(plan, {**cfg, "metadata": {**cfg["metadata"], "scope": plan.scope}})
        usage = {}
        answer = st.write_stream(stream_answer(question, history, docs, cfg, usage))
        show_sources(docs)
        if plan.scope == "across_schemes":
            n = sum(1 for d in docs if d.metadata.get("title", "").startswith("Scheme - "))
            scope_note = (f"Filtered to {n} scheme(s) in {', '.join(plan.departments)}" if plan.departments
                          else f"Searched all {n} schemes")
        else:
            scope_note = "Searched best-matching passages"
        note = (f"{scope_note} · tokens {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out"
                if usage else scope_note)
        st.caption(note)
    st.session_state.messages.append({"role": "assistant", "content": answer, "docs": docs, "note": note})


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        refresh_data()
    else:
        main()