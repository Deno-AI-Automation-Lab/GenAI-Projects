# 🌾 Tamil Nadu Agriculture Scheme Assistant

A bilingual (English / தமிழ்) RAG chatbot that answers farmers' questions about Tamil Nadu Government
agriculture schemes — subsidy patterns, eligibility, documents required and G.O. details — **using only
the official scheme pages** from [tnagrisnet.tn.gov.in](https://www.tnagrisnet.tn.gov.in/people_app/GoScheme).

Built with **LangChain + FAISS + BM25 + OpenAI + Streamlit**, in a single Python file.

---

## ✨ Features

**Data ingestion**
- Crawls the scheme list page and its scheme sub-links (depth 2, max 150 pages), respecting `robots.txt`
- `WebBaseLoader` for static pages; **Playwright** for JavaScript/AJAX pages
- Expands paginated DataTables to show all rows and clicks every **"Scheme Details View" popup** to capture components and subsidy patterns
- Downloads linked G.O. PDFs; scanned pages and images are read with **GPT-4o vision** (Tamil transcription + English translation)
- Writes an English **KEY FACTS** summary for every G.O. so English questions find Tamil documents
- Skips placeholder/test rows (`tester`, `dummy`, …) and cleans runaway OCR repetition
- Vision and summary results are **cached** in `ocr_cache/`, so weekly refreshes don't pay twice
- Anything you save yourself in `manual_sources/` (PDF, HTML, PNG/JPG/WEBP, TXT) is included too

**Indexing**
- All sources are written into **one PDF** (`tn_agri_schemes.pdf`), then chunked (1200 / 200 overlap) and embedded into **one FAISS index**
- Every chunk is prefixed with its section title, so a chunk from the middle of a long G.O. still knows its scheme
- Content hash check — re-embedding is skipped when nothing changed
- Auto-refresh when data is older than **7 days**

**Question answering**
- **Query planner** (structured output): rewrites follow-up questions, detects named schemes, department filters and scope
- **Hybrid retrieval**: FAISS semantic search + BM25 keyword search, interleaved
- `across_schemes` questions ("list all horticulture schemes") get an exact scheme index built by code, not the LLM
- `specific` questions get the **full scheme record** + G.O. key facts + neighbouring G.O. chunks
- Strict grounding: "I could not find this in the scheme documents." when the answer isn't there
- Streaming answers with chat memory (last 4 turns)

**Farmer-friendly UI**
- English / Tamil interface toggle
- One-tap topic tiles (machinery, irrigation, solar, organic, sugarcane, women & SC/ST …)
- 🎤 Voice questions (speech-to-text) and 🔊 Listen (text-to-speech)
- Browse all schemes as cards with search + department filter
- Share answer on WhatsApp, save answer, 👍/👎 feedback (CSV + LangSmith)
- Admin panel: stats, download scheme PDF / feedback CSV, manual refresh, tracing status

---

## 🧱 Architecture

```
GoScheme page ─┐
Scheme links ──┼─► Crawler (WebBaseLoader / Playwright + popups)
G.O. PDFs ─────┤        │  scanned pages / images ─► GPT-4o vision (cached)
manual_sources ┘        ▼
                tn_agri_schemes.pdf  ──► chunk ──► OpenAI embeddings ──► FAISS index
                                                                  └──► BM25 (built at load)
User question ─► Planner (gpt-4o-mini) ─► Hybrid retrieval / scheme catalogue
                                       ─► gpt-4o-mini (streaming, grounded) ─► Streamlit UI
```

---

## 📋 Requirements

- Python **3.10+**
- An **OpenAI API key** (chat, vision, embeddings, speech)
- Windows recommended: the PDF uses the **Nirmala UI** font to render Tamil.
  On Linux/macOS it falls back to Helvetica and Tamil text in the PDF will not display correctly
  (edit `pick_font()` to point to a Tamil font such as Noto Sans Tamil).

---

## 🚀 Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 2. Install packages
pip install -r requirements.txt

# 3. Install the Playwright browser
python -m playwright install chromium
```

Create a `.env` file next to the script:

```env
OPENAI_API_KEY=sk-...

# Optional - LangSmith tracing & feedback
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=tn-agri-scheme-rag
```

---

## ▶️ Run

```bash
# Start the app (builds the index automatically on first run)
streamlit run tnagrischemes.py

# Fetch + rebuild only, no UI (use this for scheduled refreshes)
python tnagrischemes.py --refresh
```

The first build can take a while — scanned G.O. PDFs go through the vision model page by page.
Later refreshes reuse `ocr_cache/` and are much faster.

### Weekly refresh on Windows (Task Scheduler)

```bat
schtasks /create /tn "TN Agri Scheme Refresh" /sc weekly /d SUN /st 02:00 ^
  /tr "\"C:\path\to\.venv\Scripts\python.exe\" \"C:\path\to\tnagrischemes.py\" --refresh"
```

---

## 📁 Project layout

```
.
├── tnagrischemes.py        # the whole app
├── requirements.txt
├── .env                    # API keys (do not commit)
├── manual_sources/         # your own saved PDFs / HTML / screenshots / TXT
├── tn_agri_schemes.pdf     # generated: all collected content in one PDF
├── faiss_index/            # generated: vector index
├── index_meta.json         # generated: content hash, last updated/checked, counts
├── ocr_cache/              # generated: cached vision transcriptions + G.O. summaries
└── feedback.csv            # generated: 👍/👎 ratings
```

Suggested `.gitignore`:

```
.env
.venv/
faiss_index/
ocr_cache/
tn_agri_schemes.pdf
index_meta.json
feedback.csv
__pycache__/
```

---

## ⚙️ Key settings (top of the script)

| Setting | Default | What it controls |
|---|---|---|
| `START_URL` | GoScheme page | Where the crawl starts |
| `MAX_DEPTH` / `MAX_PAGES` | 2 / 150 | Crawl depth and page limit |
| `REQUEST_DELAY` | 1.0 s | Politeness delay between requests |
| `USE_VISION` | `True` | OCR of images / scanned PDF pages |
| `VISION_MODEL` | `gpt-4o` | Model for Tamil scans |
| `OCR_WORKERS` | 4 | Parallel vision calls (lower if rate-limited) |
| `CHAT_MODEL` | `gpt-4o-mini` | Planner, answers, G.O. summaries |
| `EMBED_MODEL` | `text-embedding-3-small` | Embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1200 / 200 | Chunking |
| `TOP_K` | 12 | Passages given to the model |
| `MEMORY_TURNS` | 4 | Q&A pairs remembered |
| `REFRESH_DAYS` | 7 | Auto-refresh interval |
| `INDEX_VERSION` / `OCR_VERSION` | 4 / v2 | Bump to force a rebuild / re-OCR |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | Add it to `.env` next to the script and restart |
| "No content collected" | The site may block automation — open each page, **Ctrl+P → Save as PDF** into `manual_sources/`, then click **Refresh data now** |
| Playwright errors | Run `python -m playwright install chromium` again |
| `pymupdf` blocked on Windows | Allow it in your antivirus / Smart App Control; the chat still runs, but PDF reading needs it |
| OpenAI rate-limit errors during refresh | Lower `OCR_WORKERS` to 1–2 |
| Tamil shows as boxes in the PDF | Point `pick_font()` to a Tamil-capable `.ttf` |
| Voice / feedback buttons missing | Upgrade Streamlit: `pip install -U streamlit` |

---

## ⚠️ Disclaimer

Information is taken from the official Tamil Nadu agriculture scheme pages and government orders, and
G.O. text may be machine-transcribed from scans. Always confirm with your nearest Agriculture /
Horticulture office before applying. This is a personal research project, not an official government tool.
