# 🌾 Tamil Nadu Agriculture Scheme Assistant

## Overview

This is a chatbot for farmers. It answers questions about Tamil Nadu Government agriculture schemes, in **English or Tamil**.

- It uses **only** the official scheme website: https://www.tnagrisnet.tn.gov.in/people_app/GoScheme
- It does **not** search the internet when answering.
- The script works in **three stages**:
  1. Collect the data
  2. Build the knowledge base
  3. Answer questions

---

## Stage 1: Collect the data

**Step 1. Check permission**
- Reads the website's `robots.txt` before fetching any page.
- **Tools:** `urllib.robotparser`, `requests`

**Step 2. Read the scheme list page**
- Opens the page in a browser.
- Shows all table rows.
- Clicks each scheme's popup to read its subsidy details.
- **Tools:** Playwright (headless Chrome), BeautifulSoup

**Step 3. Read the scheme detail pages**
- Normal pages are fetched directly.
- Pages built by JavaScript are opened in the browser instead.
- **Tools:** LangChain WebBaseLoader, Playwright

**Step 4. Read the Government Order (G.O.) PDFs and images**
- PDFs with text are read directly.
- Scanned Tamil pages and images are converted to text and translated to English.
- **Tools:** PyMuPDF, OpenAI `gpt-4o` (vision)

**Step 5. Summarise each G.O.**
- Writes an English "key facts" list for each G.O.: number, date, budget, components and eligibility.
- **Tool:** OpenAI `gpt-4o-mini`

**Step 6. Add your own files (optional)**
- Reads any PDF, HTML, image or TXT file you place in the `manual_sources/` folder.

---

## Stage 2: Build the knowledge base

**Step 7. Combine everything into one PDF**
- All collected content is saved in `tn_agri_schemes.pdf`.
- **Tool:** ReportLab

**Step 8. Create the search index**
- Splits the PDF into small chunks.
- Converts each chunk into embeddings (numbers that capture meaning).
- Saves a meaning-based index and a keyword index.
- **Tools:** PyPDFLoader, RecursiveCharacterTextSplitter, OpenAI `text-embedding-3-small`, FAISS, BM25

---

## Stage 3: Answer questions

**Step 9. Understand the question**
- Rewrites follow-up questions so they make sense alone (for example, "its documents" → "documents for scheme X").
- Decides whether the question is about one scheme or all schemes.
- **Tools:** OpenAI `gpt-4o-mini`, Pydantic

**Step 10. Find the relevant information**
- Takes the full record of the scheme being asked about.
- Adds the best passages found by meaning search and by keyword search.
- **Tools:** FAISS + BM25 (hybrid search)

**Step 11. Write the answer**
- Uses only the information found in Step 10.
- Gives a short answer in simple words, in the same language as the question.
- Shows the answer as it is being written.
- **Tool:** OpenAI `gpt-4o-mini`

**Step 12. Show it in the app**
- Chat window with one-tap topic buttons and a scheme browser
- Voice input and a "Listen" button
- WhatsApp sharing and 👍/👎 feedback
- **Tools:** Streamlit, OpenAI speech-to-text and text-to-speech

**Step 13. Monitor and keep the data fresh**
- Records every step and all feedback.
- Collects the data again automatically when it is more than 7 days old.
- **Tools:** LangSmith, `index_meta.json`

---

## Setup

**1. Install the packages**

```bat
pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters ^
            faiss-cpu pypdf pymupdf rank_bm25 beautifulsoup4 lxml reportlab requests python-dotenv playwright
python -m playwright install chromium
```

**2. Create a `.env` file next to the script**

```ini
OPENAI_API_KEY=sk-...

# Optional - for LangSmith monitoring
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=tn-agri-scheme-rag
```

---

## Run

| Purpose | Command |
|---|---|
| Start the app | `streamlit run tnagrischemes.py` |
| Refresh the data only (for a weekly schedule) | `python tnagrischemes.py --refresh` |

---

> **Note:** Information comes from the official scheme pages. Farmers should confirm details with their nearest Agriculture / Horticulture office before applying.
