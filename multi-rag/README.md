# 📄 Multi-PDF Q&A (RAG with LangChain + FAISS + OpenAI + Streamlit)

A simple Retrieval-Augmented Generation (RAG) app that **indexes every PDF in its own folder** and lets
you ask questions across all of them. Answers come only from your documents, with **numbered citations**
showing the file name and page number, plus the exact retrieved text so you can verify each answer.

## ✨ Features

- No upload step: every `*.pdf` next to `rag_app.py` is indexed automatically
- One FAISS vector index across all PDFs
- Each chunk is tagged with its **file name** and **page number** (1-based)
- Grounded answers with `[1]`, `[2][3]` citations and an "I could not find it" fallback
- Sources panel with an expandable view of the retrieved text
- Sidebar listing the indexed files and chunk count, plus a **Rebuild index** button
- Chat history for the current session

## 🧱 How it works

```
PDFs in folder ─► PyPDFLoader (page by page) ─► RecursiveCharacterTextSplitter (1000 / 150)
                ─► OpenAI embeddings (text-embedding-3-small) ─► FAISS index

Question ─► FAISS similarity search (top 5) ─► gpt-4o-mini (answer ONLY from context) ─► answer + sources
```

## 📋 Requirements

- Python 3.9+
- An OpenAI API key
- PDFs with a **text layer** (scanned image-only PDFs are skipped)

## 🚀 Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

Create a `.env` file next to `rag_app.py`:

```env
OPENAI_API_KEY=sk-...
```

Copy the PDFs you want to query into the same folder.

## ▶️ Run

```bash
streamlit run rag_app.py
```

The app opens at **http://localhost:8501**, builds the index and shows how many chunks were created.

## 💬 Using it

1. Type a question, e.g. *"What are the eligibility criteria?"* or *"Compare the pricing in both documents."*
2. Read the answer and check the **Sources** list (file and page).
3. Expand **Show the retrieved text** to see exactly what the model was given.
4. After adding, removing or changing PDFs, click **Rebuild index** in the sidebar.

## ⚙️ Settings (top of the script)

| Setting | Default | Meaning |
|---|---|---|
| `CHAT_MODEL` | `gpt-4o-mini` | Model that writes the answer |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Chunk length and overlap in characters |
| `TOP_K` | 5 | Chunks retrieved per question |
| `INDEX_DIR` | `faiss_index/` | Where the index is saved |

## ⚠️ Things to know

| Point | Detail |
|---|---|
| Re-embedding on every new session | The index is **saved** to `faiss_index/` but not **loaded** back. Each new browser session or restart re-embeds all PDFs, which costs time and API usage on large folders |
| No conversational memory | Each question is answered on its own. Follow-ups like "what about the second one?" won't resolve |
| Scanned PDFs | Pages with no extractable text are skipped; run OCR first if you need them |
| Chat history | Kept only for the browser session; it's cleared on rebuild |

## 🛠️ Troubleshooting

| Message | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | Add it to `.env` next to `rag_app.py` and restart |
| `No PDF files found in ...` | Put `.pdf` files in the same folder as the script |
| `No text found in the PDFs` | The PDFs are scanned images; OCR them first |

## 💡 Ideas for next steps

- Load `faiss_index/` on startup and rebuild only when the PDFs change (compare file hashes)
- Add chat memory and question rewriting for follow-ups
- Add hybrid search (BM25 + FAISS) for exact terms and numbers
- Stream answers with `st.write_stream`
- Add LangSmith tracing

## 📁 Project layout

```
.
├── rag_app.py
├── requirements.txt
├── .env              # OPENAI_API_KEY (do not commit)
├── *.pdf             # your documents
└── faiss_index/      # generated vector index
```
