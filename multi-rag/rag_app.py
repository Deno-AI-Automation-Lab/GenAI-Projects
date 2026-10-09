"""Multi-PDF RAG app: LangChain + FAISS + OpenAI + Streamlit.

Indexes every PDF in the same folder as this script automatically (no upload needed).

Run:
    pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pypdf python-dotenv
    Put OPENAI_API_KEY=sk-... in a .env file next to this script (or set it as an environment variable)
    streamlit run rag_app.py
"""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- settings ----------
APP_DIR = Path(__file__).resolve().parent   # PDFs are read from this folder
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
INDEX_DIR = APP_DIR / "faiss_index"         # local copy of the vector DB on disk

st.set_page_config(page_title="Multi-PDF Q&A", page_icon="📄")
st.title("📄 Ask questions across your PDFs")

# ---------- API key from environment / .env ----------
load_dotenv(APP_DIR / ".env")
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Put OPENAI_API_KEY=sk-... in a .env file next to rag_app.py "
             "(or set it as an environment variable) and restart the app.")
    st.stop()

# ---------- session state ----------
st.session_state.setdefault("vectorstore", None)   # the FAISS store
st.session_state.setdefault("indexed_files", [])
st.session_state.setdefault("chunk_count", 0)
st.session_state.setdefault("chat", [])            # list of (question, answer, sources)


def load_and_chunk(pdf_paths):
    """Read every PDF page by page, tag each page with file name + page number, then chunk."""
    pages = []
    for path in pdf_paths:
        for doc in PyPDFLoader(str(path)).load():
            if not doc.page_content.strip():
                continue                      # skip empty / scanned pages with no text
            doc.metadata = {
                "source": path.name,                          # file name shown to the reader
                "page": doc.metadata.get("page", 0) + 1,      # 1-based page number
            }
            pages.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(pages)    # chunks keep the source + page metadata


def build_index():
    """Chunk and embed ALL PDFs in the folder into ONE FAISS store and keep it in session_state."""
    pdfs = sorted(APP_DIR.glob("*.pdf"))
    if not pdfs:
        st.error(f"No PDF files found in {APP_DIR}")
        st.stop()

    with st.spinner(f"Reading, chunking and embedding {len(pdfs)} PDF file(s)..."):
        chunks = load_and_chunk(pdfs)
        if not chunks:
            st.error("No text found in the PDFs (they may be scanned images).")
            st.stop()
        store = FAISS.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))
        store.save_local(str(INDEX_DIR))

    st.session_state.vectorstore = store
    st.session_state.indexed_files = [p.name for p in pdfs]
    st.session_state.chunk_count = len(chunks)
    st.session_state.chat = []


def answer(question, store):
    """Retrieve the most relevant chunks from FAISS and ask the LLM to answer only from them."""
    docs = store.similarity_search(question, k=TOP_K)

    context = "\n\n".join(
        f"[{i}] (file: {d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite the sources you used with their numbers in square brackets, e.g. [1] or [2][3]. "
        "If the answer is not in the context, say you could not find it in the documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return llm.invoke(prompt).content, docs


def show_sources(docs):
    """List each retrieved chunk with file and page, plus the exact text so the reader can verify it."""
    st.markdown("**Sources**")
    for i, d in enumerate(docs, start=1):
        st.markdown(f"[{i}] **{d.metadata['source']}**, page {d.metadata['page']}")
    with st.expander("Show the retrieved text"):
        for i, d in enumerate(docs, start=1):
            st.markdown(f"**[{i}] {d.metadata['source']} – page {d.metadata['page']}**")
            st.text(d.page_content)


# ---------- build the index automatically on first load ----------
if st.session_state.vectorstore is None:
    build_index()
    st.success(f"Index built: {st.session_state.chunk_count} chunks from "
               f"{len(st.session_state.indexed_files)} PDF file(s).")

# ---------- sidebar: what is indexed ----------
with st.sidebar:
    st.header("Indexed documents")
    st.write(f"**{st.session_state.chunk_count} chunks** from {len(st.session_state.indexed_files)} file(s):")
    for name in st.session_state.indexed_files:
        st.caption(f"• {name}")
    if st.button("Rebuild index", help="Use after adding, removing or changing PDFs in the folder"):
        build_index()
        st.rerun()

# ---------- main: Q&A ----------
for q, a, docs in st.session_state.chat:          # replay earlier questions
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        show_sources(docs)

question = st.chat_input("Ask something about your PDFs")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            text, docs = answer(question, st.session_state.vectorstore)
        st.write(text)
        show_sources(docs)
    st.session_state.chat.append((question, text, docs))
