"""PDF -> chunks -> OpenAI embeddings -> FAISS (LangChain).

Run directly to (re)build the index:
    python -m rag.vectorstore
"""
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Repeated page header in the handbook, e.g.
# "FiberNet Broadband - Customer Support Troubleshooting Handbook Page 7"
_HEADER_RE = re.compile(r"FiberNet Broadband - Customer Support Troubleshooting Handbook\s*Page\s*\d+")
# Section headings like "7. Slow Internet / Speed Complaints"
_SECTION_RE = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][^\n]{3,80})$", re.MULTILINE)


def _embeddings() -> OpenAIEmbeddings:
    config.require_api_key()
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)


def load_pdf(pdf_path: Path) -> List[Document]:
    """Load the PDF page by page, clean headers and tag each page with its section."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = PyPDFLoader(str(pdf_path)).load()
    current_section = "Introduction"
    cleaned: List[Document] = []
    for doc in pages:
        text = _HEADER_RE.sub("", doc.page_content).strip()
        match = _SECTION_RE.search(text)
        if match:
            current_section = f"{match.group(1)}. {match.group(2).strip()}"
        cleaned.append(Document(
            page_content=text,
            metadata={
                "source": pdf_path.name,
                "page": doc.metadata.get("page", 0) + 1,  # 1-based for humans
                "section": current_section,
            },
        ))
    return cleaned


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        # Prefix the section so each chunk carries its own context into the embedding
        c.page_content = f"[Section: {c.metadata['section']}]\n{c.page_content}"
    return chunks


def build_vectorstore(pdf_path: Path = config.PDF_PATH,
                      save_dir: Path = config.VECTORSTORE_DIR) -> FAISS:
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    store = FAISS.from_documents(chunks, _embeddings())
    save_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(save_dir))
    print(f"Indexed {len(docs)} pages into {len(chunks)} chunks -> {save_dir}")
    return store


def load_vectorstore(save_dir: Path = config.VECTORSTORE_DIR) -> FAISS:
    """Load the saved index, building it first if it doesn't exist yet."""
    if not (save_dir / "index.faiss").exists():
        return build_vectorstore(save_dir=save_dir)
    # Safe here: we only load an index this app created itself.
    return FAISS.load_local(str(save_dir), _embeddings(), allow_dangerous_deserialization=True)


if __name__ == "__main__":
    build_vectorstore()
