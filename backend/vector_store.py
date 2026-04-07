from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path
from pprint import pprint

from backend.llm import get_llm

from backend.embeddings import get_embeddings

INDEX_DIR = Path("faiss_index")


def build_vector_store(documents: Iterable[Document]) -> FAISS:
    """Create a FAISS index from LangChain documents."""
    embeddings = get_embeddings()
    return FAISS.from_documents(list(documents), embeddings)


def save_vector_store(vector_store: FAISS, index_dir: Path | str = INDEX_DIR) -> None:
    """Persist the FAISS index locally for later reuse."""
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))


def load_vector_store(index_dir: Path | str = INDEX_DIR) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    index_path = Path(index_dir)
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def similarity_search(query: str, k: int = 4, index_dir: Path | str = INDEX_DIR) -> list[Document]:
    """Run semantic search against the persisted FAISS index."""
    vector_store = load_vector_store(index_dir)
    return vector_store.similarity_search(query, k=k)


