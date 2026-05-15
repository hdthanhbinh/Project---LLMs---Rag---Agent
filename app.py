# app.py
import sys
import os
import inspect
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.backend.llm import get_llm
from src.backend.prompt_builder import PromptBuilder
from src.backend.rag_service import RagService
from src.backend.corag_service import CoRAGService
from src.backend.self_rag_service import SelfRAGService          # ← NEW
from src.backend.retriever import HybridRetriever, KeywordRetriever, SemanticRetriever
from src.backend.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    INDEX_DIR,
    delete_vector_store,
    index_exists,
)
from src.backend.history_store import add_entry, get_all_history, clear
from src.processor import load_document, split_text, get_embedding_model


ROOT_DIR = Path(__file__).resolve().parent
DATA_UPLOADS_DIR = ROOT_DIR / "data" / "uploads"


@dataclass
class DocumentFilter:
    sources:   list[str] | None = None
    file_type: str | None       = None
    date_from: str | None       = None
    date_to:   str | None       = None


class RAGChain:
    """
    Manages the vector index and exposes RAG, CoRAG, and Self-RAG over the
    same vectorstore.  Re-ranking is an optional flag that can be toggled at
    query time without rebuilding the index.

    RAG      : retrieve once → generate                      (fastest)
    CoRAG    : decompose → multi-retrieve → synthesize        (thorough)
    Self-RAG : retrieve → generate → self-evaluate → retry   (accurate)
    """

    def __init__(self):
        self.vectorstore  = None
        self.all_chunks   = []
        self.loaded_files: dict[str, list] = {}
        self.conversation_memory: list[dict] = []

        self.llm            = get_llm()
        self.prompt_builder = PromptBuilder()

        self.rag_service:      RagService      | None = None
        self.corag_service:    CoRAGService    | None = None
        self.self_rag_service: SelfRAGService  | None = None   # ← NEW

        # Persists across ask_* calls so the UI toggle is sticky
        self._enable_rerank: bool = False

    # ------------------------------------------------------------------ #
    #  Re-rank toggle                                                      #
    # ------------------------------------------------------------------ #

    def set_rerank(self, enabled: bool) -> None:
        """Enable or disable cross-encoder re-ranking.  Rebuilds services."""
        if enabled != self._enable_rerank:
            self._enable_rerank = enabled
            self._rebuild_services()

    def is_rerank_enabled(self) -> bool:
        return self._enable_rerank

    # ------------------------------------------------------------------ #
    #  Add file to index (merge)                                           #
    # ------------------------------------------------------------------ #

    def add_document(
        self,
        file_path: str,
        original_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> int:
        docs   = load_document(file_path)
        chunks = split_text(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        resolved_file_path = str(Path(file_path).resolve())
        for idx, chunk in enumerate(chunks, 1):
            chunk.metadata["source"]        = original_name
            chunk.metadata["source_path"]   = resolved_file_path
            chunk.metadata["chunk_size"]    = chunk_size
            chunk.metadata["chunk_overlap"] = chunk_overlap
            chunk.metadata["chunk_index"]   = idx
            chunk.metadata["chunk_id"]      = (
                f"{original_name}:{chunk.metadata.get('page', 'na')}:{idx}"
            )

        self.loaded_files[original_name] = chunks
        self.all_chunks = [c for cs in self.loaded_files.values() for c in cs]

        embedding = get_embedding_model()

        if self.vectorstore is None:
            self.vectorstore = build_vector_store(chunks)
        else:
            from langchain_community.vectorstores import FAISS
            new_vs = FAISS.from_documents(chunks, embedding)
            self.vectorstore.merge_from(new_vs)

        save_vector_store(self.vectorstore)
        self._rebuild_services()
        return len(chunks)

    # ------------------------------------------------------------------ #
    #  Internal: rebuild all three services                                #
    # ------------------------------------------------------------------ #

    def _rebuild_services(self) -> None:
        """Rebuild RAG + CoRAG + Self-RAG from current vectorstore and rerank flag."""
        if self.vectorstore is None or not self.all_chunks:
            self.rag_service      = None
            self.corag_service    = None
            self.self_rag_service = None
            return

        semantic = SemanticRetriever(vector_store=self.vectorstore, k=4)
        keyword  = KeywordRetriever(documents=self.all_chunks, k=3)
        hybrid   = HybridRetriever(
            semantic,
            keyword,
            alpha=0.5,
            top_k=4,
            enable_rerank=self._enable_rerank,
        )

        self.rag_service = RagService(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            retrieve=hybrid,
            k=4,
        )
        self.corag_service = CoRAGService(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            retrieve=hybrid,
            k=4,
        )
        self.self_rag_service = SelfRAGService(          # ← NEW
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            retrieve=hybrid,
            k=4,
        )

    # ------------------------------------------------------------------ #
    #  Load from disk                                                      #
    # ------------------------------------------------------------------ #

    def load_from_disk_and_build(self) -> bool:
        if not index_exists(INDEX_DIR):
            return False
        try:
            self.vectorstore = load_vector_store()
            self.all_chunks  = list(self.vectorstore.docstore._dict.values())
            for chunk in self.all_chunks:
                src = chunk.metadata.get("source", "unknown")
                self.loaded_files.setdefault(src, []).append(chunk)
            self._rebuild_services()
            return True
        except Exception as exc:
            print(f"Không thể load index từ disk: {exc}")
            return False

    # ------------------------------------------------------------------ #
    #  Remove file from index                                              #
    # ------------------------------------------------------------------ #

    def remove_document(self, original_name: str) -> bool:
        if original_name not in self.loaded_files:
            return False
        del self.loaded_files[original_name]
        self.all_chunks = [c for cs in self.loaded_files.values() for c in cs]
        if not self.all_chunks:
            self.vectorstore      = None
            self.rag_service      = None
            self.corag_service    = None
            self.self_rag_service = None
            delete_vector_store()
            return True
        self.vectorstore = build_vector_store(self.all_chunks)
        save_vector_store(self.vectorstore)
        self._rebuild_services()
        return True

    # ------------------------------------------------------------------ #
    #  RAG query                                                           #
    # ------------------------------------------------------------------ #

    def ask_rag(
        self,
        question: str,
        filter: DocumentFilter | None = None,
        save_history: bool = True,
        use_conversation: bool = False,
        enable_rerank: bool | None = None,
    ) -> dict:
        if enable_rerank is not None:
            self.set_rerank(enable_rerank)

        if self.rag_service is None:
            return _not_ready(question)

        answer_params = inspect.signature(self.rag_service.answer).parameters
        if use_conversation and "chat_history" not in answer_params:
            self._rebuild_services()

        if (
            use_conversation
            and "chat_history" in inspect.signature(self.rag_service.answer).parameters
        ):
            result = self.rag_service.answer(
                question,
                filter=filter,
                chat_history=self.conversation_memory,
                rewrite_query=True,
            )
        else:
            result = self.rag_service.answer(question, filter=filter)

        result["meta"]["method"] = "rag"

        if save_history:
            _save(result)
        return result

    # ------------------------------------------------------------------ #
    #  CoRAG query                                                         #
    # ------------------------------------------------------------------ #

    def ask_corag(
        self,
        question: str,
        filter: DocumentFilter | None = None,
        save_history: bool = True,
        use_conversation: bool = False,
        enable_rerank: bool | None = None,
    ) -> dict:
        if enable_rerank is not None:
            self.set_rerank(enable_rerank)

        if self.corag_service is None:
            return _not_ready(question)

        answer_params = inspect.signature(self.corag_service.answer).parameters
        if use_conversation and "chat_history" not in answer_params:
            self._rebuild_services()

        if (
            use_conversation
            and "chat_history" in inspect.signature(self.corag_service.answer).parameters
        ):
            result = self.corag_service.answer(
                question,
                filter=filter,
                chat_history=self.conversation_memory,
                rewrite_query=True,
            )
        else:
            result = self.corag_service.answer(question, filter=filter)

        if save_history:
            _save(result)
        return result

    # ------------------------------------------------------------------ #
    #  Self-RAG query                                ← NEW               #
    # ------------------------------------------------------------------ #

    def ask_self_rag(
        self,
        question: str,
        filter: DocumentFilter | None = None,
        save_history: bool = True,
        use_conversation: bool = False,
        enable_rerank: bool | None = None,
    ) -> dict:
        """
        Self-RAG: retrieve → generate → self-evaluate → optional retry.

        Returns: { question, rewritten_question, answer, sources,
                   confidence, self_eval, meta }

        meta includes:
          method, latency, retried, fallback,
          rerank_enabled, rerank_model, rerank_latency, candidate_count
        """
        if enable_rerank is not None:
            self.set_rerank(enable_rerank)

        if self.self_rag_service is None:
            return _not_ready(question)

        answer_params = inspect.signature(self.self_rag_service.answer).parameters
        if use_conversation and "chat_history" not in answer_params:
            self._rebuild_services()

        if (
            use_conversation
            and "chat_history" in inspect.signature(self.self_rag_service.answer).parameters
        ):
            result = self.self_rag_service.answer(
                question,
                filter=filter,
                chat_history=self.conversation_memory,
                rewrite_query=True,
            )
        else:
            result = self.self_rag_service.answer(question, filter=filter)

        if save_history:
            _save(result)
        return result

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def get_loaded_files(self) -> list[str]:
        return list(self.loaded_files.keys())

    def get_history(self) -> list:
        return get_all_history()

    def add_conversation_turn(self, question: str, answer: str) -> None:
        question = (question or "").strip()
        answer   = (answer   or "").strip()
        if not question:
            return
        self.conversation_memory.append({"question": question, "answer": answer})
        self.conversation_memory = self.conversation_memory[-8:]

    def get_conversation_memory(self) -> list[dict]:
        return list(self.conversation_memory)

    def clear_conversation_memory(self) -> None:
        self.conversation_memory = []

    def clear_history(self) -> None:
        self.clear_conversation_memory()
        clear()

    def clear_index(self, delete_uploaded_files: bool = False) -> None:
        self.vectorstore      = None
        self.all_chunks       = []
        self.loaded_files     = {}
        self.conversation_memory = []
        self.rag_service      = None
        self.corag_service    = None
        self.self_rag_service = None
        delete_vector_store()

        if delete_uploaded_files:
            DATA_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            for path in DATA_UPLOADS_DIR.rglob("*"):
                if path.is_file():
                    path.unlink()


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _not_ready(question: str) -> dict:
    return {
        "question":           question,
        "rewritten_question": question,
        "answer":             "Chưa nạp tài liệu. Hãy upload file trước.",
        "sources":            [],
        "sub_questions":      [],
        "confidence":         None,
        "self_eval":          None,
        "meta":               {},
    }


def _save(result: dict) -> None:
    try:
        add_entry(
            question=result["question"],
            ans=result["answer"],
            sources=result.get("sources", []),
            meta=result.get("meta", {}),
        )
    except Exception as exc:
        print(f"Warning: không lưu được history: {exc}")


# ------------------------------------------------------------------ #
#  CLI smoke-test                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Dùng: python app.py <file> <câu_hỏi>")
        sys.exit(1)

    file_path     = sys.argv[1]
    question      = sys.argv[2]
    original_name = Path(file_path).name

    rag = RAGChain()
    n   = rag.add_document(file_path, original_name)
    print(f"Đã index {n} chunks từ '{original_name}'\n")

    print("=" * 60)
    print("RAG (no rerank)")
    print("=" * 60)
    r1 = rag.ask_rag(question, save_history=False, enable_rerank=False)
    print(f"Trả lời: {r1['answer']}")
    print(f"Meta: {r1['meta']}")

    print("\n" + "=" * 60)
    print("CoRAG (with rerank)")
    print("=" * 60)
    r2 = rag.ask_corag(question, save_history=False, enable_rerank=True)
    print(f"Sub-questions: {r2.get('sub_questions', [])}")
    print(f"Trả lời: {r2['answer']}")
    print(f"Meta: {r2['meta']}")

    print("\n" + "=" * 60)
    print("Self-RAG")
    print("=" * 60)
    r3 = rag.ask_self_rag(question, save_history=False)
    print(f"Rewritten: {r3.get('rewritten_question')}")
    print(f"Confidence: {r3.get('confidence')}")
    print(f"Self-eval: {r3.get('self_eval')}")
    print(f"Trả lời: {r3['answer']}")
    print(f"Meta: {r3['meta']}")