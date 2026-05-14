
# app.py
import sys
import os
import inspect
from dataclasses import dataclass
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.backend.llm import get_llm
from src.backend.prompt_builder import PromptBuilder
from src.backend.rag_service import RagService
from src.backend.corag_service import CoRAGService
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
    Quản lý index + cung cấp cả RAG và CoRAG trên cùng 1 vectorstore.

    RAG   : retrieve 1 lần → generate  (nhanh hơn)
    CoRAG : decompose → retrieve nhiều lần → synthesize  (chính xác hơn)
    """

    def __init__(self):
        self.vectorstore  = None
        self.all_chunks   = []
        self.loaded_files = {}   # { original_name: [chunk, ...] }
        self.conversation_memory = []

        self.llm            = get_llm()
        self.prompt_builder = PromptBuilder()

        # Hai service chạy song song
        self.rag_service   = None
        self.corag_service = None

    # ------------------------------------------------------------------ #
    #  Thêm file vào index (merge)                                         #
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
            chunk.metadata["source"] = original_name
            chunk.metadata["source_path"] = resolved_file_path
            chunk.metadata["chunk_size"] = chunk_size
            chunk.metadata["chunk_overlap"] = chunk_overlap
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = (
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

    def _rebuild_services(self):
        """Rebuild cả RAG lẫn CoRAG từ cùng 1 vectorstore."""
        if self.vectorstore is None or not self.all_chunks:
            self.rag_service   = None
            self.corag_service = None
            return

        semantic = SemanticRetriever(vector_store=self.vectorstore, k=4)
        keyword  = KeywordRetriever(documents=self.all_chunks, k=3)
        hybrid   = HybridRetriever(semantic, keyword, alpha=0.5, top_k=4)

        # RAG dùng RagService gốc của backend
        self.rag_service = RagService(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            retrieve=hybrid,
            k=4,
        )

        # CoRAG dùng CoRAGService mới — cùng LLM, cùng retriever
        self.corag_service = CoRAGService(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            retrieve=hybrid,
            k=4,
        )

    # ------------------------------------------------------------------ #
    #  Load từ disk                                                        #
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
        except Exception as e:
            print(f"Không thể load index từ disk: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Xoá file khỏi index                                                 #
    # ------------------------------------------------------------------ #
    def remove_document(self, original_name: str) -> bool:
        if original_name not in self.loaded_files:
            return False
        del self.loaded_files[original_name]
        self.all_chunks = [c for cs in self.loaded_files.values() for c in cs]
        if not self.all_chunks:
            self.vectorstore   = None
            self.rag_service   = None
            self.corag_service = None
            delete_vector_store()
            return True
        self.vectorstore = build_vector_store(self.all_chunks)
        save_vector_store(self.vectorstore)
        self._rebuild_services()
        return True

    # ------------------------------------------------------------------ #
    #  RAG: retrieve 1 lần → generate                                     #
    # ------------------------------------------------------------------ #
    def ask_rag(
        self,
        question: str,
        filter: DocumentFilter | None = None,
        save_history: bool = True,
        use_conversation: bool = False,
    ) -> dict:
        """
        RAG thuần: retrieve top-k chunks → generate.
        Trả về: { question, answer, sources, meta }
        """
        if self.rag_service is None:
            return _not_ready(question)

        answer_params = inspect.signature(self.rag_service.answer).parameters
        if use_conversation and "chat_history" not in answer_params:
            self._rebuild_services()

        if use_conversation and "chat_history" in inspect.signature(self.rag_service.answer).parameters:
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
    #  CoRAG: decompose → multi-retrieve → synthesize                     #
    # ------------------------------------------------------------------ #
    def ask_corag(
        self,
        question: str,
        filter: DocumentFilter | None = None,
        save_history: bool = True,
        use_conversation: bool = False,
    ) -> dict:
        """
        CoRAG: phân tách câu hỏi → retrieve nhiều lần → tổng hợp.
        Trả về: { question, answer, sources, sub_questions, meta }
        """
        if self.corag_service is None:
            return _not_ready(question)

        answer_params = inspect.signature(self.corag_service.answer).parameters
        if use_conversation and "chat_history" not in answer_params:
            self._rebuild_services()

        if use_conversation and "chat_history" in inspect.signature(self.corag_service.answer).parameters:
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
    #  Tiện ích                                                            #
    # ------------------------------------------------------------------ #
    def get_loaded_files(self) -> list[str]:
        return list(self.loaded_files.keys())

    def get_history(self) -> list:
        return get_all_history()

    def add_conversation_turn(self, question: str, answer: str) -> None:
        question = (question or "").strip()
        answer = (answer or "").strip()
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
        """Clear in-memory and persisted vector index."""
        self.vectorstore = None
        self.all_chunks = []
        self.loaded_files = {}
        self.conversation_memory = []
        self.rag_service = None
        self.corag_service = None
        delete_vector_store()

        if delete_uploaded_files:
            DATA_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            for path in DATA_UPLOADS_DIR.rglob("*"):
                if path.is_file():
                    path.unlink()


# ------------------------------------------------------------------ #
#  Helpers nội bộ                                                      #
# ------------------------------------------------------------------ #
def _not_ready(question: str) -> dict:
    return {
        "question":      question,
        "answer":        "Chưa nạp tài liệu. Hãy upload file trước.",
        "sources":       [],
        "sub_questions": [],
        "meta":          {},
    }

def _save(result: dict):
    try:
        add_entry(
            question=result["question"],
            ans=result["answer"],
            sources=result.get("sources", []),
            meta=result.get("meta", {}),
        )
    except Exception as e:
        print(f"Warning: không lưu được history: {e}")


# ------------------------------------------------------------------ #
#  Chạy thử từ terminal                                                #
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
    print("RAG")
    print("=" * 60)
    r1 = rag.ask_rag(question, save_history=False)
    print(f"Trả lời: {r1['answer']}")
    print(f"Latency: {r1['meta'].get('latency')}s")

    print("\n" + "=" * 60)
    print("CoRAG")
    print("=" * 60)
    r2 = rag.ask_corag(question, save_history=False)
    print(f"Sub-questions: {r2.get('sub_questions', [])}")
    print(f"Trả lời: {r2['answer']}")
    print(f"Latency: {r2['meta'].get('latency')}s")
