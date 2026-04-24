
# app.py
import sys
import os
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
)
from src.backend.history_store import add_entry, get_all_history, clear
from src.processor import load_document, split_text, get_embedding_model


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

        self.llm            = get_llm()
        self.prompt_builder = PromptBuilder()

        # Hai service chạy song song
        self.rag_service   = None
        self.corag_service = None

    # ------------------------------------------------------------------ #
    #  Thêm file vào index (merge)                                         #
    # ------------------------------------------------------------------ #
    def add_document(self, file_path: str, original_name: str) -> int:
        docs   = load_document(file_path)
        chunks = split_text(docs)

        for chunk in chunks:
            chunk.metadata["source"] = original_name

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
        if not INDEX_DIR.exists():
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
    ) -> dict:
        """
        RAG thuần: retrieve top-k chunks → generate.
        Trả về: { question, answer, sources, meta }
        """
        if self.rag_service is None:
            return _not_ready(question)

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
    ) -> dict:
        """
        CoRAG: phân tách câu hỏi → retrieve nhiều lần → tổng hợp.
        Trả về: { question, answer, sources, sub_questions, meta }
        """
        if self.corag_service is None:
            return _not_ready(question)

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

    def clear_history(self) -> None:
        clear()


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
