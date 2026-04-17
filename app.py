# app.py
import sys
import os
from dataclasses import dataclass
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.backend.llm import get_llm
from src.backend.prompt_builder import PromptBuilder
from src.backend.rag_service import RagService
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
    """
    Bộ lọc metadata khi tìm kiếm trong FAISS.
    sources: list tên file gốc, vd ["Nhom28_(Cau2-3).pdf", "Cau2.docx"]
    """
    sources: list[str] | None = None
    file_type: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class RAGChain:
    """
    Pipeline RAG hỗ trợ multi-document:
      - Mỗi lần upload file mới sẽ MERGE vào index hiện có (không ghi đè)
      - source metadata = tên file GỐC (không phải tên file tạm)
      - DocumentFilter lọc đúng theo tên file gốc
      - Hỗ trợ xoá từng file khỏi index
    """

    def __init__(self):
        self.vectorstore  = None
        self.rag_service  = None
        self.all_chunks   = []      # tất cả chunks của mọi file đã nạp
        self.loaded_files = {}      # { original_name: [chunk, ...] }

        self.llm            = get_llm()
        self.prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------ #
    #  Thêm 1 file vào index (merge, không ghi đè)                         #
    # ------------------------------------------------------------------ #
    def add_document(self, file_path: str, original_name: str) -> int:
        """
        Load file từ file_path, nhưng gán source = original_name (tên gốc).
        Merge vào FAISS index hiện có thay vì tạo mới.

        Args:
            file_path:     đường dẫn file tạm trên disk (vd /tmp/tmpXXX.pdf)
            original_name: tên file gốc người dùng upload (vd "Nhom28_(Cau2-3).pdf")

        Returns:
            số chunks được thêm vào index
        """
        docs   = load_document(file_path)
        chunks = split_text(docs)

        # Ghi đè source = tên file GỐC để filter hoạt động đúng
        for chunk in chunks:
            chunk.metadata["source"] = original_name

        # Lưu vào registry
        self.loaded_files[original_name] = chunks
        self.all_chunks = [c for cs in self.loaded_files.values() for c in cs]

        embedding = get_embedding_model()

        if self.vectorstore is None:
            # Lần đầu: tạo mới
            self.vectorstore = build_vector_store(chunks)
        else:
            # Lần sau: merge file mới vào index hiện có
            from langchain_community.vectorstores import FAISS
            new_vs = FAISS.from_documents(chunks, embedding)
            self.vectorstore.merge_from(new_vs)

        save_vector_store(self.vectorstore)
        self._rebuild_service()

        return len(chunks)

    def _rebuild_service(self):
        """Rebuild HybridRetriever + RagService từ vectorstore + all_chunks hiện tại."""
        if self.vectorstore is None or not self.all_chunks:
            self.rag_service = None
            return

        semantic = SemanticRetriever(vector_store=self.vectorstore, k=4)
        keyword  = KeywordRetriever(documents=self.all_chunks, k=3)
        hybrid   = HybridRetriever(semantic, keyword, alpha=0.5, top_k=4)

        self.rag_service = RagService(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            retrieve=hybrid,
            k=4,
        )

    # ------------------------------------------------------------------ #
    #  Load index từ disk khi khởi động lại app                            #
    # ------------------------------------------------------------------ #
    def load_from_disk_and_build(self) -> bool:
        if not INDEX_DIR.exists():
            return False
        try:
            self.vectorstore = load_vector_store()
            self.all_chunks  = list(self.vectorstore.docstore._dict.values())
            # Khôi phục loaded_files từ metadata source
            for chunk in self.all_chunks:
                src = chunk.metadata.get("source", "unknown")
                self.loaded_files.setdefault(src, []).append(chunk)
            self._rebuild_service()
            return True
        except Exception as e:
            print(f"Không thể load index từ disk: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Xoá 1 file khỏi index                                              #
    # ------------------------------------------------------------------ #
    def remove_document(self, original_name: str) -> bool:
        """
        Xoá 1 file khỏi index. Rebuild lại FAISS từ các file còn lại.
        Trả về True nếu xoá thành công.
        """
        if original_name not in self.loaded_files:
            return False

        del self.loaded_files[original_name]
        self.all_chunks = [c for cs in self.loaded_files.values() for c in cs]

        if not self.all_chunks:
            self.vectorstore = None
            self.rag_service = None
            return True

        self.vectorstore = build_vector_store(self.all_chunks)
        save_vector_store(self.vectorstore)
        self._rebuild_service()
        return True

    # ------------------------------------------------------------------ #
    #  Hỏi đáp                                                            #
    # ------------------------------------------------------------------ #
    def ask_with_sources(
        self,
        question: str,
        filter: DocumentFilter | None = None,
    ) -> dict:
        """
        Trả về dict: { question, answer, sources, meta }
        Tự động lưu vào data/chat_history.json.
        """
        if self.rag_service is None:
            return {
                "question": question,
                "answer": "Chưa nạp tài liệu. Hãy upload file trước.",
                "sources": [],
                "meta": {},
            }

        result = self.rag_service.answer(question, filter=filter)

        try:
            add_entry(
                question=result["question"],
                ans=result["answer"],
                sources=result["sources"],
                meta=result["meta"],
            )
        except Exception as e:
            print(f"Warning: không lưu được history: {e}")

        return result

    def ask(self, question: str, filter: DocumentFilter | None = None) -> str:
        return self.ask_with_sources(question, filter=filter)["answer"]

    # ------------------------------------------------------------------ #
    #  Danh sách file đã nạp                                              #
    # ------------------------------------------------------------------ #
    def get_loaded_files(self) -> list[str]:
        """Trả về list tên file gốc đã được index."""
        return list(self.loaded_files.keys())

    # ------------------------------------------------------------------ #
    #  Quản lý lịch sử                                                    #
    # ------------------------------------------------------------------ #
    def get_history(self) -> list:
        return get_all_history()

    def clear_history(self) -> None:
        clear()


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
    print(f"Đã index {n} chunks từ '{original_name}'")

    result = rag.ask_with_sources(question)
    print(f"\nCâu trả lời:\n{result['answer']}")
    for src in result["sources"]:
        print(f"  [{src['index']}] {src['source']} — trang {src['page']}")