# app.py
# Người 1 – AI Engineer: RAG Chain core
# Tích hợp với backend của Người 2 (src/backend/)

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path

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
from src.processor import load_document, split_text, get_embedding_model


class RAGChain:
    """
    Lớp trung tâm điều phối toàn bộ pipeline RAG.
    Tích hợp với backend của Người 2:
      - HybridRetriever (FAISS semantic + BM25 keyword)
      - RagService (xử lý câu hỏi, trả về answer + sources + latency)
      - FAISS lưu xuống disk (src/faiss_index/) để tái sử dụng

    Sơ đồ luồng dữ liệu:
      file_path
        → load_document()           # Người 4: đọc PDF/DOCX
        → split_text()              # Người 4: chia chunks có metadata
        → build_vector_store()      # Người 2: tạo FAISS index
        → save_vector_store()       # Người 2: lưu xuống disk
        → HybridRetriever           # Người 2: kết hợp semantic + keyword
        → RagService.answer()       # Người 2: sinh câu trả lời
    """

    def __init__(self):
        self.vectorstore = None     # FAISS index
        self.rag_service = None     # RagService của Người 2
        self.chunks = []            # giữ lại chunks để khởi tạo KeywordRetriever

        # Khởi tạo LLM và PromptBuilder từ backend Người 2
        self.llm = get_llm()                    # ChatOllama(qwen2.5:1.5b)
        self.prompt_builder = PromptBuilder()   # prompt tự động detect ngôn ngữ

    # ------------------------------------------------------------------ #
    #  BƯỚC 1: Nhận file path → tạo FAISS vectorstore                     #
    # ------------------------------------------------------------------ #
    def build_vectorstore(self, file_path: str) -> int:
        """
        Load document → split chunks → tạo FAISS index → lưu xuống disk.

        Khác với version cũ (in-memory), giờ FAISS được lưu vào
        src/faiss_index/ để có thể tải lại mà không cần xử lý lại file.

        Trả về số lượng chunks đã được index.
        """
        # Bước 1a: Load file → list[Document] có metadata đầy đủ
        docs = load_document(file_path)

        # Bước 1b: Chia nhỏ thành chunks → list[Document]
        # Metadata mỗi chunk: { source, file_type, date_uploaded }
        self.chunks = split_text(docs)

        # Bước 1c: Tạo FAISS index (dùng hàm của Người 2)
        # build_vector_store() tự gọi get_embedding_model() bên trong
        self.vectorstore = build_vector_store(self.chunks)

        # Bước 1d: Lưu xuống disk → src/faiss_index/
        # Lần sau load lại không cần xử lý file nữa
        save_vector_store(self.vectorstore)

        return len(self.chunks)

    # ------------------------------------------------------------------ #
    #  BƯỚC 2: Tạo RagService từ vectorstore + Ollama                     #
    # ------------------------------------------------------------------ #
    def build_chain(self):
        """
        Khởi tạo HybridRetriever và RagService từ backend của Người 2.

        HybridRetriever kết hợp:
          - SemanticRetriever: FAISS cosine similarity (tìm theo nghĩa)
          - KeywordRetriever:  BM25 (tìm theo từ khóa chính xác)
          → Kết quả tốt hơn dùng 1 retriever đơn lẻ
        """
        if self.vectorstore is None:
            raise ValueError("Gọi build_vectorstore() trước!")

        # --- Semantic retriever: dùng FAISS ---
        semantic_retriever = SemanticRetriever(
            vector_store=self.vectorstore,
            k=3,
        )

        # --- Keyword retriever: dùng BM25 ---
        # Cần truyền vào list[Document] (chunks) để BM25 index
        keyword_retriever = KeywordRetriever(
            documents=self.chunks,
            k=3,
        )

        # --- Hybrid retriever: kết hợp cả 2 ---
        # alpha=0.5: cân bằng 50% semantic + 50% keyword
        hybrid_retriever = HybridRetriever(
            semantic_retriever=semantic_retriever,
            keyword_retriever=keyword_retriever,
            alpha=0.5,
        )

        # --- RagService: xử lý toàn bộ pipeline hỏi-đáp ---
        # Trả về dict gồm: answer, sources (citation), meta (latency, model)
        self.rag_service = RagService(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            retrieve=hybrid_retriever,
            k=3,
        )

    # ------------------------------------------------------------------ #
    #  BƯỚC 3: Nhận câu hỏi → trả về câu trả lời                         #
    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> str:
        """
        Gửi câu hỏi vào RagService, nhận câu trả lời dạng string.

        RagService.answer() trả về dict:
          {
            "question": "...",
            "answer":   "...",       ← đây là phần ta cần
            "sources":  [...],       ← citation (tuần 3 Người 3 sẽ hiển thị)
            "meta":     { latency, model, k }
          }
        """
        if self.rag_service is None:
            return "Chưa nạp tài liệu. Hãy upload file trước."

        result = self.rag_service.answer(question)
        return result["answer"]

    def ask_with_sources(self, question: str) -> dict:
        """
        Giống ask() nhưng trả về toàn bộ dict (answer + sources + meta).
        Người 3 (Streamlit) dùng hàm này để hiển thị citation.
        """
        if self.rag_service is None:
            return {"answer": "Chưa nạp tài liệu.", "sources": [], "meta": {}}

        return self.rag_service.answer(question)

    # ------------------------------------------------------------------ #
    #  Hàm tiện ích: chạy cả pipeline trong 1 lệnh                        #
    # ------------------------------------------------------------------ #
    def load_and_build(self, file_path: str) -> int:
        """
        Gộp bước 1 + 2 lại để Người 3 (Streamlit) tiện gọi.
        Trả về số chunks để hiển thị thông báo trên giao diện.
        """
        num_chunks = self.build_vectorstore(file_path)
        self.build_chain()
        return num_chunks

    def load_from_disk_and_build(self) -> bool:
        """
        Tải FAISS index đã lưu từ disk (không cần upload lại file).
        Dùng khi khởi động lại app mà index vẫn còn.
        Trả về True nếu load thành công, False nếu chưa có index.
        """
        if not INDEX_DIR.exists():
            return False

        try:
            self.vectorstore = load_vector_store()
            # Lấy documents từ docstore để khởi tạo KeywordRetriever
            self.chunks = list(self.vectorstore.docstore._dict.values())
            self.build_chain()
            return True
        except Exception as e:
            print(f"Không thể load index từ disk: {e}")
            return False


# ------------------------------------------------------------------ #
#  Chạy thử nhanh từ terminal (không cần Streamlit)                   #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python app.py <đường_dẫn_file> <câu_hỏi>")
        print('Ví dụ:     python app.py data/report/gutenberg.pdf "Tài liệu này nói về gì?"')
        sys.exit(1)

    file_path = sys.argv[1]
    question  = sys.argv[2]

    print(f"Đang nạp file: {file_path}")
    rag = RAGChain()
    num = rag.load_and_build(file_path)
    print(f"Đã index {num} chunks. Đang hỏi...")

    result = rag.ask_with_sources(question)
    print(f"\nCâu trả lời:\n{result['answer']}")
    print(f"\nNguồn trích dẫn:")
    for src in result["sources"]:
        print(f"  [{src['index']}] {src['source']} - trang {src['page']}")
    print(f"\nLatency: {result['meta'].get('latency')}s | Model: {result['meta'].get('model')}")