# app.py
# Người 1 – AI Engineer: RAG Chain core
# Tích hợp với backend của Người 2 (src/backend/)

import sys
import os
from dataclasses import dataclass, field

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
from src.backend.history_store import add_entry, get_all_history, clear  # Người 2: lịch sử hội thoại
from src.processor import load_document, split_text


# ------------------------------------------------------------------ #
#  DocumentFilter — dùng để lọc tài liệu khi hỏi đáp                  #
#  (tương đương DocumentFilter trong src/api/main.py của Người 2)     #
# ------------------------------------------------------------------ #
@dataclass
class DocumentFilter:
    """
    Bộ lọc metadata khi tìm kiếm trong FAISS.
    Người 3 (Streamlit) có thể truyền filter này vào ask_with_sources()
    để giới hạn tìm kiếm chỉ trong 1 file cụ thể.

    Ví dụ:
        f = DocumentFilter(sources=["gutenberg.pdf"])
        result = rag.ask_with_sources("Tóm tắt nội dung?", filter=f)
    """
    sources: list[str] | None = None       # lọc theo tên file, vd: ["gutenberg.pdf"]
    file_type: str | None = None           # lọc theo loại: "pdf" hoặc "docx"
    date_from: str | None = None           # lọc theo ngày upload từ (ISO string)
    date_to: str | None = None             # lọc theo ngày upload đến (ISO string)


# ------------------------------------------------------------------ #
#  RAGChain — lớp trung tâm điều phối toàn bộ pipeline RAG            #
# ------------------------------------------------------------------ #
class RAGChain:
    """
    Điều phối pipeline RAG hoàn chỉnh, tích hợp backend của Người 2:
      - HybridRetriever  : FAISS semantic + BM25 keyword (RRF scoring)
      - RagService       : sinh câu trả lời + trả về sources + latency
      - history_store    : lưu mỗi Q&A vào data/chat_history.json
      - DocumentFilter   : lọc tìm kiếm theo metadata (tên file, loại, ngày)

    Sơ đồ luồng dữ liệu:
      file_path
        → load_document()        # Người 4: đọc PDF/DOCX → list[Document]
        → split_text()           # Người 4: chia chunks có metadata
        → build_vector_store()   # Người 2: tạo FAISS index
        → save_vector_store()    # Người 2: lưu xuống src/faiss_index/
        → HybridRetriever        # Người 2: semantic + keyword + filter
        → RagService.answer()    # Người 2: sinh câu trả lời
        → add_entry()            # Người 2: lưu Q&A vào chat_history.json
    """

    def __init__(self):
        self.vectorstore = None     # FAISS index
        self.rag_service = None     # RagService của Người 2
        self.chunks = []            # giữ lại để KeywordRetriever index BM25

        # Khởi tạo LLM và PromptBuilder từ backend Người 2
        self.llm = get_llm()                    # ChatOllama(qwen2.5:1.5b, temperature=0.1)
        self.prompt_builder = PromptBuilder()   # tự động detect ngôn ngữ Anh/Việt

    # ------------------------------------------------------------------ #
    #  BƯỚC 1: Nhận file path → tạo FAISS vectorstore                     #
    # ------------------------------------------------------------------ #
    def build_vectorstore(self, file_path: str) -> int:
        """
        Load document → split chunks (có metadata) → tạo FAISS → lưu disk.

        Metadata mỗi chunk:
            { "source": "file.pdf", "file_type": "pdf", "date_uploaded": "..." }
        Lưu ý: file_type KHÔNG có dấu chấm đầu (Người 4 đã cập nhật).

        Trả về số chunks đã được index.
        """
        # Bước 1a: Load file → list[Document] có metadata đầy đủ
        docs = load_document(file_path)

        # Bước 1b: Chia nhỏ thành chunks → list[Document]
        self.chunks = split_text(docs)

        # Bước 1c: Tạo FAISS index (hàm của Người 2, tự gọi get_embedding_model)
        self.vectorstore = build_vector_store(self.chunks)

        # Bước 1d: Lưu xuống src/faiss_index/ để tái sử dụng khi khởi động lại
        save_vector_store(self.vectorstore)

        return len(self.chunks)

    # ------------------------------------------------------------------ #
    #  BƯỚC 2: Tạo RagService + HybridRetriever                           #
    # ------------------------------------------------------------------ #
    def build_chain(self):
        """
        Khởi tạo HybridRetriever và RagService.

        HybridRetriever (Người 2) dùng RRF scoring:
          - SemanticRetriever: FAISS similarity (tìm theo nghĩa)
          - KeywordRetriever:  BM25 (tìm theo từ khóa chính xác)
          → Kết quả tốt hơn dùng đơn lẻ, đặc biệt với tiếng Việt

        RagService (Người 2) có thêm:
          - Giới hạn context (3500 ký tự tổng, 1200/doc)
          - Dedup chunks trùng lặp
          - Fallback: nếu LLM lỗi → thử lại với ít context hơn
          - Nhận filter để lọc theo metadata
        """
        if self.vectorstore is None:
            raise ValueError("Gọi build_vectorstore() trước!")

        semantic_retriever = SemanticRetriever(
            vector_store=self.vectorstore,
            k=4,
        )

        keyword_retriever = KeywordRetriever(
            documents=self.chunks,
            k=3,
        )

        # alpha=0.5: cân bằng 50% semantic + 50% keyword
        hybrid_retriever = HybridRetriever(
            semantic_retriever=semantic_retriever,
            keyword_retriever=keyword_retriever,
            alpha=0.5,
            top_k=4,
        )

        self.rag_service = RagService(
            llm=self.llm,
            prompt_builder=self.prompt_builder,
            retrieve=hybrid_retriever,
            k=4,
        )

    # ------------------------------------------------------------------ #
    #  BƯỚC 3a: Hỏi đáp — trả về string đơn giản                         #
    # ------------------------------------------------------------------ #
    def ask(self, question: str, filter: DocumentFilter | None = None) -> str:
        """
        Gửi câu hỏi → nhận câu trả lời dạng string.
        Tự động lưu Q&A vào data/chat_history.json.

        Args:
            question: câu hỏi của người dùng
            filter:   DocumentFilter để giới hạn tìm kiếm (tuỳ chọn)
        """
        result = self.ask_with_sources(question, filter=filter)
        return result["answer"]

    # ------------------------------------------------------------------ #
    #  BƯỚC 3b: Hỏi đáp — trả về đầy đủ dict (answer + sources + meta)   #
    # ------------------------------------------------------------------ #
    def ask_with_sources(
        self,
        question: str,
        filter: DocumentFilter | None = None,
    ) -> dict:
        """
        Gửi câu hỏi → nhận dict đầy đủ:
          {
            "question": "...",
            "answer":   "...",
            "sources":  [{ index, source, page, content }, ...],
            "meta":     { k, latency, model }
          }

        Tự động lưu vào data/chat_history.json sau mỗi lần hỏi.

        Args:
            question: câu hỏi của người dùng
            filter:   DocumentFilter để lọc tài liệu (tuỳ chọn)
                      Ví dụ: DocumentFilter(sources=["bao_cao.pdf"])
        """
        if self.rag_service is None:
            return {
                "question": question,
                "answer": "Chưa nạp tài liệu. Hãy upload file trước.",
                "sources": [],
                "meta": {},
            }

        # Gọi RagService (Người 2) — filter được truyền xuống HybridRetriever
        result = self.rag_service.answer(question, filter=filter)

        # Lưu vào data/chat_history.json (Người 2: history_store)
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

    # ------------------------------------------------------------------ #
    #  Quản lý lịch sử hội thoại                                          #
    # ------------------------------------------------------------------ #
    def get_history(self) -> list:
        """
        Lấy toàn bộ lịch sử Q&A từ data/chat_history.json.
        Người 3 (Streamlit) dùng để hiển thị trong sidebar.

        Mỗi entry:
            { id, question, answer, sources, meta, timestamp }
        """
        return get_all_history()

    def clear_history(self) -> None:
        """
        Xoá toàn bộ lịch sử trong data/chat_history.json.
        Gọi khi người dùng bấm nút 'Xoá chat'.
        """
        clear()

    # ------------------------------------------------------------------ #
    #  Hàm tiện ích: gộp bước 1 + 2                                       #
    # ------------------------------------------------------------------ #
    def load_and_build(self, file_path: str) -> int:
        """
        Gộp build_vectorstore() + build_chain() lại thành 1 lệnh.
        Người 3 (Streamlit) gọi hàm này sau khi user upload file.
        Trả về số chunks để hiển thị thông báo trên UI.
        """
        num_chunks = self.build_vectorstore(file_path)
        self.build_chain()
        return num_chunks

    def load_from_disk_and_build(self) -> bool:
        """
        Tải FAISS index đã lưu từ src/faiss_index/ (không cần upload lại).
        Dùng khi khởi động lại app mà index vẫn còn trên disk.

        Trả về True nếu load thành công, False nếu chưa có index.
        """
        if not INDEX_DIR.exists():
            return False

        try:
            self.vectorstore = load_vector_store()
            # Lấy Document objects từ docstore để khởi tạo KeywordRetriever
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
    print(f"Đã index {num} chunks. Đang hỏi...\n")

    # Hỏi không filter
    result = rag.ask_with_sources(question)
    print(f"Câu trả lời:\n{result['answer']}")

    if result["sources"]:
        print(f"\nNguồn trích dẫn:")
        for src in result["sources"]:
            print(f"  [{src['index']}] {src['source']} — trang {src['page']}")

    print(f"\nLatency : {result['meta'].get('latency')}s")
    print(f"Model   : {result['meta'].get('model')}")

    # Thử filter — chỉ tìm trong file vừa upload
    file_name = Path(file_path).name
    print(f"\n--- Hỏi lại với filter chỉ file '{file_name}' ---")
    f = DocumentFilter(sources=[file_name])
    result2 = rag.ask_with_sources(question, filter=f)
    print(f"Câu trả lời:\n{result2['answer']}")

    # Xem lịch sử
    history = rag.get_history()
    print(f"\nLịch sử: {len(history)} câu hỏi đã lưu.")