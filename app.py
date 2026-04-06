# app.py
# Người 1 – AI Engineer: RAG Chain core

import sys
import os

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# Dùng 3 hàm riêng lẻ từ processor.py của Người 4
# (thay vì process_pipeline() để lấy được Document objects có metadata)
from processor import load_document, split_text, get_embedding_model


class RAGChain:
    """
    Lớp trung tâm điều phối toàn bộ pipeline RAG.
    Gồm 3 bước chính: build_vectorstore → build_chain → ask

    Sơ đồ luồng dữ liệu:
      file_path
        → load_document()       # Người 4: đọc PDF/DOCX thành Document objects
        → split_text()          # Người 4: chia nhỏ thành chunks có metadata
        → FAISS.from_documents()# Bước này: embed + lưu vào vector index
        → RetrievalQA chain     # Bước này: kết nối retriever + LLM + prompt
        → ask(question)         # Bước này: trả về câu trả lời string
    """

    def __init__(self):
        self.vectorstore = None   # FAISS index (được tạo sau khi upload file)
        self.chain = None         # RetrievalQA chain (được tạo sau khi có vectorstore)

    # ------------------------------------------------------------------ #
    #  BƯỚC 1: Nhận file path → tạo FAISS vectorstore                     #
    # ------------------------------------------------------------------ #
    def build_vectorstore(self, file_path: str) -> int:
        """
        Gọi load_document() và split_text() từ processor.py của Người 4,
        sau đó đưa vào FAISS để tạo vector index.

        Dùng from_documents() thay vì from_texts() để giữ nguyên metadata
        của từng chunk (source, file_type, date_uploaded) — cần thiết cho
        citation tracking ở tuần 3.

        Trả về số lượng chunks đã được index.
        """
        # Bước 1a: Load file → list[Document]
        # Mỗi Document có page_content (nội dung) và metadata (nguồn gốc)
        docs = load_document(file_path)

        # Bước 1b: Chia nhỏ thành chunks → list[Document]
        # Mỗi chunk kế thừa metadata từ Document gốc:
        #   {
        #     "source": "gutenberg.pdf",
        #     "file_type": ".pdf",
        #     "date_uploaded": "2026-04-06T..."
        #   }
        chunks = split_text(docs)

        # Bước 1c: Lấy embedding model (MPNet 768-dim multilingual)
        embedding_model = get_embedding_model()

        # Bước 1d: Tạo FAISS index từ chunks + embedding model
        # from_documents() tự động:
        #   1. Embed từng chunk.page_content thành vector 768 chiều
        #   2. Lưu vector + metadata vào in-memory FAISS index
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )

        return len(chunks)  # trả về để Streamlit (Người 3) hiển thị thông báo

    # ------------------------------------------------------------------ #
    #  BƯỚC 2: Tạo RetrievalQA chain từ vectorstore + Ollama              #
    # ------------------------------------------------------------------ #
    def build_chain(self):
        """
        Kết nối 3 thành phần thành 1 chain hoàn chỉnh:
          FAISS retriever  → tìm top-3 chunks liên quan với câu hỏi
          PromptTemplate   → đóng gói context + câu hỏi thành prompt
          OllamaLLM        → sinh câu trả lời từ prompt
        """
        if self.vectorstore is None:
            raise ValueError("Gọi build_vectorstore() trước!")

        # --- Retriever: tìm top-3 chunks gần nhất với câu hỏi ---
        # search_type="similarity": dùng cosine similarity
        # k=3: lấy 3 chunks liên quan nhất → đủ context, không quá dài
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        # --- Prompt template ---
        # {context}: 3 chunks tìm được (LangChain tự điền)
        # {question}: câu hỏi người dùng (LangChain tự điền)
        # Hướng dẫn model trả lời ngắn gọn và bám sát tài liệu
        prompt_template = """Bạn là trợ lý thông minh. Hãy dùng ngữ cảnh dưới đây \
để trả lời câu hỏi một cách ngắn gọn (3-4 câu).
Nếu thông tin không có trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu."

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # --- OllamaLLM: chạy local, không cần internet ---
        # temperature=0.3: thấp → câu trả lời ổn định, bám sát tài liệu
        # (tài liệu gốc dùng 0.7 nhưng với RAG nên dùng thấp hơn
        #  để tránh model "sáng tạo" ngoài ngữ cảnh)
        llm = OllamaLLM(
            model="qwen2.5:7b",
            temperature=0.3,
        )

        # --- RetrievalQA: chain tích hợp retriever + llm + prompt ---
        # chain_type="stuff": nhét thẳng 3 chunks vào 1 prompt rồi gửi 1 lần
        #   → đơn giản nhất, phù hợp cho tuần 1
        #   → Qwen2.5:7b có context 128K tokens, đủ chứa 3 chunks
        # return_source_documents=False: tuần 3 đổi thành True để làm citation
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
        )

    # ------------------------------------------------------------------ #
    #  BƯỚC 3: Nhận câu hỏi → trả về câu trả lời                         #
    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> str:
        """
        Gửi câu hỏi vào chain, nhận câu trả lời dạng string.
        Chain tự động thực hiện:
          1. Embed câu hỏi thành vector
          2. Tìm top-3 chunks gần nhất trong FAISS
          3. Ghép chunks vào prompt template
          4. Gửi prompt lên Ollama (Qwen2.5:7b)
          5. Trả về câu trả lời
        """
        if self.chain is None:
            return "Chưa nạp tài liệu. Hãy upload file trước."

        # invoke() trả về dict với key "result" chứa câu trả lời string
        response = self.chain.invoke({"query": question})
        return response["result"]

    # ------------------------------------------------------------------ #
    #  Hàm tiện ích: chạy cả pipeline trong 1 lệnh                        #
    # ------------------------------------------------------------------ #
    def load_and_build(self, file_path: str) -> int:
        """
        Gộp bước 1 + 2 lại để Người 3 (Streamlit) tiện gọi.
        Người 3 chỉ cần gọi hàm này sau khi user upload file,
        sau đó gọi ask() để hỏi đáp.
        Trả về số chunks để hiển thị thông báo trên giao diện.
        """
        num_chunks = self.build_vectorstore(file_path)
        self.build_chain()
        return num_chunks


# ------------------------------------------------------------------ #
#  Chạy thử nhanh từ terminal (không cần Streamlit)                   #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python app.py <đường_dẫn_file> <câu_hỏi>")
        print('Ví dụ:     python app.py data/gutenberg.pdf "Tài liệu này nói về gì?"')
        sys.exit(1)

    file_path = sys.argv[1]
    question  = sys.argv[2]

    print(f"Đang nạp file: {file_path}")
    rag = RAGChain()
    num = rag.load_and_build(file_path)
    print(f"Đã index {num} chunks. Đang hỏi...")

    answer = rag.ask(question)
    print(f"\nCâu trả lời:\n{answer}")