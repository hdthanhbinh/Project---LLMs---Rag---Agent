# app.py
# Người 1 – AI Engineer: RAG Chain core

from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
# Source - https://stackoverflow.com/a/79814778
# Posted by Bazzas
# Retrieved 2026-04-06, License - CC BY-SA 4.0
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from processor import process_pipeline  # lúc này đã tìm được processor.py trong thư mục src

class RAGChain:
    """
    Lớp trung tâm điều phối toàn bộ pipeline RAG.
    Gồm 3 bước chính: build_vectorstore → build_chain → ask
    """

    def __init__(self):
        self.vectorstore = None   # FAISS index (được tạo sau khi upload file)
        self.chain = None         # RetrievalQA chain (được tạo sau khi có vectorstore)

    # ------------------------------------------------------------------ #
    #  BƯỚC 1: Nhận file path → tạo FAISS vectorstore                     #
    # ------------------------------------------------------------------ #
    def build_vectorstore(self, file_path: str) -> int:
        """
        Gọi process_pipeline() của Người 4, lấy chunks + embedding model,
        rồi đưa vào FAISS để tạo vector index.

        Trả về số lượng chunks đã được index.
        """
        # process_pipeline trả về:
        #   {
        #     "chunks": ["đoạn văn 1", "đoạn văn 2", ...],   ← list[str]
        #     "embedding_model": HuggingFaceEmbeddings(...)
        #   }
        result = process_pipeline(file_path)

        chunks = result["chunks"]
        embedding_model = result["embedding_model"]

        # FAISS.from_texts() nhận list[str] + embedding model
        # → tự động embed từng chunk → lưu vào in-memory FAISS index
        self.vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model,
        )

        return len(chunks)  # trả về để Streamlit (Người 3) có thể hiển thị

    # ------------------------------------------------------------------ #
    #  BƯỚC 2: Tạo RetrievalQA chain từ vectorstore + Ollama              #
    # ------------------------------------------------------------------ #
    def build_chain(self):
        """
        Kết nối 3 thành phần thành 1 chain hoàn chỉnh:
          FAISS retriever (tìm top-3 chunks liên quan)
          + PromptTemplate (đóng gói context + câu hỏi)
          + Ollama LLM (sinh câu trả lời)
        """
        if self.vectorstore is None:
            raise ValueError("Gọi build_vectorstore() trước!")

        # --- Retriever: tìm top-3 chunks gần nhất với câu hỏi ---
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",   # cosine similarity
            search_kwargs={"k": 3},     # lấy 3 chunks liên quan nhất
        )

        # --- Prompt template ---
        # {context}: 3 chunks tìm được (LangChain tự điền)
        # {question}: câu hỏi người dùng (LangChain tự điền)
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

        # --- Ollama LLM: chạy local, không cần internet ---
        llm = Ollama(
            model="qwen2.5:7b",
            temperature=0.3,      # thấp → câu trả lời ổn định, ít "sáng tạo"
        )

        # --- RetrievalQA: chain tích hợp retriever + llm + prompt ---
        # chain_type="stuff": nhét thẳng 3 chunks vào prompt (đơn giản nhất)
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,  # tuần sau mới cần (citation)
            chain_type_kwargs={"prompt": prompt},
        )

    # ------------------------------------------------------------------ #
    #  BƯỚC 3: Nhận câu hỏi → trả về câu trả lời                         #
    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> str:
        """
        Gửi câu hỏi vào chain, nhận câu trả lời dạng string.
        Chain sẽ tự động:
          1. Embed câu hỏi
          2. Tìm top-3 chunks trong FAISS
          3. Ghép chunks vào prompt
          4. Gửi prompt lên Ollama
          5. Trả về câu trả lời
        """
        if self.chain is None:
            return "Chưa nạp tài liệu. Hãy upload file trước."

        # invoke() trả về dict với key "result"
        response = self.chain.invoke({"query": question})
        return response["result"]

    # ------------------------------------------------------------------ #
    #  Hàm tiện ích: chạy cả pipeline trong 1 lệnh                        #
    # ------------------------------------------------------------------ #
    def load_and_build(self, file_path: str) -> int:
        """
        Gộp bước 1 + 2 lại.
        Người 3 (Streamlit) chỉ cần gọi hàm này sau khi user upload file.
        Trả về số chunks để hiển thị thông báo.
        """
        num_chunks = self.build_vectorstore(file_path)
        self.build_chain()
        return num_chunks


# ------------------------------------------------------------------ #
#  Chạy thử nhanh (không cần Streamlit)                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

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