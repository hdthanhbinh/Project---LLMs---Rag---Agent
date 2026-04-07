from pathlib import Path
from pprint import pprint

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.prompt_builder import PromptBuilder
from backend.retriever import SemanticRetriever, KeywordRetriever , HybridRetriever
import time

from backend.llm import get_llm
from backend.vector_store import (
    INDEX_DIR,
    build_vector_store,
    load_vector_store,
    save_vector_store,
)
LOADER_CONFIG = {
            "path": "data",
            "glob": "**/*.pdf",
            "silent_errors": True,
            "loader_cls": PyPDFLoader,
        }

class RagService:
    def __init__(self , vector_store, llm, prompt_builder, retrieve , k: int = 3):
            self.k = k
            self.vector_store = vector_store
            self.llm = llm
            self.prompt_builder  = prompt_builder  
            self.retrieve  = retrieve        
    


    def load_documents(self) -> list:
        documents = []
        data_dir = Path(LOADER_CONFIG["path"])
        loader_cls = LOADER_CONFIG["loader_cls"]
        glob_pattern = LOADER_CONFIG["glob"]

        for pdf_path in data_dir.glob(glob_pattern):
            try:
                loader = loader_cls(str(pdf_path))
                documents.extend(loader.load())
            except Exception as exc:
                if not LOADER_CONFIG["silent_errors"]:
                    print(f"Error loading {pdf_path}: {exc}")

        return documents


    def split_documents(self, documents: list) -> list:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        return text_splitter.split_documents(documents)


    def build_index_if_missing(self) -> None:
        if INDEX_DIR.exists():
            return

        print("Building FAISS index...")
        documents = self.load_documents()
        pprint(f"Total documents: {len(documents)}")

        chunks = self.split_documents(documents)
        pprint(f"Total chunks: {len(chunks)}")

        vector_store = build_vector_store(chunks)
        save_vector_store(vector_store)


    def get_vector_store(self) :
        self.build_index_if_missing()
        return load_vector_store()

    def format_context(self, results: list) -> str:
        return "\n\n".join(
            f"[Tài liệu {i}] Nguồn: {doc.metadata.get('source')}, Trang: {doc.metadata.get('page')}\n"
            f"Nội dung: {doc.page_content}"
            for i, doc in enumerate(results, 1)
        )

    # def question(self, question: str, k: int = 3) -> str:
    #     query = question
    #     docs = self.retrieve.retrieve(query)
    #     context = "\n\n".join(doc.page_content for doc in docs)
    #     prompt = self.prompt_builder.build_prompt(query, context)
    #     response = self.llm.invoke(prompt)
    #     return response.content
    
    def answer(self , question: str) -> str:
        start = time.time()
        if not isinstance(question, str):
            question = str(question)
        question = question.strip()
        if not question:
            return {
                "question": question,
                "answer": "",
                
                "sources": [list],
            }
        docs = self.retrieve.retrieve(question)
        context = self.format_context(docs)
        prompt = self.prompt_builder.get_rag_prompt().format(context=context, question=question)
        response = self.llm.invoke(prompt)
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                "index": i,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page") or 0,
                "content": doc.page_content,
            })
        end = time.time()
        return {
            "question": question,
            "answer": response.content,
            
            "sources": sources,
            "meta": {
            "k": len(docs),
            "latency": round(end - start, 2),
            "model": "qwen2.5"
        }
        }
    




# def main() -> None:
#     llm = get_llm()
#     rag_service = RagService(...)
#     vector_store = rag_service.get_vector_store()
#     prompt_builder = PromptBuilder()
#     retriever = SemanticRetriever(vector_store)
#     rag_service = RagService(vector_store, llm, prompt_builder, retriever)

#     while True:
#         try:
#             question = input("Nhap cau hoi (go 'exit' de thoat): ")
#         except (EOFError, KeyboardInterrupt):
#             break
#         question = question.strip()
#         if not question:
#             continue
#         if question.lower() in {"exit", "quit"}:
#             break
#         try:
#             result = rag_service.answer(question)
#         except KeyboardInterrupt:
#             print("\n[Stopped] Answer generation cancelled.")
#             continue
#         print("\n===== Answer =====")
#         print(result["answer"])
#         print("\n===== Sources =====")
#         for source in result["sources"]:
#             print(f"- {source['source']} (page {source['page']})")


# if __name__ == "__main__":
#     main()



