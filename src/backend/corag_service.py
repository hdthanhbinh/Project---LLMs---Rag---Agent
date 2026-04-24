# src/backend/corag_service.py
#
# CoRAG = Chain-of-Thought RAG
# Ý tưởng: thay vì chỉ retrieve 1 lần rồi generate,
# CoRAG lặp lại quá trình: sinh sub-questions → retrieve → tổng hợp.
#
# Luồng:
#   1. Decompose: LLM phân tách câu hỏi gốc thành 2-3 sub-questions
#   2. Retrieve:  với MỖI sub-question, chạy HybridRetriever độc lập
#   3. Synthesize: gộp tất cả context tìm được → LLM sinh câu trả lời cuối

from typing import Any
import time
import re


class CoRAGService:
    MAX_SUB_QUESTIONS   = 3      # tối đa bao nhiêu sub-question
    MAX_CONTEXT_PER_DOC = 800    # ký tự tối đa mỗi chunk
    MAX_CONTEXT_TOTAL   = 4000   # ký tự tối đa tổng context

    def __init__(self, llm, prompt_builder, retrieve, k: int = 3):
        self.llm            = llm
        self.prompt_builder = prompt_builder
        self.retrieve       = retrieve
        self.k              = k

    # ------------------------------------------------------------------ #
    #  Bước 1: Decompose câu hỏi thành sub-questions                       #
    # ------------------------------------------------------------------ #
    def decompose(self, question: str) -> list[str]:
        """
        Dùng LLM để tách câu hỏi gốc thành các sub-questions nhỏ hơn.
        Nếu LLM lỗi hoặc chỉ có 1 ý, trả về [question] gốc.
        """
        decompose_prompt = (
            "You are a query decomposition assistant.\n"
            "Break the following question into 2-3 specific sub-questions "
            "that together cover the full answer. "
            "Return ONLY a numbered list, one sub-question per line. "
            "If the question is already atomic, return it as-is.\n\n"
            f"Question: {question}\n\nSub-questions:"
        )
        try:
            response = self.llm.invoke(decompose_prompt)
            text = getattr(response, "content", str(response)).strip()

            # Parse numbered list: "1. ...", "2. ..." hoặc "- ..."
            lines = text.splitlines()
            sub_qs = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Bỏ số đầu dòng "1.", "2." hoặc dấu "-"
                cleaned = re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", line).strip()
                if cleaned and len(cleaned) > 5:
                    sub_qs.append(cleaned)

            # Giới hạn số sub-questions và fallback
            sub_qs = sub_qs[:self.MAX_SUB_QUESTIONS]
            return sub_qs if sub_qs else [question]

        except Exception as e:
            print(f"CoRAG decompose warning: {e}")
            return [question]

    # ------------------------------------------------------------------ #
    #  Bước 2: Retrieve cho từng sub-question                              #
    # ------------------------------------------------------------------ #
    def retrieve_all(self, sub_questions: list[str], filter=None) -> list:
        """
        Chạy retriever cho mỗi sub-question, gộp kết quả, dedup theo key.
        """
        seen_keys = set()
        all_docs  = []

        for sq in sub_questions:
            try:
                docs = self.retrieve.retrieve(sq, filter=filter)
                for doc in docs:
                    key = (
                        doc.metadata.get("source", ""),
                        str(doc.metadata.get("page", "")),
                        doc.page_content[:80],
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_docs.append(doc)
            except Exception as e:
                print(f"CoRAG retrieve warning for '{sq}': {e}")

        return all_docs

    # ------------------------------------------------------------------ #
    #  Format context có ghi rõ sub-question nào tìm ra chunk nào          #
    # ------------------------------------------------------------------ #
    def format_context(self, docs: list) -> str:
        blocks = []
        total  = 0

        for i, doc in enumerate(docs, 1):
            remaining = self.MAX_CONTEXT_TOTAL - total
            if remaining <= 80:
                break
            limit   = min(self.MAX_CONTEXT_PER_DOC, remaining)
            content = " ".join(doc.page_content.split())
            content = content[:limit] + ("..." if len(content) > limit else "")

            block = (
                f"[Doc {i}] Source: {doc.metadata.get('source','unknown')}, "
                f"Page: {doc.metadata.get('page', 0)}\n"
                f"Content: {content}"
            )
            blocks.append(block)
            total += len(block)

        return "\n\n".join(blocks)

    def build_sources(self, docs: list) -> list[dict[str, Any]]:
        sources = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page") or 0
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 0
            sources.append({
                "index":   i,
                "source":  doc.metadata.get("source", "unknown"),
                "page":    page,
                "content": doc.page_content,
            })
        return sources

    # ------------------------------------------------------------------ #
    #  Bước 3: Synthesize — sinh câu trả lời cuối từ toàn bộ context       #
    # ------------------------------------------------------------------ #
    def synthesize(self, question: str, sub_questions: list[str], docs: list) -> str:
        context = self.format_context(docs)

        sub_q_text = "\n".join(f"- {q}" for q in sub_questions)

        synthesize_prompt = (
            "You are a technical document assistant. "
            "Answer the main question using ONLY the provided context. "
            "Always respond in the same language as the main question. "
            "Be concise and cite the source when possible.\n\n"
            f"Main question: {question}\n\n"
            f"Sub-questions used for retrieval:\n{sub_q_text}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Instructions:\n"
            "- Use only the context above.\n"
            "- DO NOT mention'[Doc X]' or any internal reference labels in your answer.\n"
            "- If you cannot answer based on the context, say: 'Tôi không tìm thấy thông tin phù hợp.'\n"
            "- Be concise and clear.\n\n"
            "Answer:"
        )

        try:
            response = self.llm.invoke(synthesize_prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as e:
            return f"Lỗi sinh câu trả lời: {e}"

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #
    def answer(self, question: str, filter=None) -> dict[str, Any]:
        start    = time.time()
        question = (question or "").strip()

        if not question:
            return {
                "question":      question,
                "answer":        "",
                "sources":       [],
                "sub_questions": [],
                "meta":          {"latency": 0.0, "model": getattr(self.llm, "model", "unknown"), "method": "corag"},
            }

        # Bước 1: Decompose
        sub_questions = self.decompose(question)

        # Bước 2: Retrieve cho mỗi sub-question
        docs = self.retrieve_all(sub_questions, filter=filter)

        if not docs:
            return {
                "question":      question,
                "answer":        "Tôi không tìm thấy thông tin phù hợp.",
                "sources":       [],
                "sub_questions": sub_questions,
                "meta":          {
                    "latency": round(time.time() - start, 2),
                    "model":   getattr(self.llm, "model", "unknown"),
                    "method":  "corag",
                },
            }

        # Bước 3: Synthesize
        answer_text = self.synthesize(question, sub_questions, docs)

        return {
            "question":      question,
            "answer":        answer_text,
            "sources":       self.build_sources(docs),
            "sub_questions": sub_questions,   # để UI hiển thị bước phân tích
            "meta": {
                "latency": round(time.time() - start, 2),
                "model":   getattr(self.llm, "model", "unknown"),
                "method":  "corag",
                "num_sub": len(sub_questions),
                "num_docs": len(docs),
            },
        }