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
from langchain_core.messages import AIMessage, HumanMessage


class CoRAGService:
    MAX_SUB_QUESTIONS   = 3      # tối đa bao nhiêu sub-question
    MAX_CONTEXT_PER_DOC = 800    # ký tự tối đa mỗi chunk
    MAX_CONTEXT_TOTAL   = 4000   # ký tự tối đa tổng context

    def __init__(self, llm, prompt_builder, retrieve, k: int = 3):
        self.llm            = llm
        self.prompt_builder = prompt_builder
        self.retrieve       = retrieve
        self.k              = k

    def compact_text(self, text: str, limit: int) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        trimmed = compact[: max(limit - 3, 0)].rstrip()
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        return f"{trimmed}..."

    def format_history(self, chat_history: list[dict[str, str]] | None) -> str:
        if not chat_history:
            return "No prior conversation."

        lines = []
        for turn in chat_history[-6:]:
            user_text = self.compact_text(turn.get("question", ""), 300)
            answer_text = self.compact_text(turn.get("answer", ""), 500)
            lines.append(f"User: {user_text}\nAssistant: {answer_text}")
        return "\n\n".join(lines) if lines else "No prior conversation."

    def _history_messages(self, chat_history: list[dict[str, str]] | None) -> list:
        messages = []
        for turn in (chat_history or [])[-6:]:
            question = (turn.get("question") or "").strip()
            answer = (turn.get("answer") or "").strip()
            if question:
                messages.append(HumanMessage(content=question))
            if answer:
                messages.append(AIMessage(content=answer))
        return messages

    def rewrite_question(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        if not chat_history:
            return question

        prompt_template = self.prompt_builder.get_condense_question_prompt()
        prompt = prompt_template.invoke(
            {
                "chat_history": self._history_messages(chat_history),
                "question": question,
            }
        )
        try:
            response = self.llm.invoke(prompt)
            rewritten = getattr(response, "content", str(response)).strip()
            return rewritten or question
        except Exception as exc:
            print(f"CoRAG rewrite warning: {exc}")
            return question

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
        return self.prompt_builder.format_citation_context(
            docs,
            max_context_chars_per_doc=self.MAX_CONTEXT_PER_DOC,
            max_context_chars_total=self.MAX_CONTEXT_TOTAL,
        )

    def _page_label(self, metadata: dict[str, Any]) -> str:
        """Unified page formatting: 0-indexed PDF → 1-indexed display, DOCX → N/A"""
        file_type = str(metadata.get("file_type", "")).lstrip(".").lower()
        page = metadata.get("page")

        # DOCX files don't have page numbers
        if file_type == "docx":
            return "N/A"
        
        # For PDF, page is 0-indexed from PDFPlumberLoader; convert to 1-indexed for display
        if page is None or page == "":
            return "N/A"
        
        try:
            page_num = int(page)
            # Convert 0-indexed to 1-indexed for display
            return str(page_num + 1)
        except (TypeError, ValueError):
            return "N/A"

    def build_sources(self, docs: list) -> list[dict[str, Any]]:
        sources = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            page_label = self._page_label(metadata)

            sources.append({
                "index":   i,
                "source":  metadata.get("source", "unknown"),
                "page":    page_label,
                "page_number": metadata.get("page"),
                "file_type": metadata.get("file_type"),
                "source_path": metadata.get("source_path"),
                "chunk_id": metadata.get("chunk_id"),
                "chunk_index": metadata.get("chunk_index"),
                "char_start": metadata.get("char_start"),
                "char_end": metadata.get("char_end"),
                "content": doc.page_content,
            })
        return sources

    def _append_citations_if_missing(self, answer_text: str, sources: list[dict[str, Any]]) -> str:
        if not sources:
            return answer_text

        if "Tôi không tìm thấy thông tin phù hợp" in (answer_text or ""):
            return answer_text

        if re.search(r"\[(\d+)\]", answer_text or ""):
            return answer_text

        citation_list = ", ".join(f"[{src['index']}]" for src in sources)
        stripped = (answer_text or "").rstrip()
        if not stripped:
            return f"Nguồn tham khảo: {citation_list}."
        if stripped.endswith((".", "!", "?", ":")):
            return f"{stripped} Nguồn tham khảo: {citation_list}."
        return f"{stripped}. Nguồn tham khảo: {citation_list}."

    # ------------------------------------------------------------------ #
    #  Bước 3: Synthesize — sinh câu trả lời cuối từ toàn bộ context       #
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        question: str,
        sub_questions: list[str],
        docs: list,
        chat_history: list[dict[str, str]] | None = None,
        retrieval_question: str | None = None,
    ) -> str:
        context = self.format_context(docs)

        sub_q_text = "\n".join(f"- {q}" for q in sub_questions)

        synthesize_prompt = (
            "You are a technical document assistant. "
            "Answer the main question using ONLY the provided context. "
            "Always respond in the same language as the main question. "
            "Be concise. Citations are mandatory: use markers like [1], [2] "
            "from the numbered context for every factual claim.\n\n"
            f"Conversation history:\n{self.format_history(chat_history)}\n\n"
            f"Main question: {question}\n"
            f"Standalone question used for retrieval: {retrieval_question or question}\n\n"
            f"Sub-questions used for retrieval:\n{sub_q_text}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Instructions:\n"
            "- Use only the context above.\n"
            "- Use the conversation history only to understand follow-up intent.\n"
            "- DO NOT mention '[Doc X]' or any internal reference labels in your answer.\n"
            "- If you cannot answer based on the context, say: 'Tôi không tìm thấy thông tin phù hợp.'\n"
            "- Citations are mandatory: use markers like [1], [2] that match the numbered references in the context.\n"
            "- Every factual claim from the context should include at least one citation marker.\n"
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
    def answer(
        self,
        question: str,
        filter=None,
        chat_history: list[dict[str, str]] | None = None,
        rewrite_query: bool = True,
    ) -> dict[str, Any]:
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
        retrieval_question = (
            self.rewrite_question(question, chat_history)
            if rewrite_query else question
        )

        sub_questions = self.decompose(retrieval_question)

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
                    "retrieval_question": retrieval_question,
                    "conversation_turns": len(chat_history or []),
                },
            }

        # Bước 3: Synthesize
        answer_text = self.synthesize(
            question,
            sub_questions,
            docs,
            chat_history=chat_history,
            retrieval_question=retrieval_question,
        )
        answer_text = self._append_citations_if_missing(answer_text, self.build_sources(docs))

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
                "retrieval_question": retrieval_question,
                "conversation_turns": len(chat_history or []),
            },
        }
