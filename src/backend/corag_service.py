# src/backend/corag_service.py
#
# CoRAG = Chain-of-Thought RAG
# Flow:
#   1. Decompose: LLM breaks the question into 2-3 sub-questions
#   2. Retrieve:  run HybridRetriever independently for each sub-question
#   3. Synthesize: merge all context → LLM produces final answer

from typing import Any
import time
import re
from langchain_core.messages import AIMessage, HumanMessage


class CoRAGService:
    MAX_SUB_QUESTIONS   = 3
    MAX_CONTEXT_PER_DOC = 800
    MAX_CONTEXT_TOTAL   = 4000

    def __init__(self, llm, prompt_builder, retrieve, k: int = 3):
        self.llm            = llm
        self.prompt_builder = prompt_builder
        self.retrieve       = retrieve
        self.k              = k

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

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
            user_text   = self.compact_text(turn.get("question", ""), 300)
            answer_text = self.compact_text(turn.get("answer",   ""), 500)
            lines.append(f"User: {user_text}\nAssistant: {answer_text}")
        return "\n\n".join(lines) if lines else "No prior conversation."

    def _history_messages(self, chat_history: list[dict[str, str]] | None) -> list:
        messages = []
        for turn in (chat_history or [])[-6:]:
            question = (turn.get("question") or "").strip()
            answer   = (turn.get("answer")   or "").strip()
            if question:
                messages.append(HumanMessage(content=question))
            if answer:
                messages.append(AIMessage(content=answer))
        return messages

    def _get_rerank_meta(self) -> dict:
        """
        Read last_rerank_info from the retriever if available.
        Aggregates across multiple retrieve() calls by taking the max latency
        and the total candidate_count.
        """
        info = getattr(self.retrieve, "last_rerank_info", {}) or {}
        return {
            "rerank_enabled":  info.get("rerank_enabled", False),
            "rerank_model":    info.get("rerank_model"),
            "rerank_latency":  info.get("rerank_latency", 0.0),
            "candidate_count": info.get("candidate_count", 0),
        }

    # ------------------------------------------------------------------ #
    #  Step 0: Rewrite question                                            #
    # ------------------------------------------------------------------ #

    def rewrite_question(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        if not chat_history:
            return question
        prompt_template = self.prompt_builder.get_condense_question_prompt()
        prompt = prompt_template.invoke({
            "chat_history": self._history_messages(chat_history),
            "question": question,
        })
        try:
            response = self.llm.invoke(prompt)
            rewritten = getattr(response, "content", str(response)).strip()
            return rewritten or question
        except Exception as exc:
            print(f"CoRAG rewrite warning: {exc}")
            return question

    # ------------------------------------------------------------------ #
    #  Step 1: Decompose                                                   #
    # ------------------------------------------------------------------ #

    def decompose(self, question: str) -> list[str]:
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
            lines = text.splitlines()
            sub_qs = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                cleaned = re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", line).strip()
                if cleaned and len(cleaned) > 5:
                    sub_qs.append(cleaned)
            sub_qs = sub_qs[: self.MAX_SUB_QUESTIONS]
            return sub_qs if sub_qs else [question]
        except Exception as exc:
            print(f"CoRAG decompose warning: {exc}")
            return [question]

    # ------------------------------------------------------------------ #
    #  Step 2: Retrieve for each sub-question                              #
    # ------------------------------------------------------------------ #

    def retrieve_all(self, sub_questions: list[str], filter=None) -> list:
        """
        Run retriever for each sub-question, merge and deduplicate.
        Accumulates rerank_info across calls (max latency, total candidates).
        """
        seen_keys: set = set()
        all_docs: list = []

        # Accumulators for rerank meta across sub-question retrieves
        any_reranked = False
        total_latency = 0.0
        total_candidates = 0
        last_model = None

        for sq in sub_questions:
            try:
                docs = self.retrieve.retrieve(sq, filter=filter)

                # Collect rerank info for this call
                info = getattr(self.retrieve, "last_rerank_info", {}) or {}
                if info.get("rerank_enabled"):
                    any_reranked = True
                    total_latency += info.get("rerank_latency", 0.0)
                    total_candidates += info.get("candidate_count", 0)
                    last_model = info.get("rerank_model")

                for doc in docs:
                    key = (
                        doc.metadata.get("source", ""),
                        str(doc.metadata.get("page", "")),
                        doc.page_content[:80],
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_docs.append(doc)
            except Exception as exc:
                print(f"CoRAG retrieve warning for '{sq}': {exc}")

        # Store aggregated rerank info back on the retriever so
        # _get_rerank_meta() returns the right aggregated numbers.
        self.retrieve.last_rerank_info = {
            "rerank_enabled":  any_reranked,
            "rerank_model":    last_model,
            "rerank_latency":  round(total_latency, 3),
            "candidate_count": total_candidates,
        }

        return all_docs

    # ------------------------------------------------------------------ #
    #  Context / source helpers                                            #
    # ------------------------------------------------------------------ #

    def format_context(self, docs: list) -> str:
        return self.prompt_builder.format_citation_context(
            docs,
            max_context_chars_per_doc=self.MAX_CONTEXT_PER_DOC,
            max_context_chars_total=self.MAX_CONTEXT_TOTAL,
        )

    def _page_label(self, metadata: dict[str, Any]) -> str:
        file_type = str(metadata.get("file_type", "")).lstrip(".").lower()
        page = metadata.get("page")
        if file_type == "docx":
            return "N/A"
        if page is None or page == "":
            return "N/A"
        try:
            return str(int(page) + 1)
        except (TypeError, ValueError):
            return "N/A"

    def build_sources(self, docs: list) -> list[dict[str, Any]]:
        sources = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            sources.append({
                "index":       i,
                "source":      metadata.get("source", "unknown"),
                "page":        self._page_label(metadata),
                "page_number": metadata.get("page"),
                "file_type":   metadata.get("file_type"),
                "source_path": metadata.get("source_path"),
                "chunk_id":    metadata.get("chunk_id"),
                "chunk_index": metadata.get("chunk_index"),
                "char_start":  metadata.get("char_start"),
                "char_end":    metadata.get("char_end"),
                "content":     doc.page_content,
            })
        return sources

    def _append_citations_if_missing(
        self, answer_text: str, sources: list[dict[str, Any]]
    ) -> str:
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
    #  Step 3: Synthesize                                                  #
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
            "- Every factual claim from the context should include at least one citation marker.\n"
            "- Be concise and clear.\n\n"
            "Answer:"
        )
        try:
            response = self.llm.invoke(synthesize_prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as exc:
            return f"Lỗi sinh câu trả lời: {exc}"

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
                "meta":          {
                    "latency": 0.0,
                    "model": getattr(self.llm, "model", "unknown"),
                    "method": "corag",
                },
            }

        # Step 0: rewrite
        retrieval_question = (
            self.rewrite_question(question, chat_history)
            if rewrite_query else question
        )

        # Step 1: decompose
        sub_questions = self.decompose(retrieval_question)

        # Step 2: retrieve for each sub-question
        docs = self.retrieve_all(sub_questions, filter=filter)

        if not docs:
            return {
                "question":      question,
                "answer":        "Tôi không tìm thấy thông tin phù hợp.",
                "sources":       [],
                "sub_questions": sub_questions,
                "meta": {
                    "latency": round(time.time() - start, 2),
                    "model":   getattr(self.llm, "model", "unknown"),
                    "method":  "corag",
                    "retrieval_question": retrieval_question,
                    "conversation_turns": len(chat_history or []),
                    **self._get_rerank_meta(),
                },
            }

        # Step 3: synthesize
        answer_text = self.synthesize(
            question, sub_questions, docs,
            chat_history=chat_history,
            retrieval_question=retrieval_question,
        )
        answer_text = self._append_citations_if_missing(
            answer_text, self.build_sources(docs)
        )

        return {
            "question":      question,
            "answer":        answer_text,
            "sources":       self.build_sources(docs),
            "sub_questions": sub_questions,
            "meta": {
                "latency":    round(time.time() - start, 2),
                "model":      getattr(self.llm, "model", "unknown"),
                "method":     "corag",
                "num_sub":    len(sub_questions),
                "num_docs":   len(docs),
                "retrieval_question":  retrieval_question,
                "conversation_turns":  len(chat_history or []),
                **self._get_rerank_meta(),
            },
        }