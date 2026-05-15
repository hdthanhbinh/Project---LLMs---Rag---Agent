# src/backend/self_rag_service.py
#
# Self-RAG = Retrieve → Generate → Self-Evaluate → (optional retry)
#
# Flow:
#   1. Rewrite  : resolve follow-up / ambiguous questions via chat history
#   2. Retrieve : HybridRetriever (same as RAG / CoRAG)
#   3. Generate : draft answer from retrieved context
#   4. Self-eval: LLM judges
#                   a) context_relevant  – was the retrieved context useful?
#                   b) answer_grounded   – is every claim supported by context?
#                   c) confidence        – float 0-1
#   5. Retry    : if confidence < CONFIDENCE_THRESHOLD, refine query and retrieve again
#   6. Fallback : if self-eval crashes entirely, return plain RAG answer with confidence=None
#
# Output schema:
#   question, rewritten_question, answer, sources, confidence, self_eval, meta

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


class SelfRAGService:
    MAX_CONTEXT_PER_DOC  = 800
    MAX_CONTEXT_TOTAL    = 4000
    CONFIDENCE_THRESHOLD = 0.5   # retry once if confidence falls below this
    MAX_RETRIES          = 1     # single retry keeps latency reasonable

    def __init__(self, llm, prompt_builder, retrieve, k: int = 3):
        self.llm            = llm
        self.prompt_builder = prompt_builder
        self.retrieve       = retrieve
        self.k              = k

    # ------------------------------------------------------------------ #
    #  Internal utilities                                                  #
    # ------------------------------------------------------------------ #

    def _compact(self, text: str, limit: int) -> str:
        c = " ".join((text or "").split())
        if len(c) <= limit:
            return c
        t = c[: max(limit - 3, 0)].rstrip()
        if " " in t:
            t = t.rsplit(" ", 1)[0]
        return f"{t}..."

    def _page_label(self, metadata: dict) -> str:
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

    def _history_messages(self, chat_history: list[dict] | None) -> list:
        messages = []
        for turn in (chat_history or [])[-6:]:
            q = (turn.get("question") or "").strip()
            a = (turn.get("answer")   or "").strip()
            if q:
                messages.append(HumanMessage(content=q))
            if a:
                messages.append(AIMessage(content=a))
        return messages

    def _format_history(self, chat_history: list[dict] | None) -> str:
        if not chat_history:
            return "No prior conversation."
        lines = []
        for turn in (chat_history or [])[-6:]:
            u = self._compact(turn.get("question", ""), 300)
            a = self._compact(turn.get("answer",   ""), 500)
            lines.append(f"User: {u}\nAssistant: {a}")
        return "\n\n".join(lines) if lines else "No prior conversation."

    def _get_rerank_meta(self) -> dict:
        info = getattr(self.retrieve, "last_rerank_info", {}) or {}
        return {
            "rerank_enabled":  info.get("rerank_enabled",  False),
            "rerank_model":    info.get("rerank_model"),
            "rerank_latency":  info.get("rerank_latency",  0.0),
            "candidate_count": info.get("candidate_count", 0),
        }

    def _format_context(self, docs: list) -> str:
        return self.prompt_builder.format_citation_context(
            docs,
            max_context_chars_per_doc=self.MAX_CONTEXT_PER_DOC,
            max_context_chars_total=self.MAX_CONTEXT_TOTAL,
        )

    def build_sources(self, docs: list) -> list[dict[str, Any]]:
        sources = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            sources.append({
                "index":       i,
                "source":      meta.get("source", "unknown"),
                "page":        self._page_label(meta),
                "page_number": meta.get("page"),
                "file_type":   meta.get("file_type"),
                "source_path": meta.get("source_path"),
                "chunk_id":    meta.get("chunk_id"),
                "chunk_index": meta.get("chunk_index"),
                "char_start":  meta.get("char_start"),
                "char_end":    meta.get("char_end"),
                "content":     doc.page_content,
            })
        return sources

    def _append_citations_if_missing(self, answer_text: str, sources: list[dict]) -> str:
        if not sources:
            return answer_text
        if "Tôi không tìm thấy thông tin phù hợp" in (answer_text or ""):
            return answer_text
        if re.search(r"\[(\d+)\]", answer_text or ""):
            return answer_text
        citation_list = ", ".join(f"[{s['index']}]" for s in sources)
        stripped = (answer_text or "").rstrip()
        if not stripped:
            return f"Nguồn tham khảo: {citation_list}."
        if stripped.endswith((".", "!", "?", ":")):
            return f"{stripped} Nguồn tham khảo: {citation_list}."
        return f"{stripped}. Nguồn tham khảo: {citation_list}."

    # ------------------------------------------------------------------ #
    #  Step 1: Rewrite question                                            #
    # ------------------------------------------------------------------ #

    def rewrite_question(
        self, question: str, chat_history: list[dict] | None = None
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
            print(f"SelfRAG rewrite warning: {exc}")
            return question

    # ------------------------------------------------------------------ #
    #  Step 3: Generate draft answer                                       #
    # ------------------------------------------------------------------ #

    def _generate(
        self,
        question: str,
        retrieval_question: str,
        docs: list,
        chat_history: list[dict] | None = None,
    ) -> str:
        context = self._format_context(docs)
        generate_prompt = (
            "You are a technical document assistant. "
            "Answer the question using ONLY the provided context. "
            "Always respond in the same language as the question. "
            "Be concise. Use citation markers like [1], [2] for every factual claim.\n\n"
            f"Conversation history:\n{self._format_history(chat_history)}\n\n"
            f"Question: {question}\n"
            f"Standalone question used for retrieval: {retrieval_question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Instructions:\n"
            "- Use ONLY the context above.\n"
            "- DO NOT mention '[Doc X]' or internal labels.\n"
            "- If the context is insufficient, say exactly: "
            "'Tôi không tìm thấy thông tin phù hợp.'\n"
            "- Every factual claim must include at least one citation marker.\n\n"
            "Answer:"
        )
        try:
            response = self.llm.invoke(generate_prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as exc:
            return f"Lỗi sinh câu trả lời: {exc}"

    # ------------------------------------------------------------------ #
    #  Step 4: Self-evaluate                                               #
    # ------------------------------------------------------------------ #

    def _self_evaluate(
        self, question: str, context: str, answer: str
    ) -> dict | None:
        """
        Returns a dict:
            context_relevant : bool
            answer_grounded  : bool
            confidence       : float  0.0 – 1.0
            reasoning        : str

        Returns None on parse / LLM failure (triggers fallback path).
        """
        eval_prompt = (
            "You are a self-evaluation critic for a RAG system.\n"
            "Evaluate the draft answer against the question and context.\n\n"
            "Criteria:\n"
            "1. context_relevant – Does the context contain enough information "
            "to properly answer the question? (true/false)\n"
            "2. answer_grounded  – Is the draft answer fully supported by the context "
            "without hallucination? (true/false)\n"
            "3. confidence       – Your overall confidence that the answer is correct "
            "and complete, as a float 0.0 (cannot answer) to 1.0 (fully confident).\n\n"
            f"Question: {question}\n\n"
            f"Context (truncated to 1 500 chars):\n{context[:1500]}\n\n"
            f"Draft answer (truncated to 800 chars):\n{answer[:800]}\n\n"
            "Respond ONLY with valid JSON — no markdown fences, no extra text.\n"
            "The 'reasoning' field MUST be written in Vietnamese.\n"
            'Example: {"context_relevant": true, "answer_grounded": true, '
            '"confidence": 0.85, "reasoning": "Ngữ cảnh cung cấp đủ thông tin và câu trả lời bám sát tài liệu."}'
        )
        try:
            response = self.llm.invoke(eval_prompt)
            raw = getattr(response, "content", str(response)).strip()
            # Strip ```json ... ``` fences if present
            raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            return {
                "context_relevant": bool(parsed.get("context_relevant", True)),
                "answer_grounded":  bool(parsed.get("answer_grounded",  True)),
                "confidence":       max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
                "reasoning":        str(parsed.get("reasoning", "")),
            }
        except Exception as exc:
            print(f"SelfRAG self-eval warning: {exc}")
            return None

    # ------------------------------------------------------------------ #
    #  Step 5 (helper): Refine query before retry                         #
    # ------------------------------------------------------------------ #

    def _refine_query(self, question: str, eval_result: dict) -> str:
        refine_prompt = (
            "The previous retrieval did not return sufficiently relevant context.\n"
            f"Original question: {question}\n"
            f"Self-evaluation reasoning: {eval_result.get('reasoning', '')}\n\n"
            "Rewrite the question with more specific keywords so a vector search "
            "retrieves better documents. "
            "Return ONLY the rewritten question, no explanation."
        )
        try:
            response = self.llm.invoke(refine_prompt)
            refined = getattr(response, "content", str(response)).strip()
            return refined or question
        except Exception:
            return question

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def answer(
        self,
        question: str,
        filter=None,
        chat_history: list[dict] | None = None,
        rewrite_query: bool = True,
    ) -> dict[str, Any]:
        start    = time.time()
        question = (question or "").strip()

        if not question:
            return {
                "question":            question,
                "rewritten_question":  question,
                "answer":              "",
                "sources":             [],
                "confidence":          None,
                "self_eval":           None,
                "meta": {"latency": 0.0, "method": "self_rag"},
            }

        # ── Step 1: rewrite ───────────────────────────────────────────
        rewritten = (
            self.rewrite_question(question, chat_history)
            if rewrite_query else question
        )

        # ── Step 2: retrieve ──────────────────────────────────────────
        docs = self.retrieve.retrieve(rewritten, filter=filter)
        rerank_meta = self._get_rerank_meta()

        if not docs:
            return {
                "question":           question,
                "rewritten_question": rewritten,
                "answer":             "Tôi không tìm thấy thông tin phù hợp.",
                "sources":            [],
                "confidence":         0.0,
                "self_eval": {
                    "context_relevant": False,
                    "answer_grounded":  False,
                    "confidence":       0.0,
                    "reasoning":        "No documents retrieved.",
                },
                "meta": {
                    "latency":            round(time.time() - start, 2),
                    "model":              getattr(self.llm, "model", "unknown"),
                    "method":             "self_rag",
                    "retried":            False,
                    "fallback":           False,
                    "conversation_turns": len(chat_history or []),
                    **rerank_meta,
                },
            }

        # ── Step 3: generate draft ────────────────────────────────────
        answer_text = self._generate(question, rewritten, docs, chat_history)

        # ── Step 4: self-evaluate ─────────────────────────────────────
        context_str = self._format_context(docs)
        eval_result = self._self_evaluate(question, context_str, answer_text)
        retried     = False

        # ── Fallback: eval failed completely ─────────────────────────
        if eval_result is None:
            answer_text = self._append_citations_if_missing(
                answer_text, self.build_sources(docs)
            )
            return {
                "question":           question,
                "rewritten_question": rewritten,
                "answer":             answer_text,
                "sources":            self.build_sources(docs),
                "confidence":         None,
                "self_eval":          None,
                "meta": {
                    "latency":            round(time.time() - start, 2),
                    "model":              getattr(self.llm, "model", "unknown"),
                    "method":             "self_rag",
                    "num_docs":           len(docs),
                    "retried":            False,
                    "fallback":           True,
                    "conversation_turns": len(chat_history or []),
                    **rerank_meta,
                },
            }

        # ── Step 5: retry once if confidence is low ───────────────────
        if eval_result["confidence"] < self.CONFIDENCE_THRESHOLD:
            refined_query = self._refine_query(question, eval_result)
            retry_docs    = self.retrieve.retrieve(refined_query, filter=filter)
            rerank_meta   = self._get_rerank_meta()   # refresh after second call

            if retry_docs:
                docs         = retry_docs
                rewritten    = refined_query
                answer_text  = self._generate(question, rewritten, docs, chat_history)
                context_str  = self._format_context(docs)
                new_eval     = self._self_evaluate(question, context_str, answer_text)
                if new_eval is not None:
                    eval_result = new_eval
                retried = True

        answer_text = self._append_citations_if_missing(
            answer_text, self.build_sources(docs)
        )

        return {
            "question":           question,
            "rewritten_question": rewritten,
            "answer":             answer_text,
            "sources":            self.build_sources(docs),
            "confidence":         eval_result.get("confidence"),
            "self_eval":          eval_result,
            "meta": {
                "latency":            round(time.time() - start, 2),
                "model":              getattr(self.llm, "model", "unknown"),
                "method":             "self_rag",
                "num_docs":           len(docs),
                "retried":            retried,
                "fallback":           False,
                "conversation_turns": len(chat_history or []),
                **rerank_meta,
            },
        }