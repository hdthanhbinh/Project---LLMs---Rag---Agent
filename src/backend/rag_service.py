from typing import Any
import time
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
class RagService:
    def __init__(self, llm, prompt_builder, retrieve, k: int = 3):
        self.k = k
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.retrieve = retrieve
        # Default context limits used by format_context.
        self.MAX_CONTEXT_CHARS_PER_DOC = 2000
        self.MAX_CONTEXT_CHARS_TOTAL = 6000

    def normalize_question(self, question: Any) -> str:
        if question is None:
            return ""
        return str(question).strip()

    def compact_text(self, text: str, limit: int) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        trimmed = compact[: max(limit - 3, 0)].rstrip()
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        return f"{trimmed}..."

    def select_docs(self, docs: list) -> list:
        selected = []
        seen = set()

        for doc in docs:
            metadata = doc.metadata or {}
            key = (
                metadata.get("source", "unknown"),
                metadata.get("page", 0),
                self.compact_text(doc.page_content, 120),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(doc)
            

        return selected

    def build_sources(self, docs: list) -> list[dict[str, Any]]:
        sources = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page") or 0
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 0

            sources.append(
                {
                    "index": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": page,
                    "content": doc.page_content,
                }
            )
        return sources

    def build_response(
        self,
        question: str,
        answer: str,
        docs: list | None = None,
        latency: float = 0.0,
    ) -> dict[str, Any]:
        docs = docs or []
        return {
            "question": question,
            "answer": answer,
            "sources": self.build_sources(docs),
            "meta": {
                "k": len(docs),
                "latency": round(latency, 2),
                "model": getattr(self.llm, "model", "unknown"),
            },
        }

    def format_context(self, results: list) -> str:
        blocks = []
        total_chars = 0

        for i, doc in enumerate(results, 1):
            remaining = self.MAX_CONTEXT_CHARS_TOTAL - total_chars
            if remaining <= 80:
                break

            per_doc_limit = min(self.MAX_CONTEXT_CHARS_PER_DOC, remaining)
            content = self.compact_text(doc.page_content, per_doc_limit)
            if not content:
                continue

            block = (
                f"[Tai lieu {i}] Nguon: {doc.metadata.get('source', 'unknown')}, "
                f"Trang: {doc.metadata.get('page', 0)}\n"
                f"Noi dung: {content}"
            )
            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks)

    def invoke_with_fallback(self, question: str, docs: list):
        prompt_template = self.prompt_builder.get_rag_prompt()
        context = self.format_context(docs)
        print(f"DEBUG prompt docs={len(docs)} context_chars={len(context)}")
        prompt = prompt_template.invoke({"context": context, "question": question})

        try:
            return self.llm.invoke(prompt), docs
        except Exception as e:  # ✅ bind 'e'
            print(f"ERROR: {e}")
            if len(docs) <= 1:
                raise

            fallback_docs = docs[:1]
            fallback_context = self.format_context(fallback_docs)
            print(f"DEBUG retry with smaller context docs={len(fallback_docs)} "
                f"context_chars={len(fallback_context)}")
            fallback_prompt = prompt_template.invoke(
                {"context": fallback_context, "question": question}
            )
            return self.llm.invoke(fallback_prompt), fallback_docs

    def answer(self, question: str,
           filter=None) -> dict[str, Any]:
        start = time.time()
        question = self.normalize_question(question)

        if not question:
            return self.build_response(question=question, answer="",
                                    docs=[], latency=time.time() - start)

        # ✅ truyền filter xuống retriever
        raw_docs = self.retrieve.retrieve(question, filter=filter)
        docs = self.select_docs(raw_docs)

        if not docs:
            msg = (
                "Tôi không tìm thấy thông tin phù hợp trong các tài liệu được chọn."
                if filter else
                "Tôi không tìm thấy thông tin phù hợp."
            )
            return self.build_response(question=question, answer=msg,
                                    docs=[], latency=time.time() - start)

        try:
            response, docs_used = self.invoke_with_fallback(question, docs)
            answer_text = getattr(response, "content", str(response)).strip()
        except Exception as exc:
            err_text = str(exc).lower()
            if "winerror 10061" in err_text or "connection refused" in err_text:
                answer_text = (
                    "Không thể kết nối tới Ollama. "
                    "Hãy mở terminal và chạy `ollama serve`, sau đó thử lại."
                )
                return self.build_response(
                    question=question,
                    answer=answer_text,
                    docs=[],
                    latency=time.time() - start,
                )
            raise

        return self.build_response(question=question, answer=answer_text,
                                docs=docs_used, latency=time.time() - start)
