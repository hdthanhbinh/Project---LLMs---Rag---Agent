from typing import Any
import time
import torch
import re

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

    def _page_label(self, metadata: dict[str, Any]) -> str:
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
            metadata = doc.metadata or {}
            page_label = self._page_label(metadata)

            sources.append(
                {
                    "index": i,
                    "source": metadata.get("source", "unknown"),
                    "page": page_label,
                    "page_number": metadata.get("page"),
                    "file_type": metadata.get("file_type"),
                    "source_path": metadata.get("source_path"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "char_start": metadata.get("char_start"),
                    "char_end": metadata.get("char_end"),
                    "content": doc.page_content,
                }
            )
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
        return self.prompt_builder.format_citation_context(
            results,
            max_context_chars_per_doc=self.MAX_CONTEXT_CHARS_PER_DOC,
            max_context_chars_total=self.MAX_CONTEXT_CHARS_TOTAL,
        )

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
            answer_text = self._append_citations_if_missing(answer_text, self.build_sources(docs_used))
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
