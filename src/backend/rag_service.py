from typing import Any
import time


class RagService:
    def __init__(self, llm, prompt_builder, retrieve, k: int = 3):
        self.k = k
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.retrieve = retrieve

    def normalize_question(self, question: Any) -> str:
        if question is None:
            return ""
        return str(question).strip()

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
        return "\n\n".join(
            f"[Tai lieu {i}] Nguon: {doc.metadata.get('source')}, Trang: {doc.metadata.get('page')}\n"
            f"Noi dung: {doc.page_content}"
            for i, doc in enumerate(results, 1)
        )

    def answer(self, question: str) -> dict[str, Any]:
        start = time.time()
        question = self.normalize_question(question)

        if not question:
            return self.build_response(
                question=question,
                answer="",
                docs=[],
                latency=time.time() - start,
            )

        docs = self.retrieve.retrieve(question)
        if not docs:
            return self.build_response(
                question=question,
                answer="Toi khong tim thay thong tin phu hop.",
                docs=[],
                latency=time.time() - start,
            )

        context = self.format_context(docs)
        prompt = self.prompt_builder.get_rag_prompt().format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer_text = getattr(response, "content", str(response)).strip()

        return self.build_response(
            question=question,
            answer=answer_text,
            docs=docs,
            latency=time.time() - start,
        )
