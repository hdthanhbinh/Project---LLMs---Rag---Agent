from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class PromptBuilder:
    def __init__(self):
        self.system_behavior = (
            "You are a technical document assistant. "
            "Always respond in the same language as the user's question. "
            "Use ONLY information from the CONTEXT section to answer. "
            "If the context is empty, irrelevant, or insufficient, "
            "respond exactly with: 'Tôi không tìm thấy thông tin phù hợp.' "
            "Never fabricate information outside the context. "
            "Be concise and direct. "
            "Citations are mandatory: use markers like [1], [2] from the numbered context. "
            "Every factual claim from the context should include at least one citation marker."
        )

    def _compact_text(self, text: str, limit: int) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        trimmed = compact[: max(limit - 3, 0)].rstrip()
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        return f"{trimmed}..."

    def _page_label(self, metadata: dict | None) -> str:
        metadata = metadata or {}
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

    def format_citation_context(
        self,
        docs: list,
        max_context_chars_per_doc: int = 2000,
        max_context_chars_total: int = 6000,
    ) -> str:
        blocks = []
        total_chars = 0

        for i, doc in enumerate(docs, 1):
            remaining = max_context_chars_total - total_chars
            if remaining <= 80:
                break

            per_doc_limit = min(max_context_chars_per_doc, remaining)
            content = self._compact_text(doc.page_content, per_doc_limit)
            if not content:
                continue

            metadata = doc.metadata or {}
            source_name = metadata.get("source", "unknown")
            page_label = self._page_label(metadata)
            block = (
                f"[{i}] source={source_name} page={page_label}\n"
                f"content: {content}"
            )
            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks)

    def get_rag_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.system_behavior),
            ("human", (
                "CONTEXT:\n{context}\n\n"
                "QUESTION: {question}\n\n"
                "Instructions:\n"
                "- Use only the context above.\n"
                "- If insufficient, say exactly: "
                "'Tôi không tìm thấy thông tin phù hợp.'\n"
                "- Citations are mandatory: use markers like [1], [2] that match the numbered references in the context.\n"
                "- Every factual claim from the context should include at least one citation marker.\n"
                "- Keep your answer concise and clear."
            )),
        ])

    def get_condense_question_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", (
                "Dua vao lich su chat ben duoi, hay dien dat lai cau hoi cua nguoi dung "
                "thanh MOT CAU DUY NHAT, doc lap, du nghia ma khong can xem lich su. "
                "Chi tra ve cau hoi, khong giai thich them."
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
