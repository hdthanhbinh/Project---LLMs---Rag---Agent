from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptBuilder:
    def __init__(self):
        self.system_behavior = (
            "Bạn là một trợ lý ảo thông minh chuyên về tài liệu kỹ thuật. "
            "Dựa vào ngôn ngữ trong câu hỏi để trả lại ngôn ngữ phù hợp. "
            "Chỉ sử dụng thông tin trong phần NGỮ CẢNH dưới đây để trả lời. "
            "Nếu NGỮ CẢNH trống hoặc không liên quan, hãy nói 'Tôi không tìm thấy thông tin phù hợp.' "
            "Không tự bịa thêm thông tin ngoài ngữ cảnh. "
            "Luôn trả lời bằng ngôn ngữ của người dùng."
        )

    def get_rag_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.system_behavior),
            ("human", "NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}"),
        ])

    def get_condense_question_prompt(self):
        # Diễn đạt lại câu hỏi thành câu độc lập dựa trên lịch sử chat
        return ChatPromptTemplate.from_messages([
            ("system", (
                "Dựa vào lịch sử chat bên dưới, hãy diễn đạt lại câu hỏi của người dùng "
                "thành MỘT CÂU DUY NHẤT, độc lập, đủ nghĩa mà không cần xem lịch sử. "
                "Chỉ trả về câu hỏi, không giải thích thêm."
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])