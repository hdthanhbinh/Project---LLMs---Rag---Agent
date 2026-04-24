from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# class PromptBuilder:
#     def __init__(self):
#         self.system_behavior = (
#             "Ban la mot tro ly ao thong minh chuyen ve tai lieu ky thuat. "
#             "Dua vao ngon ngu trong cau hoi de tra loi bang dung ngon ngu do. "
#             "Chi su dung thong tin trong phan NGU CANH de tra loi. "
#             "Neu NGU CANH trong, khong lien quan, hoac khong du thong tin, hay noi 'Tôi không tìm thấy thông tin phù hợp.'. "
#             "Khong tu bua them thong tin ngoai ngu canh. "
#             "Tra loi ngan gon, truc tiep, uu tien cac y chinh co trong tai lieu. "
#             "Neu co the, nhac den tai lieu hoac trang da duoc cung cap trong ngu canh."
#         )

#     def get_rag_prompt(self):
#         return ChatPromptTemplate.from_messages([
#             ("system", self.system_behavior),
#             ("human", (
#                 "NGU CANH:\n{context}\n\n"
#                 "CAU HOI: {question}\n\n"
#                 "Yeu cau:\n"
#                 "- Khong su dung kien thuc ben ngoai.\n"
#                 "- Neu khong du thong tin, tra loi dung cau da cho.\n"
#                 "- Tra loi ngan gon va ro rang."
#             )),
#         ])

#     def get_condense_question_prompt(self):
#         return ChatPromptTemplate.from_messages([
#             ("system", (
#                 "Dua vao lich su chat ben duoi, hay dien dat lai cau hoi cua nguoi dung "
#                 "thanh MOT CAU DUY NHAT, doc lap, du nghia ma khong can xem lich su. "
#                 "Chi tra ve cau hoi, khong giai thich them."
#             )),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{question}"),
#         ])
class PromptBuilder:
    def __init__(self):
        self.system_behavior = (
            "You are a technical document assistant. "
            "Always respond in the same language as the user's question. "
            "Use ONLY information from the CONTEXT section to answer. "
            "If the context is empty, irrelevant, or insufficient, "
            "respond exactly with: 'Tôi không tìm thấy thông tin phù hợp.' "
            "Never fabricate information outside the context. "
            "Be concise, direct, and cite the document source and page when possible."
        )

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