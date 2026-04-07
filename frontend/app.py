import streamlit as st
import requests

st.title("RAG Chatbot")

question = st.text_input("Nhập câu hỏi:")

if st.button("Hỏi"):
    res = requests.post(
        "http://127.0.0.1:8000/ask",
        json={"question": question}
    )

    data = res.json()

    st.write("### 🧠 Answer")
    st.write(data["answer"])

    st.write("### 📚 Sources")
    for s in data["sources"]:
        st.write(f"- {s['source']} (page {s['page']})")