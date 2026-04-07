from langchain_ollama import ChatOllama

LLM_MODEL_NAME = "qwen2.5:3b"

def get_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0.1,
    )
