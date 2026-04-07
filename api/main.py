from contextlib import asynccontextmanager
from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import asyncio

from backend.rag_service import RagService
from backend.llm import get_llm
from backend.prompt_builder import PromptBuilder
from backend.vector_store import load_vector_store
from backend.retriever import HybridRetriever , SemanticRetriever, KeywordRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # ===== INIT 1 lần =====
        llm = get_llm()
        prompt_builder = PromptBuilder()
        vector_store = load_vector_store()

        documents = list(vector_store.docstore._dict.values())  # get all documents from the vector store's docstore

        semantic_retriever = SemanticRetriever(vector_store)

        keyword_retriever = KeywordRetriever(documents)

        retriever = HybridRetriever(semantic_retriever, keyword_retriever)

        app.state.rag_service = RagService(vector_store, llm, prompt_builder, retriever)

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise 
    yield #still running...
    print("🛑 Shutting down...")
app = FastAPI(lifespan=lifespan)

# ===== Request =====
class Question(BaseModel):
    question: str


class SourceItem(BaseModel):
    index: int
    source: str
    page: int
    content: str

class Answer(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem] = []
    meta: dict = {}

@app.post("/ask", response_model=Answer)
async def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, app.state.rag_service.answer, q.question)
        return result
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
@app.get("/health")
def health():
    return {"status": "ok", "rag_ready": hasattr(app.state, "rag")}