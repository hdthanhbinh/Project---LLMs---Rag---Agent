from contextlib import asynccontextmanager
import asyncio
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from src.backend.llm import get_llm
from src.backend.prompt_builder import PromptBuilder
from src.backend.rag_service import RagService
from src.backend.retriever import HybridRetriever, KeywordRetriever, SemanticRetriever
from src.backend.vector_store import INDEX_DIR, build_vector_store, load_vector_store, save_vector_store
from src.processor import process_multiple_documents
from datetime import datetime
import traceback

DATA_DIR = Path("data/uploads")
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


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
    sources: list[SourceItem] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)


class UploadResponse(BaseModel):
    message: str
    files: list[str] = Field(default_factory=list)
    chunk_count: int = 0



def build_rag_service(vector_store, llm, prompt_builder) -> RagService:
    documents = list(vector_store.docstore._dict.values())
    semantic_retriever = SemanticRetriever(vector_store)
    keyword_retriever = KeywordRetriever(documents)
    retriever = HybridRetriever(semantic_retriever, keyword_retriever)
    return RagService(llm, prompt_builder, retriever)



def collect_uploaded_files() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )



def resolve_upload_path(filename: str) -> Path:
    safe_name = Path(filename or "uploaded_file").name
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return DATA_DIR / f"{stem}_{timestamp}{suffix}"



def rebuild_index(file_paths: list[str], llm, prompt_builder):
    chunks = process_multiple_documents(file_paths)
    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    rag_service = build_rag_service(vector_store, llm, prompt_builder)
    return vector_store, rag_service, len(chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = get_llm()
    app.state.prompt_builder = PromptBuilder()
    app.state.rag_service = None
    app.state.index_lock = asyncio.Lock()

    try:
        if INDEX_DIR.exists():
            vector_store = load_vector_store()
            app.state.rag_service = build_rag_service(
                vector_store,
                app.state.llm,
                app.state.prompt_builder,
            )
    except Exception as exc:
        print(f"Startup warning: {exc}")

    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)



def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    upload_schema = openapi_schema.get("components", {}).get("schemas", {}).get("Body_upload_upload_post")
    if upload_schema:
        files_schema = upload_schema.get("properties", {}).get("files")
        if files_schema and files_schema.get("type") == "array":
            files_schema["items"] = {"type": "string", "format": "binary"}

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.post("/ask", response_model=Answer)
async def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if app.state.rag_service is None:
        raise HTTPException(status_code=503, detail="RAG index is not ready. Upload documents first.")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, app.state.rag_service.answer, q.question)
        return result
    except Exception as exc:
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=str(exc)) 


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_ready": app.state.rag_service is not None,
        "indexed": INDEX_DIR.exists(),
        "uploaded_files": len(collect_uploaded_files()),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    MAX_FILE_SIZE = 10 * 1024 * 1024

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported: {file.filename}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"{file.filename} vượt quá 10MB.")

        target_path = resolve_upload_path(file.filename or "uploaded_file")
        target_path.write_bytes(content)
        saved_files.append(str(target_path))  

    all_files = [str(path) for path in collect_uploaded_files()]
    chunk_count = 0 

    try:
        loop = asyncio.get_running_loop()
        async with app.state.index_lock:
            _, rag_service, chunk_count = await asyncio.wait_for(
                loop.run_in_executor(None, rebuild_index, all_files, app.state.llm, app.state.prompt_builder),
                timeout=120.0
            )
        app.state.rag_service = rag_service
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Index rebuild timed out.")
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to process uploaded files.")

    return UploadResponse(
        message="Files uploaded and indexed successfully.",
        files=[Path(path).name for path in saved_files],
        chunk_count=chunk_count,
    )
