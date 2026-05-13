from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
import uuid
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
from src.backend.history_store import add_entry, get_all_history, get_by_id, clear

import traceback

DATA_DIR = Path("data/uploads")
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

class DocumentFilter(BaseModel):
    sources: list[str] | None = None        
    file_type: str | None = None           
    date_from: str | None = None            
    date_to: str | None = None 

class Question(BaseModel):
    question: str
    filter: DocumentFilter | None = None


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
    total_indexed: int = 0 # tong so file da duoc index

class HistoryEntry(BaseModel):
    question: str
    answer: str
    timestamp: str
    sources: list[SourceItem] = Field(default_factory=list)  # as default =[list]
    meta: dict = Field(default_factory=dict)

class HistoryResponse(BaseModel):
    total: int
    entries: list[HistoryEntry] = Field(default_factory=list)

def build_rag_service(vector_store, llm, prompt_builder) -> RagService:
    documents = list(vector_store.docstore._dict.values())
    semantic_retriever = SemanticRetriever(vector_store)
    keyword_retriever = KeywordRetriever(documents)
    retriever = HybridRetriever(semantic_retriever, keyword_retriever)
    return RagService(llm, prompt_builder, retriever)

class DocumentInfo(BaseModel):
    name: str
    original_name: str
    file_type: str
    size_kb: float
    date_uploaded: str


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentInfo]


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
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DATA_DIR / f"{stem}_{timestamp}_{unique_id}{suffix}"



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
        print(f"Startup: loaded index OK")
    except Exception as exc:
        print(f"Startup warning: {exc}")

    yield
    print("Shutting down...")


app = FastAPI(
    title="Multi-Document RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    upload_schema = (
        openapi_schema.get("components", {})
        .get("schemas", {})
        .get("Body_upload_upload_post")
    )
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
        result = await loop.run_in_executor(
            None,
            app.state.rag_service.answer,
            q.question,
            q.filter)
        add_entry (
            question = result["question"],
            ans = result["answer"],
            sources = result["sources"],
            meta = result["meta"]
        
        )
        
        
        return result
    except Exception as exc:
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=str(exc)) 

@app.get("/history", response_model=HistoryResponse)
def get_history():
    # get toan bo lich su
    entries = get_all_history()
    return HistoryResponse(total=len(entries), entries=entries)

@app.get("/history/{entry_id}", response_model=HistoryEntry)
def get_history_entry(entry_id: int):
    entry = get_by_id(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No history entry with id {entry_id}")
    return entry
@app.delete("/history")
def clear_history():
    clear()
    return {"message": "Chat history cleared."}

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
        raise HTTPException(400, "Không có file nào được gửi lên.")

    saved: list[str] = []
    skipped: list[str] = []

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()

        # Validate extension
        if suffix not in SUPPORTED_EXTENSIONS:
            skipped.append(f"{file.filename} (định dạng không hỗ trợ)")
            continue

        # Read file content
        content = await file.read()

        # Lưu file
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest = resolve_upload_path(file.filename)
        dest.write_bytes(content)
        saved.append(str(dest))

    if not saved:
        raise HTTPException(
            400,
            f"Không có file hợp lệ nào được upload. Bỏ qua: {skipped}",
        )

    # Reindex toàn bộ — file cũ + file mới vừa upload
    async with app.state.index_lock:
        all_files = [str(p) for p in collect_uploaded_files()]
        loop = asyncio.get_running_loop()
        _, rag_service, chunk_count = await loop.run_in_executor(
            None,
            rebuild_index,
            all_files,
            app.state.llm,
            app.state.prompt_builder,
        )
        app.state.rag_service = rag_service

    all_indexed = collect_uploaded_files()

    return UploadResponse(
        message=f"Upload thành công {len(saved)} file"
        + (f", bỏ qua {len(skipped)} file" if skipped else ""),
        files=[Path(p).name for p in saved],
        skipped=skipped,
        chunk_count=chunk_count,
        total_indexed=len(all_indexed),
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """
    Liệt kê tất cả file đã upload.
    Dùng field 'name' để truyền vào filter.sources khi /ask.
    """
    files = collect_uploaded_files()
    docs = []
    for f in files:
        # Lấy original name: bỏ timestamp và uuid suffix
        # Tên file: {stem}_{timestamp}_{uuid}{suffix}
        parts = f.stem.rsplit("_", 2)
        original_stem = parts[0] if len(parts) == 3 else f.stem
        docs.append(DocumentInfo(
            name=f.name,
            original_name=original_stem + f.suffix,
            file_type=f.suffix.lstrip("."),
            size_kb=round(f.stat().st_size / 1024, 1),
            date_uploaded=datetime.fromtimestamp(
                f.stat().st_mtime
            ).isoformat(),
        ))
    return DocumentListResponse(total=len(docs), documents=docs)


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Xóa 1 file theo tên và reindex lại."""
    target = DATA_DIR / filename
    if not target.exists():
        raise HTTPException(404, f"Không tìm thấy file '{filename}'.")

    target.unlink()

    async with app.state.index_lock:
        remaining = [str(p) for p in collect_uploaded_files()]
        if remaining:
            loop = asyncio.get_running_loop()
            _, rag_service, chunk_count = await loop.run_in_executor(
                None,
                rebuild_index,
                remaining,
                app.state.llm,
                app.state.prompt_builder,
            )
            app.state.rag_service = rag_service
        else:
            # Không còn file nào → xóa index
            app.state.rag_service = None
            if INDEX_DIR.exists():
                import shutil
                shutil.rmtree(INDEX_DIR)
            chunk_count = 0

    return {
        "message": f"Đã xóa '{filename}' và reindex.",
        "remaining_files": len(collect_uploaded_files()),
        "chunk_count": chunk_count,
    }


