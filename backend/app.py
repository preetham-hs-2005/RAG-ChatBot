import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rag_pipeline import (
    AgentAction,
    ask_agent,
    build_or_load_index,
    list_documents,
    reset_memory,
    save_uploaded_file,
)
from .config import GEMINI_API_KEY


app = FastAPI(
    title="Agentic RAG Assistant",
    description="Enterprise-ready RAG API with agent routing, memory, citations, summaries, comparison, and web fallback.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = "default"


class CitationResponse(BaseModel):
    id: int
    title: str
    source_type: str
    snippet: str
    page: Optional[str] = None
    url: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    action: AgentAction
    confidence: str
    citations: List[CitationResponse]
    follow_up_questions: List[str]
    used_web: bool


class DocumentResponse(BaseModel):
    name: str
    path: str
    size_bytes: int
    modified_at: float


class UploadResponse(BaseModel):
    uploaded: List[str]
    document_count: int
    indexed: bool
    indexing_error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    document_count: int
    gemini_configured: bool


@app.on_event("startup")
def startup() -> None:
    try:
        build_or_load_index(force_rebuild=False)
    except Exception:
        pass


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        document_count=len(list_documents()),
        gemini_configured=bool(GEMINI_API_KEY),
    )


@app.get("/documents", response_model=List[DocumentResponse])
def documents() -> List[DocumentResponse]:
    return [DocumentResponse(**document) for document in list_documents()]


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    uploaded: List[str] = []
    allowed_extensions = {".pdf", ".txt", ".md", ".docx", ".csv", ".html", ".htm"}

    for upload in files:
        _, extension = os.path.splitext(upload.filename or "")
        if extension.lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.filename}")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        try:
            destination = save_uploaded_file(upload.filename or "document", tmp_path)
            uploaded.append(os.path.basename(destination))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    try:
        build_or_load_index(force_rebuild=True)
    except Exception as exc:
        return UploadResponse(
            uploaded=uploaded,
            document_count=len(list_documents()),
            indexed=False,
            indexing_error=str(exc),
        )

    return UploadResponse(uploaded=uploaded, document_count=len(list_documents()), indexed=True)


@app.post("/documents/rebuild", response_model=HealthResponse)
def rebuild_index() -> HealthResponse:
    try:
        build_or_load_index(force_rebuild=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return HealthResponse(
        status="rebuilt",
        document_count=len(list_documents()),
        gemini_configured=bool(GEMINI_API_KEY),
    )


@app.post("/query", response_model=QueryResponse)
def ask_question(req: QueryRequest) -> QueryResponse:
    try:
        result = ask_agent(req.question, session_id=req.session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        action=result.action,
        confidence=result.confidence,
        citations=[CitationResponse(**citation.__dict__) for citation in result.citations],
        follow_up_questions=result.follow_up_questions,
        used_web=result.used_web,
    )


@app.delete("/memory/{session_id}", response_model=HealthResponse)
def clear_memory(session_id: str) -> HealthResponse:
    reset_memory(session_id)
    return HealthResponse(
        status="memory_cleared",
        document_count=len(list_documents()),
        gemini_configured=bool(GEMINI_API_KEY),
    )
