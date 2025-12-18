from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .rag_pipeline import query_notes, build_or_load_index

app = FastAPI(title="Gemini Course Notes RAG Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.on_event("startup")
def startup():
    build_or_load_index(force_rebuild=False)

@app.post("/query", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    answer = query_notes(req.question)
    return QueryResponse(answer=answer)
