import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import faiss
import requests
from bs4 import BeautifulSoup
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore

from .config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    INDEX_DIR,
    LLM_MODEL,
    WEB_SEARCH_ENABLED,
)


class AgentAction(str, Enum):
    DIRECT = "direct_answer"
    RETRIEVE = "retrieve_documents"
    WEB = "web_search"
    SUMMARIZE = "summarize_documents"
    COMPARE = "compare_documents"


@dataclass
class Citation:
    id: int
    title: str
    source_type: str
    snippet: str
    page: Optional[str] = None
    url: Optional[str] = None
    score: Optional[float] = None


@dataclass
class AgentResult:
    answer: str
    action: AgentAction
    confidence: str
    citations: List[Citation]
    follow_up_questions: List[str]
    used_web: bool = False


_index: Optional[VectorStoreIndex] = None
_memory: Dict[str, List[Dict[str, str]]] = {}


def configure_gemini() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured. Add it to .env before querying.")

    Settings.llm = Gemini(
        model=LLM_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0.2,
    )
    Settings.embed_model = GeminiEmbedding(
        model=EMBEDDING_MODEL,
        api_key=GEMINI_API_KEY,
    )


def list_documents() -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for root, _, files in os.walk(DATA_DIR):
        for file_name in files:
            path = os.path.join(root, file_name)
            rel_path = os.path.relpath(path, DATA_DIR)
            documents.append(
                {
                    "name": file_name,
                    "path": rel_path,
                    "size_bytes": os.path.getsize(path),
                    "modified_at": os.path.getmtime(path),
                }
            )
    return sorted(documents, key=lambda item: item["name"].lower())


def save_uploaded_file(file_name: str, source_path: str) -> str:
    safe_name = os.path.basename(file_name)
    destination = os.path.join(DATA_DIR, safe_name)
    shutil.copyfile(source_path, destination)
    return destination


def build_or_load_index(force_rebuild: bool = False) -> Optional[VectorStoreIndex]:
    global _index

    if not list_documents():
        _index = None
        return None

    configure_gemini()

    if not force_rebuild and os.path.exists(os.path.join(INDEX_DIR, "docstore.json")):
        try:
            vector_store = FaissVectorStore.from_persist_dir(INDEX_DIR)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=INDEX_DIR,
            )
            _index = load_index_from_storage(storage_context)
            return _index
        except Exception:
            pass

    documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
    embed_dim = len(Settings.embed_model.get_text_embedding("dimension probe"))
    faiss_index = faiss.IndexFlatL2(embed_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=INDEX_DIR,
    )
    _index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    storage_context.persist()
    return _index


def get_index() -> Optional[VectorStoreIndex]:
    global _index
    if _index is None:
        _index = build_or_load_index()
    return _index


def reset_memory(session_id: str) -> None:
    _memory.pop(session_id, None)


def _remember(session_id: str, role: str, content: str) -> None:
    turns = _memory.setdefault(session_id, [])
    turns.append({"role": role, "content": content})
    del turns[:-12]


def _memory_context(session_id: str) -> str:
    turns = _memory.get(session_id, [])[-8:]
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)


def _plan(question: str, session_id: str, has_documents: bool) -> AgentAction:
    text = question.lower()
    recent_context = _memory_context(session_id).lower()
    if any(word in text for word in ["compare", "contrast", "difference", "versus", " vs "]):
        return AgentAction.COMPARE
    if any(word in text for word in ["summarize", "summary", "brief", "tldr", "tl;dr"]):
        return AgentAction.SUMMARIZE
    if any(word in text for word in ["latest", "current", "today", "recent", "web", "internet", "search online"]):
        return AgentAction.WEB
    if not has_documents:
        return AgentAction.WEB if WEB_SEARCH_ENABLED else AgentAction.DIRECT
    if len(text.split()) <= 3 and any(greeting in text for greeting in ["hi", "hello", "thanks", "thank you"]):
        return AgentAction.DIRECT
    if any(ref in text for ref in ["it", "that", "they", "those", "this"]) and recent_context:
        return AgentAction.RETRIEVE
    return AgentAction.RETRIEVE


def _complete(prompt: str) -> str:
    response = Settings.llm.complete(prompt)
    return str(response).strip()


def _citation_from_node(index: int, node: Any) -> Citation:
    metadata = getattr(node.node, "metadata", {}) or {}
    snippet = getattr(node.node, "text", "") or ""
    score = getattr(node, "score", None)
    title = metadata.get("file_name") or metadata.get("filename") or metadata.get("source") or "Uploaded document"
    page = metadata.get("page_label") or metadata.get("page_number")
    return Citation(
        id=index,
        title=str(title),
        source_type="document",
        snippet=snippet[:700],
        page=str(page) if page is not None else None,
        score=float(score) if score is not None else None,
    )


def _retrieve_context(question: str, top_k: int = 8) -> Tuple[str, List[Citation]]:
    index = get_index()
    if index is None:
        return "", []
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    citations = [_citation_from_node(i + 1, node) for i, node in enumerate(nodes)]
    context = "\n\n".join(
        f"[{citation.id}] {citation.title}"
        f"{' page ' + citation.page if citation.page else ''}\n{citation.snippet}"
        for citation in citations
    )
    return context, citations


def _answer_from_documents(question: str, session_id: str) -> AgentResult:
    context, citations = _retrieve_context(question)
    if not citations and WEB_SEARCH_ENABLED:
        return _answer_from_web(question, session_id)
    prompt = f"""
You are an enterprise Agentic RAG Assistant. Answer using only the document context.
Use concise, executive-ready language. Cite claims inline with bracket citations like [1].
If the documents do not contain the answer, say what is missing and suggest a useful next step.

Conversation memory:
{_memory_context(session_id)}

Document context:
{context}

Question: {question}
Answer:
"""
    answer = _complete(prompt)
    return AgentResult(
        answer=answer,
        action=AgentAction.RETRIEVE,
        confidence="high" if citations else "low",
        citations=citations,
        follow_up_questions=_follow_ups(question, answer, citations),
    )


def _summarize_documents(question: str, session_id: str) -> AgentResult:
    context, citations = _retrieve_context(question, top_k=12)
    prompt = f"""
Create a structured summary from the uploaded documents. Include:
- key points
- important entities, decisions, or dates
- risks, gaps, or ambiguities
- a short executive takeaway

Use bracket citations tied to the provided source IDs.

Conversation memory:
{_memory_context(session_id)}

Document context:
{context}

User request: {question}
Summary:
"""
    answer = _complete(prompt)
    return AgentResult(
        answer=answer,
        action=AgentAction.SUMMARIZE,
        confidence="high" if citations else "low",
        citations=citations,
        follow_up_questions=_follow_ups(question, answer, citations),
    )


def _compare_documents(question: str, session_id: str) -> AgentResult:
    context, citations = _retrieve_context(question, top_k=14)
    prompt = f"""
Compare the relevant uploaded documents or sections. Prefer a clear table when useful.
Call out similarities, differences, contradictions, missing information, and the likely reason
the differences matter. Use bracket citations for every specific claim.

Conversation memory:
{_memory_context(session_id)}

Document context:
{context}

Comparison request: {question}
Comparison:
"""
    answer = _complete(prompt)
    return AgentResult(
        answer=answer,
        action=AgentAction.COMPARE,
        confidence="medium" if citations else "low",
        citations=citations,
        follow_up_questions=_follow_ups(question, answer, citations),
    )


def _web_search(query: str, max_results: int = 5) -> List[Citation]:
    if not WEB_SEARCH_ENABLED:
        return []
    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "AgenticRAGAssistant/1.0"},
            timeout=8,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: List[Citation] = []
    for result in soup.select(".result")[:max_results]:
        link = result.select_one(".result__a")
        snippet = result.select_one(".result__snippet")
        if not link:
            continue
        results.append(
            Citation(
                id=len(results) + 1,
                title=link.get_text(" ", strip=True),
                source_type="web",
                snippet=snippet.get_text(" ", strip=True) if snippet else "",
                url=link.get("href"),
            )
        )
    return results


def _answer_from_web(question: str, session_id: str) -> AgentResult:
    citations = _web_search(question)
    context = "\n\n".join(
        f"[{citation.id}] {citation.title}\n{citation.url}\n{citation.snippet}"
        for citation in citations
    )
    prompt = f"""
Answer the question using the web search snippets below. Be explicit that web search was used.
Use bracket citations like [1]. If snippets are weak, say so and avoid overclaiming.

Conversation memory:
{_memory_context(session_id)}

Web context:
{context}

Question: {question}
Answer:
"""
    answer = _complete(prompt) if citations else "I could not find enough document or web context to answer confidently."
    return AgentResult(
        answer=answer,
        action=AgentAction.WEB,
        confidence="medium" if citations else "low",
        citations=citations,
        follow_up_questions=_follow_ups(question, answer, citations),
        used_web=True,
    )


def _direct_answer(question: str, session_id: str) -> AgentResult:
    prompt = f"""
You are a concise enterprise AI assistant. Use the conversation memory if relevant.
If the user asks for facts that require uploaded documents or current web context, say what you need.

Conversation memory:
{_memory_context(session_id)}

Question: {question}
Answer:
"""
    answer = _complete(prompt)
    return AgentResult(
        answer=answer,
        action=AgentAction.DIRECT,
        confidence="medium",
        citations=[],
        follow_up_questions=_follow_ups(question, answer, []),
    )


def _follow_ups(question: str, answer: str, citations: List[Citation]) -> List[str]:
    if not citations:
        return [
            "Which documents should I use for this?",
            "Should I search the web for supporting context?",
            "Would you like a concise summary instead?",
        ]
    prompt = f"""
Based on the user question and answer, suggest exactly three short follow-up questions.
Keep each under 14 words and make them useful for document analysis.

Question: {question}
Answer: {answer[:1600]}
Follow-up questions:
"""
    raw = _complete(prompt)
    questions = []
    for line in raw.splitlines():
        cleaned = line.strip(" -0123456789.").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:3] or [
        "Can you summarize the key evidence?",
        "What are the main risks or gaps?",
        "How do these documents compare?",
    ]


def ask_agent(question: str, session_id: str = "default") -> AgentResult:
    configure_gemini()
    _remember(session_id, "user", question)
    action = _plan(question, session_id, has_documents=bool(list_documents()))

    if action == AgentAction.COMPARE:
        result = _compare_documents(question, session_id)
    elif action == AgentAction.SUMMARIZE:
        result = _summarize_documents(question, session_id)
    elif action == AgentAction.WEB:
        result = _answer_from_web(question, session_id)
    elif action == AgentAction.DIRECT:
        result = _direct_answer(question, session_id)
    else:
        result = _answer_from_documents(question, session_id)

    _remember(session_id, "assistant", result.answer)
    return result


def query_notes(question: str) -> str:
    return ask_agent(question).answer
