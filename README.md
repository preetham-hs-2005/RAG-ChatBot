# Agentic RAG Assistant

Lightweight enterprise RAG workspace for uploaded documents. It combines a FastAPI backend, Gemini, FAISS, LlamaIndex, conversation memory, source citations, document summarization, document comparison, and web-search fallback.

## Capabilities

- Multi-document upload and indexing
- Agent routing for direct answers, document retrieval, summarization, comparison, and web fallback
- Conversation memory per chat session
- Source citations with snippets, pages when available, and web URLs
- Follow-up question suggestions
- Modern Streamlit workspace UI
- Production-ready API shape with health, upload, rebuild, query, documents, and memory endpoints

## Setup

Create a `.env` file:

```env
GEMINI_API_KEY=your_key_here
WEB_SEARCH_ENABLED=true
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the backend:

```powershell
uvicorn backend.app:app --reload
```

Run the UI in another terminal:

```powershell
streamlit run app.py
```

## API

- `GET /health` checks service status and document count.
- `GET /documents` lists uploaded documents.
- `POST /documents/upload` uploads and indexes documents.
- `POST /documents/rebuild` rebuilds the FAISS index.
- `POST /query` asks the agent a question.
- `DELETE /memory/{session_id}` clears conversation memory.

## Notes

Uploaded files are stored in `data/`, and FAISS/LlamaIndex storage is persisted in `storage/`. Both directories are created automatically.
