import os
import uuid
from datetime import datetime

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fb;
        color: #172033;
    }
    [data-testid="stSidebar"] {
        background: #101828;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #f8fafc;
    }
    .stButton > button {
        background: #ffffff;
        border: 1px solid #b8c2d1;
        border-radius: 8px;
        color: #172033;
        font-weight: 650;
        min-height: 48px;
    }
    .stButton > button:hover {
        background: #eef4ff;
        border-color: #2563eb;
        color: #0b3b8f;
    }
    .stButton > button p {
        color: inherit;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #273244;
        border-color: #475467;
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #334155;
        border-color: #93c5fd;
        color: #ffffff;
    }
    [data-testid="stFileUploader"] section {
        background: #0b1220;
        border: 1px solid #2f3b52;
        border-radius: 8px;
    }
    .hero {
        padding: 22px 4px 14px;
        border-bottom: 1px solid #d9e0ea;
        margin-bottom: 18px;
    }
    .hero h1 {
        font-size: 34px;
        line-height: 1.15;
        margin: 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #475467;
        font-size: 15px;
        margin-top: 8px;
        max-width: 820px;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 8px 0 18px;
    }
    .metric {
        background: #ffffff;
        border: 1px solid #d9e0ea;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .metric span {
        color: #667085;
        display: block;
        font-size: 12px;
    }
    .metric strong {
        color: #172033;
        display: block;
        font-size: 18px;
        margin-top: 4px;
    }
    .source {
        border-left: 3px solid #2563eb;
        padding-left: 12px;
        margin: 8px 0;
        color: #344054;
    }
    .small-muted {
        color: #667085;
        font-size: 13px;
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 8px;
    }
    @media (max-width: 900px) {
        .metric-row {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .hero h1 {
            font-size: 28px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs):
    response = requests.post(f"{API_BASE_URL}{path}", timeout=90, **kwargs)
    response.raise_for_status()
    return response.json()


def api_delete(path: str):
    response = requests.delete(f"{API_BASE_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def load_documents():
    try:
        return api_get("/documents")
    except requests.RequestException:
        return []


def format_request_error(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        detail = response.json().get("detail")
        if detail:
            return str(detail)
    except ValueError:
        pass
    return str(exc)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


with st.sidebar:
    st.title("Workspace")
    st.caption("Upload documents, rebuild the index, and manage the active conversation.")

    uploads = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md", "docx", "csv", "html", "htm"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Upload and index", use_container_width=True):
        files = [("files", (file.name, file.getvalue(), file.type or "application/octet-stream")) for file in uploads]
        with st.spinner("Indexing documents..."):
            try:
                result = api_post("/documents/upload", files=files)
                if result.get("indexed"):
                    st.success(f"Uploaded and indexed {len(result['uploaded'])} file(s).")
                else:
                    st.warning(
                        f"Uploaded {len(result['uploaded'])} file(s), but indexing did not complete: "
                        f"{result.get('indexing_error', 'unknown indexing error')}"
                    )
            except requests.RequestException as exc:
                st.error(f"Upload failed: {format_request_error(exc)}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Rebuild", use_container_width=True):
            with st.spinner("Rebuilding FAISS index..."):
                try:
                    api_post("/documents/rebuild")
                    st.success("Index rebuilt.")
                except requests.RequestException as exc:
                    st.error(f"Rebuild failed: {format_request_error(exc)}")
    with col_b:
        if st.button("Clear chat", use_container_width=True):
            try:
                api_delete(f"/memory/{st.session_state.session_id}")
            except requests.RequestException:
                pass
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

    st.divider()
    st.subheader("Documents")
    documents = load_documents()
    if documents:
        for document in documents:
            size_kb = max(1, round(document["size_bytes"] / 1024))
            modified = datetime.fromtimestamp(document["modified_at"]).strftime("%b %d, %Y")
            st.markdown(f"**{document['name']}**")
            st.caption(f"{size_kb} KB | {modified}")
    else:
        st.info("No uploaded documents yet.")


st.markdown(
    """
    <div class="hero">
      <h1>Agentic RAG Assistant</h1>
      <p>Ask, compare, summarize, and investigate uploaded documents with memory, citations, and web fallback when the document set is not enough.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    health = api_get("/health")
    backend_status = "Online"
    doc_count = health["document_count"]
    gemini_configured = health.get("gemini_configured", False)
except requests.RequestException:
    backend_status = "Offline"
    doc_count = len(load_documents())
    gemini_configured = False

st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric"><span>Backend</span><strong>{backend_status}</strong></div>
      <div class="metric"><span>Documents</span><strong>{doc_count}</strong></div>
      <div class="metric"><span>Memory</span><strong>{len(st.session_state.messages)} turns</strong></div>
      <div class="metric"><span>Gemini</span><strong>{"Configured" if gemini_configured else "Missing key"}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if backend_status == "Offline":
    st.warning("Start the FastAPI backend with `uvicorn backend.app:app --reload`.")
elif not gemini_configured:
    st.warning("Add `GEMINI_API_KEY` to `.env`, then restart the backend before indexing or asking document questions.")


def render_sources(citations):
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for citation in citations:
            label = f"[{citation['id']}] {citation['title']}"
            if citation.get("page"):
                label += f" | page {citation['page']}"
            if citation.get("url"):
                label += f" | {citation['url']}"
            st.markdown(f"**{label}**")
            st.markdown(f"<div class='source'>{citation.get('snippet', '')}</div>", unsafe_allow_html=True)


def ask(prompt: str):
    with st.spinner("The agent is deciding whether to retrieve, search, summarize, compare, or answer directly..."):
        payload = {"question": prompt, "session_id": st.session_state.session_id}
        result = api_post("/query", json=payload)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "metadata": result,
        }
    )


starter_cols = st.columns(4)
starters = [
    "Summarize the uploaded documents",
    "Compare the key documents",
    "What are the main risks or gaps?",
    "Find current web context for this topic",
]
for column, starter in zip(starter_cols, starters):
    with column:
        if st.button(starter, use_container_width=True):
            st.session_state.pending_prompt = starter


for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        metadata = message.get("metadata")
        if metadata:
            st.caption(
                f"Agent action: {metadata['action']} | confidence: {metadata['confidence']}"
                + (" | web fallback used" if metadata.get("used_web") else "")
            )
            render_sources(metadata.get("citations", []))
            follow_ups = metadata.get("follow_up_questions", [])
            if follow_ups:
                cols = st.columns(len(follow_ups))
                for followup_index, (col, question) in enumerate(zip(cols, follow_ups)):
                    with col:
                        if st.button(question, key=f"followup-{message_index}-{followup_index}"):
                            st.session_state.pending_prompt = question


prompt = st.chat_input("Ask across your documents, request a comparison, or ask for a summary...")
if prompt:
    st.session_state.pending_prompt = prompt

if st.session_state.pending_prompt:
    pending = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    try:
        ask(pending)
    except requests.RequestException as exc:
        st.error(f"Request failed: {format_request_error(exc)}")
    st.rerun()
