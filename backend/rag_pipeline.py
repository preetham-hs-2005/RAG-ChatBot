import os
from typing import Optional

import faiss
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from .config import GEMINI_API_KEY, DATA_DIR, INDEX_DIR

# -------------------------------
# Configure Gemini globally
# -------------------------------
def configure_gemini():
    Settings.llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    Settings.embed_model = GeminiEmbedding(
        model="models/embedding-001",
        api_key=GEMINI_API_KEY,
    )


# -------------------------------
# Build or Load FAISS Index
# -------------------------------
def build_or_load_index(force_rebuild: bool = False) -> VectorStoreIndex:
    configure_gemini()

    if not force_rebuild and os.path.exists(INDEX_DIR):
        try:
            vector_store = FaissVectorStore.from_persist_dir(INDEX_DIR)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=INDEX_DIR,
            )
            return load_index_from_storage(storage_context)
        except Exception:
            pass

    # Load documents
    documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()

    # Create FAISS index
    embed_dim = len(Settings.embed_model.get_text_embedding("test"))
    faiss_index = faiss.IndexFlatL2(embed_dim)

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=INDEX_DIR,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    storage_context.persist()
    return index


# -------------------------------
# Singleton Index
# -------------------------------
_index: Optional[VectorStoreIndex] = None

def get_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        _index = build_or_load_index()
    return _index


# -------------------------------
# Query Function
# -------------------------------
def query_notes(question: str) -> str:
    index = get_index()
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )
    response = query_engine.query(question)
    return str(response)
