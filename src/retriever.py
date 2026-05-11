import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bac_tutor"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
MIN_SIMILARITY = 0.4  # cosine similarity; below this the question is likely out of scope

_model = None
_collection_cache = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _collection_cache
    if _collection_cache is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection_cache = chroma_client.get_collection(COLLECTION_NAME)
    return _collection_cache


def retrieve(query: str) -> list:
    embedding = _get_model().encode(query, show_progress_bar=False).tolist()

    results = _get_collection().query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    nodes = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1.0 - dist  # cosine distance → similarity
        if similarity >= MIN_SIMILARITY:
            nodes.append({"text": doc, "metadata": meta, "similarity": similarity})
    return nodes


def format_context(nodes: list) -> str:
    parts = []
    for node in nodes:
        file_name = node["metadata"].get("file_name", "lectura")
        page = node["metadata"].get("page_label", "?")
        parts.append(f"[{file_name}, p.{page}]\n{node['text'].strip()}")
    return "\n\n---\n\n".join(parts)
