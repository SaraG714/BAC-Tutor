import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bac_tutor"
EMBED_MODEL = "gemini-embedding-001"
TOP_K = 3

_client = None
_collection_cache = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def _get_collection():
    global _collection_cache
    if _collection_cache is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection_cache = chroma_client.get_collection(COLLECTION_NAME)
    return _collection_cache


def retrieve(query: str) -> list:
    embedding = _get_client().models.embed_content(
        model=EMBED_MODEL,
        contents=query,
    ).embeddings[0].values

    results = _get_collection().query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    nodes = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        nodes.append({"text": doc, "metadata": meta})
    return nodes


def format_context(nodes: list) -> str:
    parts = []
    for node in nodes:
        file_name = node["metadata"].get("file_name", "lectura")
        page = node["metadata"].get("page_label", "?")
        parts.append(f"[{file_name}, p.{page}]\n{node['text'].strip()}")
    return "\n\n---\n\n".join(parts)
