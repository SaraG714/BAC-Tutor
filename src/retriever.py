import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "vector_index"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
MIN_SIMILARITY = 0.4

_model = None
_index = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_index():
    global _index
    if _index is None:
        embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
        with open(os.path.join(INDEX_DIR, "texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
        with open(os.path.join(INDEX_DIR, "metadatas.json"), encoding="utf-8") as f:
            metadatas = json.load(f)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        _index = {
            "embeddings": embeddings / np.maximum(norms, 1e-10),
            "texts": texts,
            "metadatas": metadatas,
        }
    return _index


def retrieve(query: str, chat_history: str = "") -> list:
    idx = _get_index()
    q_emb = _get_model().encode(query, show_progress_bar=False)
    q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)

    scores = idx["embeddings"] @ q_norm
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    nodes = []
    for i in top_indices:
        sim = float(scores[i])
        if sim >= MIN_SIMILARITY:
            nodes.append({
                "text": idx["texts"][i],
                "metadata": idx["metadatas"][i],
                "similarity": sim,
            })
    return nodes


def format_context(nodes: list) -> str:
    parts = []
    for node in nodes:
        file_name = node["metadata"].get("file_name", "lectura")
        page = node["metadata"].get("page_label", "?")
        parts.append(f"[{file_name}, p.{page}]\n{node['text'].strip()}")
    return "\n\n---\n\n".join(parts)
