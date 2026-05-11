import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_DIR = "vector_index"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5
MIN_SIMILARITY = 0.15  # lower — hybrid score compensates
BM25_WEIGHT = 0.35     # weight for keyword score; (1 - BM25_WEIGHT) for semantic

_model = None
_index = None
_bm25 = None


def _tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_index():
    global _index, _bm25
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
        _bm25 = BM25Okapi([_tokenize(t) for t in texts])
    return _index


def retrieve(query: str, chat_history: str = "") -> list:
    idx = _get_index()

    # ── Semantic scores ───────────────────────────────────────────────────────
    q_emb = _get_model().encode(query, show_progress_bar=False)
    q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)
    sem_scores = idx["embeddings"] @ q_norm

    # ── BM25 scores (normalized 0-1) ──────────────────────────────────────────
    raw_bm25 = np.array(_bm25.get_scores(_tokenize(query)), dtype=float)
    bm25_max = raw_bm25.max()
    bm25_scores = raw_bm25 / bm25_max if bm25_max > 0 else raw_bm25

    # ── Hybrid score ──────────────────────────────────────────────────────────
    hybrid = (1 - BM25_WEIGHT) * sem_scores + BM25_WEIGHT * bm25_scores

    top_indices = np.argsort(hybrid)[::-1][:TOP_K]

    nodes = []
    for i in top_indices:
        if hybrid[i] >= MIN_SIMILARITY:
            nodes.append({
                "text": idx["texts"][i],
                "metadata": idx["metadatas"][i],
                "similarity": float(hybrid[i]),
            })
    return nodes


def format_context(nodes: list) -> str:
    parts = []
    for node in nodes:
        file_name = node["metadata"].get("file_name", "lectura")
        page = node["metadata"].get("page_label", "?")
        parts.append(f"[{file_name}, p.{page}]\n{node['text'].strip()}")
    return "\n\n---\n\n".join(parts)
