import os
import re
import unicodedata
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bac_tutor"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_SEMANTIC = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5
MIN_SIMILARITY = 0.20
RRF_K = 60
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

_model = None
_collection_cache = None
_bm25_index = None
_bm25_corpus = None  # list of {id, text, metadata}
_retrieve_cache: dict = {}  # query → nodes; shared across sessions in the same process
_CACHE_MAX = 256


# ── Lazy loaders ──────────────────────────────────────────────────────────────

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _collection_cache
    if _collection_cache is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection_cache = client.get_collection(COLLECTION_NAME)
    return _collection_cache


def _get_bm25():
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        collection = _get_collection()
        total = collection.count()
        result = collection.get(
            include=["documents", "metadatas"],
            limit=max(total, 1),
        )
        _bm25_corpus = [
            {"id": doc_id, "text": text, "metadata": meta}
            for doc_id, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]
        tokenized = [_tokenize(doc["text"]) for doc in _bm25_corpus]
        _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_corpus


# ── Text utilities ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase + strip accents for BM25 so 'Próspecto' == 'prospecto'."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]{2,}", _normalize(text))


# ── Query preprocessing ───────────────────────────────────────────────────────

_META_TOKENS = {
    "no", "se", "se", "ayudame", "ayuda", "explicame", "explica",
    "eso", "ok", "si", "no", "hmm", "que", "como",
}


def _is_meta(query: str) -> bool:
    tokens = set(_tokenize(query))
    return len(query.strip()) < 25 and tokens.issubset(_META_TOKENS | {""})


def _expand_query(query: str, chat_history: str) -> str:
    """If query is a short meta-message, pull the last substantive student line."""
    if not _is_meta(query) or not chat_history:
        return query
    for line in reversed(chat_history.splitlines()):
        if line.startswith("Estudiante:"):
            content = line.replace("Estudiante:", "").strip()
            if len(content) > 20:
                return content
    return query


# ── Main retrieve ─────────────────────────────────────────────────────────────

def retrieve(query: str, chat_history: str = "") -> list:
    effective_query = _expand_query(query, chat_history)

    if effective_query in _retrieve_cache:
        return _retrieve_cache[effective_query]

    # ── Layer 1: Semantic ──────────────────────────────────────────────────────
    embedding = _get_model().encode(effective_query, show_progress_bar=False).tolist()
    sem_raw = _get_collection().query(
        query_embeddings=[embedding],
        n_results=TOP_K_SEMANTIC,
        include=["documents", "metadatas", "distances"],
    )
    semantic_hits = []
    for doc_id, doc, meta, dist in zip(
        sem_raw["ids"][0],
        sem_raw["documents"][0],
        sem_raw["metadatas"][0],
        sem_raw["distances"][0],
    ):
        sim = 1.0 - dist
        if sim >= MIN_SIMILARITY:
            semantic_hits.append({"id": doc_id, "text": doc, "metadata": meta, "similarity": sim})

    if DEBUG:
        print(f"\n── SEMANTIC (query: '{effective_query}') ──")
        for h in semantic_hits[:3]:
            print(f"  [{h['metadata'].get('file_name','')} p.{h['metadata'].get('page_label','')}] sim={h['similarity']:.3f}")
            print(f"  {h['text'][:120]}…")

    # ── Layer 2: BM25 ──────────────────────────────────────────────────────────
    bm25, corpus = _get_bm25()
    tokens = _tokenize(effective_query)
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_BM25]
    bm25_hits = [
        {"id": corpus[i]["id"], "text": corpus[i]["text"],
         "metadata": corpus[i]["metadata"], "bm25_score": float(scores[i])}
        for i in top_idx if scores[i] > 0
    ]

    if DEBUG:
        print(f"\n── BM25 (tokens: {tokens[:8]}) ──")
        for h in bm25_hits[:3]:
            print(f"  [{h['metadata'].get('file_name','')} p.{h['metadata'].get('page_label','')}] bm25={h['bm25_score']:.3f}")
            print(f"  {h['text'][:120]}…")

    # ── Layer 3: RRF fusion ────────────────────────────────────────────────────
    sem_rank = {h["id"]: rank for rank, h in enumerate(semantic_hits)}
    bm25_rank = {h["id"]: rank for rank, h in enumerate(bm25_hits)}

    # Build unified doc map (semantic takes precedence for shared docs)
    all_docs: dict = {}
    for h in bm25_hits:
        all_docs[h["id"]] = h
    for h in semantic_hits:
        all_docs[h["id"]] = h  # overwrite with richer semantic entry

    all_ids = set(sem_rank) | set(bm25_rank)
    rrf_scores = {
        doc_id: (
            1 / (RRF_K + sem_rank.get(doc_id, TOP_K_SEMANTIC + RRF_K))
            + 1 / (RRF_K + bm25_rank.get(doc_id, TOP_K_BM25 + RRF_K))
        )
        for doc_id in all_ids
    }

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_FINAL]
    final_nodes = [
        {**all_docs[doc_id], "rrf_score": rrf_score}
        for doc_id, rrf_score in ranked
        if doc_id in all_docs
    ]

    if DEBUG:
        print(f"\n── RRF FINAL ──")
        for node in final_nodes[:3]:
            print(f"  [{node['metadata'].get('file_name','')} p.{node['metadata'].get('page_label','')}]"
                  f"  rrf={node['rrf_score']:.4f}  sim={node.get('similarity', 0):.3f}")
            print(f"  {node['text'][:120]}…")

    if len(_retrieve_cache) >= _CACHE_MAX:
        _retrieve_cache.pop(next(iter(_retrieve_cache)))
    _retrieve_cache[effective_query] = final_nodes

    return final_nodes


# ── Format for prompt ─────────────────────────────────────────────────────────

def format_context(nodes: list) -> str:
    parts = []
    for node in nodes:
        file_name = node["metadata"].get("file_name", "lectura")
        page = node["metadata"].get("page_label", "?")
        parts.append(f"[{file_name}, p.{page}]\n{node['text'].strip()}")
    return "\n\n---\n\n".join(parts)
