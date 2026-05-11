"""
Run once to index PDFs from docs/ into ChromaDB.
Usage: python -m src.indexer
"""
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bac_tutor"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_STOPWORDS = {
    "para", "como", "esto", "este", "esta", "pero", "donde", "cuando",
    "tiene", "puede", "sobre", "entre", "hasta", "desde", "durante",
    "después", "antes", "también", "aunque", "porque", "través", "parte",
    "todos", "todas", "cada", "otros", "otras", "mismo", "misma",
}

_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"Cargando modelo de embeddings '{EMBED_MODEL}'...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _extract_keywords(text: str, max_kw: int = 15) -> str:
    """Extract main nouns/terms (>5 chars, no stopwords) for debug metadata."""
    words = re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{5,}", text)
    seen: set = set()
    keywords = []
    for w in words:
        wl = w.lower()
        if wl not in _STOPWORDS and wl not in seen:
            seen.add(wl)
            keywords.append(wl)
        if len(keywords) >= max_kw:
            break
    return " ".join(keywords)


def build_index():
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        raise FileNotFoundError(
            f"No se encontraron PDFs en '{DOCS_DIR}/'. "
            "Agrega las lecturas del curso antes de indexar."
        )

    print("Cargando documentos...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"  {len(documents)} páginas cargadas.")

    parser = SentenceSplitter(chunk_size=512, chunk_overlap=80)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"  {len(nodes)} chunks generados.")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Generando embeddings (local, sin límites de API)...")
    texts = [n.text for n in nodes]
    ids = [n.node_id for n in nodes]
    metadatas = [
        {
            "file_name": n.metadata.get("file_name", ""),
            "page_label": str(n.metadata.get("page_label", "")),
            "keywords": _extract_keywords(n.text),
        }
        for n in nodes
    ]

    embeddings = _get_model().encode(
        texts, batch_size=32, show_progress_bar=True
    ).tolist()

    batch_size = 100
    for i in range(0, len(nodes), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"  {min(i + batch_size, len(nodes))}/{len(nodes)} chunks guardados...")

    print(f"Indexación completa. Vectores guardados en '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    build_index()
