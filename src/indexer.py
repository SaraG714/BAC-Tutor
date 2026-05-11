"""
Run once to index PDFs into a numpy vector store (no chromadb).
Usage: python -m src.indexer
"""
import os
import json
import numpy as np
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

DOCS_DIR = "docs"
INDEX_DIR = "vector_index"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def build_index():
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        raise FileNotFoundError(
            f"No se encontraron PDFs en '{DOCS_DIR}/'. "
            "Agrega las lecturas del curso antes de indexar."
        )

    print("Cargando documentos...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"  {len(documents)} páginas cargadas.")

    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"  {len(nodes)} chunks generados.")

    print(f"Cargando modelo '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [n.text for n in nodes]
    metadatas = [
        {
            "file_name": n.metadata.get("file_name", ""),
            "page_label": str(n.metadata.get("page_label", "")),
        }
        for n in nodes
    ]

    print("Generando embeddings...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(INDEX_DIR, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    with open(os.path.join(INDEX_DIR, "metadatas.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)

    print(f"Indexación completa. Índice guardado en '{INDEX_DIR}/'.")


if __name__ == "__main__":
    build_index()
