"""
Run once to index PDFs from docs/ into ChromaDB.
Usage: python -m src.indexer
"""
import os
import time
import chromadb
from google import genai
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bac_tutor"
EMBED_MODEL = "gemini-embedding-001"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def get_embedding(text: str) -> list:
    result = _get_client().models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


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

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME)

    print("Generando embeddings y guardando en ChromaDB...")
    batch_size = 20
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        texts = [n.text for n in batch]
        ids = [n.node_id for n in batch]
        metadatas = [
            {
                "file_name": n.metadata.get("file_name", ""),
                "page_label": str(n.metadata.get("page_label", "")),
            }
            for n in batch
        ]
        embeddings = [get_embedding(t) for t in texts]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        done = min(i + batch_size, len(nodes))
        print(f"  {done}/{len(nodes)} chunks indexados...")
        if done < len(nodes):
            time.sleep(15)  # stay under 100 req/min free tier

    print(f"Indexación completa. Vectores guardados en '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    build_index()
