"""
Run once to index PDFs into a numpy vector store (no chromadb).
Usage: python -m src.indexer
"""
import os
import io
import json
import base64
import time
import numpy as np
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
import fitz  # pymupdf

load_dotenv()

DOCS_DIR = "docs"
INDEX_DIR = "vector_index"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MIN_IMAGE_SIZE = 8000  # bytes — ignora iconos y decoraciones pequeñas
IMAGE_CACHE_FILE = os.path.join(INDEX_DIR, "image_cache.json")


def _describe_image(client: Groq, img_bytes: bytes, file_name: str, page: int):
    """Sends an image to Groq Vision and returns a description in Spanish."""
    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Eres un asistente académico para el curso ISIS-2403 "
                                "Arquitectura Empresarial (Universidad de los Andes). "
                                "Describe esta figura del material del curso en español. "
                                "Explica qué muestra, qué conceptos del metamodelo BAC "
                                "aparecen (actores, productos, procesos, relaciones, etc.) "
                                "y qué relaciones o estructuras se pueden observar. "
                                "Sé preciso y usa el vocabulario del curso. Máximo 150 palabras."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  Error describiendo imagen (p.{page}, {file_name}): {e}")
        return None


def _page_has_visual_content(page) -> bool:
    """Returns True if the page has drawings, images, or non-trivial graphics."""
    # Check for raster images
    if page.get_images(full=True):
        return True
    # Check for vector drawings (paths with fill/stroke)
    drawings = page.get_drawings()
    if len(drawings) > 5:
        return True
    return False


def _load_image_cache() -> dict:
    """Loads previously described pages to avoid re-processing them."""
    if os.path.exists(IMAGE_CACHE_FILE):
        with open(IMAGE_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_image_cache(cache: dict):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(IMAGE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _extract_image_chunks(docs_dir: str, client: Groq) -> tuple[list, list]:
    """Renders pages with visual content and describes them with Groq Vision.
    Uses a cache so re-runs only process pages that failed or are new.
    """
    cache = _load_image_cache()
    texts, metadatas = [], []
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    new_entries = 0

    for pdf_file in pdf_files:
        path = os.path.join(docs_dir, pdf_file)
        print(f"  Escaneando páginas de {pdf_file}...")
        doc = fitz.open(path)
        img_count = 0

        for page_num, page in enumerate(doc, start=1):
            if not _page_has_visual_content(page):
                continue

            cache_key = f"{pdf_file}::p{page_num}"

            # Use cached description if available
            if cache_key in cache:
                texts.append(cache[cache_key]["text"])
                metadatas.append(cache[cache_key]["metadata"])
                img_count += 1
                continue

            # Render page as PNG at 150 DPI
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            if len(img_bytes) < MIN_IMAGE_SIZE:
                continue

            print(f"    p.{page_num} ({len(img_bytes) // 1024} KB) → describiendo con visión...")
            description = _describe_image(client, img_bytes, pdf_file, page_num)

            if description:
                text = f"[Figura en p.{page_num}] {description}"
                meta = {
                    "file_name": pdf_file,
                    "page_label": str(page_num),
                    "type": "image",
                }
                texts.append(text)
                metadatas.append(meta)
                cache[cache_key] = {"text": text, "metadata": meta}
                new_entries += 1
                img_count += 1
                time.sleep(1.5)  # avoid rate limits

        print(f"    {img_count} páginas con figuras ({new_entries} nuevas) de {pdf_file}.")
        doc.close()

    if new_entries > 0:
        _save_image_cache(cache)
        print(f"  Cache actualizado con {new_entries} entradas nuevas.")
    else:
        print("  Todo cargado desde cache — sin llamadas a la API.")

    return texts, metadatas


def build_index():
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        raise FileNotFoundError(
            f"No se encontraron PDFs en '{DOCS_DIR}/'. "
            "Agrega las lecturas del curso antes de indexar."
        )

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("Falta GROQ_API_KEY en el entorno.")
    client = Groq(api_key=groq_key)

    # ── Text chunks ───────────────────────────────────────────────────────────
    print("Cargando documentos...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"  {len(documents)} páginas cargadas.")

    parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"  {len(nodes)} chunks de texto generados.")

    texts = [n.text for n in nodes]
    metadatas = [
        {
            "file_name": n.metadata.get("file_name", ""),
            "page_label": str(n.metadata.get("page_label", "")),
            "type": "text",
        }
        for n in nodes
    ]

    # ── Image chunks ──────────────────────────────────────────────────────────
    print("Procesando imágenes con Groq Vision...")
    img_texts, img_metadatas = _extract_image_chunks(DOCS_DIR, client)
    print(f"  {len(img_texts)} chunks de imagen generados.")

    texts += img_texts
    metadatas += img_metadatas

    # ── Embeddings ────────────────────────────────────────────────────────────
    print(f"Cargando modelo '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Generando embeddings...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(INDEX_DIR, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    with open(os.path.join(INDEX_DIR, "metadatas.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)

    print(f"\nIndexación completa. {len(texts)} chunks totales guardados en '{INDEX_DIR}/'.")


if __name__ == "__main__":
    build_index()
