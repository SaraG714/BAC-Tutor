"""
Run once to index PDFs from docs/ into ChromaDB.
Usage: python -m src.indexer [--skip-images]
"""
import os
import re
import base64
import argparse
from io import BytesIO
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader
from groq import Groq

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

_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_VISION_DESCRIPTION_PROMPT = (
    "Eres un asistente que analiza diagramas de un libro de "
    "Arquitectura Empresarial. Describe este diagrama en texto "
    "plano de forma exhaustiva: qué elementos aparecen, cómo "
    "se relacionan, qué conceptos del metamodelo BAC muestra, "
    "y cualquier label o texto visible. Sé preciso y técnico."
)

_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"Cargando modelo de embeddings '{EMBED_MODEL}'...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _describe_image(image_b64: str, groq_client: Groq) -> str:
    response = groq_client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": _VISION_DESCRIPTION_PROMPT},
                ],
            }
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _extract_image_chunks(docs_dir: str, groq_client: Groq) -> tuple:
    """Return (texts, ids, metadatas) for all image descriptions found in PDFs."""
    from PIL import Image  # noqa: PLC0415

    pdf_files = sorted(
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if f.lower().endswith(".pdf")
    )

    total_images = 0
    for p in pdf_files:
        _r = PdfReader(p)
        total_images += sum(len(pg.images) for pg in _r.pages)

    if total_images == 0:
        print("  No se encontraron imágenes en los PDFs.")
        return [], [], []

    print(f"  {total_images} imágenes encontradas.")

    texts, ids, metadatas = [], [], []
    current = 0

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, 1):
            for img_idx, img_obj in enumerate(page.images):
                current += 1
                print(f"  Describiendo imagen {current} de {total_images} "
                      f"({pdf_name}, p.{page_num})...")
                try:
                    pil_img = Image.open(BytesIO(img_obj.data))
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    description = _describe_image(b64, groq_client)
                    texts.append(description)
                    ids.append(f"{pdf_name}_p{page_num}_img{img_idx}")
                    metadatas.append({
                        "file_name": pdf_name,
                        "page_label": str(page_num),
                        "type": "image",
                        "description": "imagen extraída automáticamente",
                        "keywords": _extract_keywords(description),
                    })
                except Exception as exc:
                    print(f"  ⚠️  Error en imagen {current}: {exc}")

    return texts, ids, metadatas


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


def build_index(skip_images: bool = False):
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

    # ── Image descriptions ────────────────────────────────────────────────────
    if not skip_images:
        print("\nExtrayendo e indexando imágenes del material...")
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        img_texts, img_ids, img_metas = _extract_image_chunks(DOCS_DIR, groq_client)

        if img_texts:
            print(f"  Generando embeddings para {len(img_texts)} descripciones de imágenes...")
            img_embeddings = _get_model().encode(
                img_texts, batch_size=32, show_progress_bar=True
            ).tolist()
            for i in range(0, len(img_texts), batch_size):
                collection.add(
                    ids=img_ids[i : i + batch_size],
                    embeddings=img_embeddings[i : i + batch_size],
                    documents=img_texts[i : i + batch_size],
                    metadatas=img_metas[i : i + batch_size],
                )
            print(f"  {len(img_texts)} descripciones de imágenes indexadas.")
    else:
        print("\nExtracción de imágenes omitida (--skip-images).")

    print(f"Indexación completa. Vectores guardados en '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexar PDFs del curso en ChromaDB.")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Omitir extracción de imágenes; solo indexar texto.",
    )
    args = parser.parse_args()
    build_index(skip_images=args.skip_images)
