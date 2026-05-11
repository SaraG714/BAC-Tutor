# BAC Tutor — Notas de desarrollo

**Curso:** ISIS-2403 · Arquitectura Empresarial  
**Monitores:** Sara García, Felipe Celis · Uniandes 2026-1  
**Objetivo:** Tutor socrático con RAG que guía sin dar respuestas directas, usando solo el material del curso.

---

## Stack técnico

| Componente | Tecnología | Razón |
|---|---|---|
| UI | Streamlit | Deploy gratuito en Streamlit Cloud |
| Embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) | Local, sin límites de API, buen soporte para español |
| Vector store | ChromaDB (cosine similarity) | Persistente, liviano, commitable al repo |
| Generación | Groq API (`llama-3.3-70b-versatile`) | 14,400 req/día gratis, excelente calidad |
| Chunking | LlamaIndex `SentenceSplitter` (512 tokens, 50 overlap) | |

### Por qué se migró de Google Gemini a Groq + sentence-transformers
La API gratuita de Google (`gemini-2.0-flash`, `gemini-embedding-001`) mostró `limit: 0` desde las primeras pruebas — cuota agotada o no habilitada para la región/cuenta. Groq ofrece 14,400 solicitudes/día gratis y sentence-transformers corre localmente sin ningún límite.

---

## Estructura de archivos

```
BAC-Tutor/
├── docs/                  # PDFs del curso (no subir a git si son privados)
├── chroma_db/             # Base vectorial — SE COMMITEA para el deploy
├── src/
│   ├── indexer.py         # Indexación local con sentence-transformers
│   ├── retriever.py       # Búsqueda semántica en ChromaDB
│   ├── tutor.py           # Lógica socrática + llamada a Groq
│   └── prompts.py         # System prompt
├── app.py                 # Interfaz Streamlit
├── .env                   # GROQ_API_KEY (no commitear)
├── .env.example
├── requirements.txt
├── FLUJO_PROYECTO.md      # Plan de desarrollo por etapas
└── CLAUDE.md              # Este archivo
```

---

## Cómo correr localmente

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell Windows
pip install -r requirements.txt

# Copiar .env.example → .env y poner la GROQ_API_KEY
# Agregar PDFs del curso en docs/

python -m src.indexer             # Solo una vez (o al agregar nuevas lecturas)
streamlit run app.py
```

## Cómo obtener la GROQ_API_KEY
1. Crear cuenta en https://console.groq.com
2. API Keys → Create API Key
3. Pegar en `.env` como `GROQ_API_KEY=gsk_...`

---

## Deploy en Streamlit Cloud

1. Asegurarse de que `chroma_db/` está commiteado al repo
2. Conectar el repo en https://share.streamlit.io
3. Entry point: `app.py`
4. Settings → Secrets → agregar `GROQ_API_KEY = "gsk_..."`

---

## Decisiones de diseño clave

- **MIN_SIMILARITY = 0.4** en `retriever.py`: umbral de similitud coseno; por debajo de esto se considera fuera del alcance del material.
- **chroma_db/ en el repo**: necesario para deploy en plataformas sin almacenamiento persistente.
- **docs/ fuera del repo**: las lecturas del curso son material de Uniandes — no se publican.
- **max_tokens = 512** en Groq: suficiente para 1-2 preguntas socráticas cortas; limita el costo.
