# BAC Tutor — Flujo de desarrollo del proyecto

**Curso:** ISIS-2403 · Arquitectura Empresarial  
**Monitores:** Sara García, Felipe Celis · 2026-1  
**Objetivo:** Tutor socrático con RAG que guía sin dar respuestas directas, usando solo el material del curso.

---

## Etapa 0 · Prerequisitos y configuración del entorno

### 0.1 Cuentas y claves de API
- [ ] Crear cuenta en [Google AI Studio](https://aistudio.google.com) y obtener API key para:
  - Gemini 2.0 Flash (generación)
  - `text-embedding-004` (embeddings)
- [ ] Crear cuenta en Streamlit Cloud (deploy gratuito)

### 0.2 Entorno local
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit
llama-index
llama-index-embeddings-google
llama-index-llms-google-genai
chromadb
google-generativeai
python-dotenv
pypdf
```

### 0.3 Variables de entorno
Crear archivo `.env` en la raíz:
```
GOOGLE_API_KEY=tu_api_key_aqui
```

---

## Etapa 1 · Preparación del material del curso

### 1.1 Recolección de PDFs
- [ ] Reunir todos los PDFs de lecturas del curso (metamodelo BAC, lecturas de Estructura de Negocio, etc.)
- [ ] Guardar en carpeta `docs/` en la raíz del proyecto
- [ ] Verificar que los PDFs tengan texto seleccionable (no imágenes escaneadas)

### 1.2 Estructura de carpetas
```
BAC-Tutor/
├── docs/                  # PDFs del curso (no subir a git si son privados)
├── chroma_db/             # Base vectorial generada automáticamente
├── src/
│   ├── indexer.py         # Script de indexación (se corre una sola vez)
│   ├── retriever.py       # Lógica de búsqueda semántica
│   ├── tutor.py           # Lógica del modo socrático + llamada a Gemini
│   └── prompts.py         # System prompt y plantillas
├── app.py                 # Interfaz Streamlit
├── .env
├── requirements.txt
└── FLUJO_PROYECTO.md
```

---

## Etapa 2 · Indexación (pipeline RAG — parte 1)

> Se ejecuta una sola vez, o cuando se añaden nuevas lecturas.

### 2.1 Chunking con LlamaIndex (`src/indexer.py`)
- Cargar PDFs desde `docs/` con `SimpleDirectoryReader`
- Dividir en chunks de ~512 tokens con overlap de ~50 tokens
- Conservar metadatos: nombre del archivo + número de página

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

reader = SimpleDirectoryReader("docs/")
documents = reader.load_data()

parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)
```

### 2.2 Embeddings con Google (`text-embedding-004`)
```python
from llama_index.embeddings.google import GoogleGenAIEmbedding

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    api_key=os.getenv("GOOGLE_API_KEY")
)
```

### 2.3 Almacenamiento en ChromaDB
```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("bac_tutor")
vector_store = ChromaVectorStore(chroma_collection=collection)
```

### 2.4 Validación de la indexación
- [ ] Verificar que ChromaDB tiene al menos tantos documentos como páginas hay en los PDFs
- [ ] Hacer una búsqueda manual de prueba: consultar "componente interno" y revisar que los fragmentos recuperados son relevantes

---

## Etapa 3 · Recuperación semántica (pipeline RAG — parte 2)

### 3.1 Retriever (`src/retriever.py`)
- Convertir la pregunta del estudiante a vector con el mismo modelo de embedding
- Buscar los **Top 3 fragmentos** más cercanos en ChromaDB
- Devolver fragmentos + metadatos (archivo, página)

```python
def retrieve(query: str, index, top_k=3):
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return nodes  # cada nodo tiene .text y .metadata
```

### 3.2 Criterios de calidad del retriever
- El fragmento recuperado debe contener información directamente relacionada con la pregunta
- Si el score de similitud es bajo (< 0.6), indicar al usuario que la pregunta está fuera del alcance del material

---

## Etapa 4 · Generación socrática (pipeline RAG — parte 3)

### 4.1 System prompt (`src/prompts.py`)

El system prompt es el núcleo del proyecto. Debe:
- Prohibir explícitamente dar la respuesta directa
- Instruir al modelo a formular 1–2 preguntas que lleven al estudiante a razonar
- Anclar las respuestas al material recuperado
- Citar la página exacta de la lectura cuando sea relevante

```python
SYSTEM_PROMPT = """
Eres BAC Tutor, un asistente socrático para el curso ISIS-2403 Arquitectura Empresarial
de la Universidad de los Andes.

Tu única fuente de conocimiento son los fragmentos del material del curso que se te
proporcionan en cada consulta. No uses conocimiento externo.

REGLAS ESTRICTAS:
1. NUNCA des la respuesta directa al estudiante.
2. Responde SIEMPRE con 1 o 2 preguntas que lo guíen a descubrir la respuesta por sí mismo.
3. Si el estudiante insiste en pedir la respuesta directa, reformula la pregunta socrática
   de forma diferente. Nunca cedas.
4. Cita la página exacta de la lectura cuando uses un concepto del material
   (ej: "según la lectura, p. 4...").
5. Usa el idioma del estudiante (español).
6. Si la pregunta está fuera del material del curso, dilo claramente y redirige.

Contexto del material del curso:
{context}

Historial de conversación:
{chat_history}

Pregunta del estudiante: {question}
"""
```

### 4.2 Llamada a Gemini 2.0 Flash (`src/tutor.py`)
```python
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

def generate_socratic_response(question, context_nodes, chat_history):
    context = "\n\n".join([
        f"[{n.metadata.get('file_name', '')}, p.{n.metadata.get('page_label', '?')}]\n{n.text}"
        for n in context_nodes
    ])
    prompt = SYSTEM_PROMPT.format(
        context=context,
        chat_history=chat_history,
        question=question
    )
    response = llm.complete(prompt)
    return response.text
```

---

## Etapa 5 · Interfaz Streamlit (`app.py`)

### 5.1 Componentes de la UI
- Header con nombre "BAC Tutor" y descripción corta
- Chat con historial de mensajes (burbujas diferenciadas estudiante/tutor)
- Input de texto en la parte inferior
- (Opcional) Botón para subir un diagrama como imagen

### 5.2 Manejo del historial de conversación
```python
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Describe tu modelo o pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # ... llamar al retriever y al tutor
```

### 5.3 Flujo completo en `app.py`
1. Usuario escribe → se agrega al historial
2. Se llama `retrieve(query)` → Top 3 fragmentos
3. Se llama `generate_socratic_response(query, fragments, history)` → pregunta socrática
4. Respuesta se muestra y se agrega al historial

---

## Etapa 6 · Validación del modo socrático

> Antes del deploy, validar que el tutor realmente no da respuestas directas.

### 6.1 Casos de prueba obligatorios

| Entrada del estudiante | Comportamiento esperado del tutor |
|---|---|
| "Puse Portal WEB como componente interno" | Pregunta qué puede contener un componente interno según la lectura |
| "¿Cuáles son los componentes de la Estructura de Negocio?" | NO lista los componentes — pregunta qué entiende el estudiante por "componente" |
| "Dame la respuesta directamente" | Reformula la pregunta socrática, no cede |
| "¿Qué es un canal de valor?" | Pregunta socrática sobre la diferencia entre canal y componente |
| Pregunta sobre tecnología/procesos | Indica que está fuera del alcance del material BAC |

### 6.2 Criterios de aceptación
- [ ] En ningún caso de prueba el tutor entrega la respuesta directa
- [ ] Todas las respuestas citan el material del curso
- [ ] El tutor se mantiene en modo socrático aunque el estudiante insista 3+ veces

---

## Etapa 7 · Deploy en Streamlit Cloud

### 7.1 Preparar el repositorio
- [ ] Crear `.gitignore` que excluya `.env`, `chroma_db/`, `docs/` (si los PDFs son privados)
- [ ] Subir `chroma_db/` pre-generado al repo (es necesario para el deploy) **o** configurar regeneración en el arrange de Streamlit Cloud

### 7.2 Secrets en Streamlit Cloud
En el dashboard de Streamlit Cloud → Settings → Secrets:
```toml
GOOGLE_API_KEY = "tu_api_key"
```

### 7.3 Pasos del deploy
1. Push del repositorio a GitHub
2. Conectar repo en Streamlit Cloud
3. Seleccionar `app.py` como entry point
4. Configurar secrets
5. Deploy → obtener URL pública

---

## Etapa 8 · Prueba con estudiantes

### 8.1 Protocolo de prueba
- Seleccionar 3–5 estudiantes de ISIS-2403
- Darles una tarea concreta: "Modela la Estructura de Negocio de [empresa X] y usa el tutor cuando tengas dudas"
- Observar sin intervenir
- Registrar: ¿llegaron a la respuesta solos? ¿cuántas preguntas necesitaron?

### 8.2 Métricas a recolectar
- Número de turnos promedio hasta que el estudiante corrige su error
- Tasa de abandono (¿dejaron de usar el tutor?)
- Errores de retrieval (¿el tutor citó mal o usó fragmentos irrelevantes?)

### 8.3 Ajustes post-prueba
- Refinar el system prompt si el tutor cede o da respuestas directas
- Ajustar `chunk_size` si los fragmentos recuperados son demasiado largos o cortos
- Agregar lecturas si hay preguntas frecuentes fuera del material actual

---

## Resumen de dependencias por etapa

| Etapa | Depende de |
|---|---|
| Etapa 1 (docs) | — |
| Etapa 2 (indexación) | Etapa 0 + Etapa 1 |
| Etapa 3 (retriever) | Etapa 2 |
| Etapa 4 (generación) | Etapa 3 |
| Etapa 5 (UI) | Etapa 4 |
| Etapa 6 (validación) | Etapa 5 |
| Etapa 7 (deploy) | Etapa 6 |
| Etapa 8 (prueba) | Etapa 7 |

---

## Checklist de entrega MVP

- [ ] PDFs indexados en ChromaDB
- [ ] Retriever devuelve fragmentos relevantes con metadatos de página
- [ ] System prompt socrático validado (no da respuestas directas)
- [ ] Chat funcional en Streamlit con historial
- [ ] Deploy público en Streamlit Cloud
- [ ] Al menos 5 casos de prueba superados
- [ ] Probado con al menos 1 estudiante real
