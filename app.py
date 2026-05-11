import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud stores secrets separately; sync to env so the rest of the
# code can use os.getenv() uniformly.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BAC Tutor",
    page_icon="🎓",
    layout="centered",
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage [data-testid="stChatMessageContent"] {
        font-size: 0.97rem;
    }
    .block-container { max-width: 760px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## BAC Tutor")
st.caption(
    "Asistente socrático para ISIS-2403 · Arquitectura Empresarial · "
    "Uniandes 2026-1"
)
st.info(
    "Describe tu modelo, comparte una duda o explica cómo clasificaste un elemento. "
    "El tutor **no da respuestas directas** — te hace las preguntas correctas.",
    icon="💡",
)
st.divider()

# ── Check API key ─────────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "Falta la variable de entorno `GROQ_API_KEY`. "
        "Agrégala en `.env` (local) o en los Secrets de Streamlit Cloud.",
        icon="🔑",
    )
    st.stop()

# ── Check vector index ───────────────────────────────────────────────────────
if not os.path.exists("vector_index"):
    st.error(
        "La base de conocimiento no está indexada. "
        "Ejecuta `python -m src.indexer` con los PDFs en `docs/` antes de continuar.",
        icon="📚",
    )
    st.stop()

# ── Import tutor (after checks so errors are visible) ────────────────────────
from src.tutor import get_response  # noqa: E402

# ── Sidebar — debug mode ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Opciones")
    debug_mode = st.toggle("Mostrar fragmentos recuperados", value=False)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_nodes" not in st.session_state:
    st.session_state.last_nodes = []

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Escribe aquí tu duda o describe tu modelo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            reply, nodes = get_response(prompt, st.session_state.messages)
        st.markdown(reply)
        if debug_mode and nodes:
            with st.expander(f"🔍 {len(nodes)} fragmentos recuperados"):
                for i, node in enumerate(nodes, 1):
                    fname = node["metadata"].get("file_name", "")
                    page = node["metadata"].get("page_label", "?")
                    sim = node.get("similarity", 0)
                    st.caption(f"**#{i}** · {fname} · p.{page} · similitud: {sim:.2f}")
                    st.text(node["text"][:400] + ("…" if len(node["text"]) > 400 else ""))
                    if i < len(nodes):
                        st.divider()

    st.session_state.last_nodes = nodes
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Monitores: Sara García · Felipe Celis &nbsp;|&nbsp; ISIS-2403 · 2026-1")
