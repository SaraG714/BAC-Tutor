import os
import base64
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

# ── Check ChromaDB ────────────────────────────────────────────────────────────
if not os.path.exists("chroma_db"):
    st.error(
        "La base de conocimiento no está indexada. "
        "Ejecuta `python -m src.indexer` con los PDFs en `docs/` antes de continuar.",
        icon="📚",
    )
    st.stop()

# ── Import tutor (after checks so errors are visible) ────────────────────────
from src.tutor import get_response, get_response_with_image  # noqa: E402

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Opciones")
    debug_mode = st.toggle("Mostrar fragmentos recuperados", value=False)

    st.divider()
    st.markdown("### Subir diagrama")
    st.caption("Se analizará junto con tu siguiente mensaje.")
    uploaded_file = st.file_uploader(
        "Diagrama (.svg, .png, .jpg)",
        type=["svg", "png", "jpg", "jpeg"],
        key=f"diagram_{st.session_state.get('uploader_key', 0)}",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        raw = uploaded_file.read()
        if uploaded_file.name.lower().endswith(".svg"):
            import cairosvg  # noqa: PLC0415
            png_bytes = cairosvg.svg2png(bytestring=raw)
        else:
            png_bytes = raw
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        st.session_state.pending_image = {"b64": b64, "name": uploaded_file.name}

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_nodes" not in st.session_state:
    st.session_state.last_nodes = []
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ── Pending image preview ─────────────────────────────────────────────────────
if st.session_state.pending_image:
    st.image(
        base64.b64decode(st.session_state.pending_image["b64"]),
        caption=f"📎 {st.session_state.pending_image['name']} — se enviará con tu próximo mensaje",
        use_container_width=False,
        width=300,
    )

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image_b64"):
            st.image(
                base64.b64decode(msg["image_b64"]),
                caption=msg.get("image_name", ""),
                use_container_width=False,
                width=200,
            )

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Escribe aquí tu duda o describe tu modelo..."):
    # Capture and clear pending image before re-render so it isn't reused
    pending = st.session_state.pending_image
    st.session_state.pending_image = None
    st.session_state.uploader_key += 1  # resets file_uploader widget

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image_b64": pending["b64"] if pending else None,
        "image_name": pending["name"] if pending else None,
    })
    with st.chat_message("user"):
        st.markdown(prompt)
        if pending:
            st.image(
                base64.b64decode(pending["b64"]),
                caption=pending["name"],
                use_container_width=False,
                width=200,
            )

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            if pending:
                reply, nodes = get_response_with_image(
                    prompt, st.session_state.messages, pending["b64"]
                )
            else:
                reply, nodes = get_response(prompt, st.session_state.messages)
        st.markdown(reply)
        if debug_mode and nodes:
            with st.expander(f"🔍 {len(nodes)} fragmentos recuperados"):
                for i, node in enumerate(nodes, 1):
                    fname = node["metadata"].get("file_name", "")
                    page = node["metadata"].get("page_label", "?")
                    sim = node.get("similarity", 0)
                    node_type = node["metadata"].get("type", "text")
                    if node_type == "image":
                        label = f"📷 imagen p.{page}"
                    else:
                        label = f"{fname} · p.{page}"
                    st.caption(f"**#{i}** · {label} · similitud: {sim:.2f}")
                    st.text(node["text"][:400] + ("…" if len(node["text"]) > 400 else ""))
                    if i < len(nodes):
                        st.divider()

    st.session_state.last_nodes = nodes
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Monitores: Sara García · Felipe Celis &nbsp;|&nbsp; ISIS-2403 · 2026-1")
