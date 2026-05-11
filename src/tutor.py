import os
from groq import Groq
from dotenv import load_dotenv
from src.prompts import SYSTEM_PROMPT
from src.retriever import retrieve, format_context

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _format_history(messages: list) -> str:
    lines = []
    for msg in messages[:-1]:
        role = "Estudiante" if msg["role"] == "user" else "BAC Tutor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "(inicio de la conversación)"


def _last_substantive_question(messages: list) -> str:
    """Return the last user message longer than 20 chars, for fallback retrieval."""
    for msg in reversed(messages[:-1]):
        if msg["role"] == "user" and len(msg["content"].strip()) > 20:
            return msg["content"]
    return ""


def get_response(question: str, messages: list) -> str:
    try:
        nodes = retrieve(question)
        # If the current message is short/meta ("no sé", "ayúdame", "?"),
        # re-retrieve using the last substantive question so context stays relevant.
        if not nodes and len(question.strip()) < 30:
            fallback = _last_substantive_question(messages)
            if fallback:
                nodes = retrieve(fallback)
    except Exception as e:
        if "getaddrinfo" in str(e) or "ConnectError" in type(e).__name__:
            return "⚠️ Sin conexión. Verifica tu internet e intenta de nuevo."
        raise

    if not nodes:
        return (
            "No encontré fragmentos del material del curso relacionados con tu pregunta. "
            "¿Puedes reformularla usando términos del metamodelo BAC?"
        )

    context = format_context(nodes)
    chat_history = _format_history(messages)

    prompt = SYSTEM_PROMPT.format(
        context=context,
        chat_history=chat_history,
        question=question,
    )

    try:
        response = _get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            return "⚠️ Límite de la API alcanzado. Espera un momento e intenta de nuevo."
        if "getaddrinfo" in err or "ConnectError" in type(e).__name__:
            return "⚠️ Sin conexión a la API de Groq. Verifica tu internet e intenta de nuevo."
        raise
