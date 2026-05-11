import os
from google import genai
from dotenv import load_dotenv
from src.prompts import SYSTEM_PROMPT
from src.retriever import retrieve, format_context

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def _format_history(messages: list) -> str:
    lines = []
    for msg in messages[:-1]:
        role = "Estudiante" if msg["role"] == "user" else "BAC Tutor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "(inicio de la conversación)"


def get_response(question: str, messages: list) -> str:
    nodes = retrieve(question)

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

    response = _get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text.strip()
