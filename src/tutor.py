import os
from groq import Groq
from dotenv import load_dotenv
from src.prompts import SYSTEM_PROMPT, FALLBACK_PROMPT, VISION_PROMPT
from src.retriever import retrieve, format_context

load_dotenv()

# Fallback chain: if the primary model hits quota, the next one is tried automatically.
# llama-3.1-8b-instant has ~14x more free requests/day than 70b-versatile.
_MODEL_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

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


def _call_llm(prompt: str) -> str:
    """Try each model in the chain; fall through on 429."""
    last_err = None
    for model in _MODEL_CHAIN:
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            etype = type(e).__name__
            if "429" in err or "rate_limit" in err.lower():
                last_err = e
                continue  # try next model
            if "ConnectionError" in etype or "getaddrinfo" in err:
                raise RuntimeError("connection") from e
            raise  # non-quota error → propagate immediately
    raise last_err  # all models exhausted


_VISION_MODEL_CHAIN = ["meta-llama/llama-4-scout-17b-16e-instruct"]


def _call_llm_with_image(prompt: str, image_b64: str) -> str:
    """Call vision model with an inline image; no text-only fallback."""
    last_err = None
    for model in _VISION_MODEL_CHAIN:
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            etype = type(e).__name__
            if "429" in err or "rate_limit" in err.lower():
                last_err = e
                continue
            if "ConnectionError" in etype or "getaddrinfo" in err:
                raise RuntimeError("connection") from e
            raise
    raise last_err


def get_response_with_image(question: str, messages: list, image_b64: str) -> tuple:
    chat_history = _format_history(messages)

    try:
        nodes = retrieve(question, chat_history=chat_history)
    except Exception as e:
        if "getaddrinfo" in str(e) or "ConnectError" in type(e).__name__:
            return "⚠️ Sin conexión. Verifica tu internet e intenta de nuevo.", []
        raise

    if not nodes:
        prompt = FALLBACK_PROMPT.format(
            chat_history=chat_history,
            question=question,
        )
    else:
        context = format_context(nodes)
        prompt = VISION_PROMPT.format(
            context=context,
            chat_history=chat_history,
            question=question,
        )

    try:
        return _call_llm_with_image(prompt, image_b64), nodes
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            return "⚠️ Cuota agotada en el modelo de visión. Intenta más tarde o consulta console.groq.com.", []
        if "connection" in err.lower() or "ConnectionError" in type(e).__name__ or "getaddrinfo" in err:
            return "⚠️ Sin conexión a la API de Groq. Verifica tu internet e intenta de nuevo.", []
        raise


def get_response(question: str, messages: list) -> tuple:
    chat_history = _format_history(messages)

    try:
        nodes = retrieve(question, chat_history=chat_history)
    except Exception as e:
        if "getaddrinfo" in str(e) or "ConnectError" in type(e).__name__:
            return "⚠️ Sin conexión. Verifica tu internet e intenta de nuevo.", []
        raise

    if not nodes:
        prompt = FALLBACK_PROMPT.format(
            chat_history=chat_history,
            question=question,
        )
    else:
        context = format_context(nodes)
        prompt = SYSTEM_PROMPT.format(
            context=context,
            chat_history=chat_history,
            question=question,
        )

    try:
        return _call_llm(prompt), nodes
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            return "⚠️ Cuota agotada en todos los modelos. Intenta mañana o habilita billing en console.groq.com.", []
        if "connection" in err.lower() or "ConnectionError" in type(e).__name__ or "getaddrinfo" in err:
            return "⚠️ Sin conexión a la API de Groq. Verifica tu internet e intenta de nuevo.", []
        raise
