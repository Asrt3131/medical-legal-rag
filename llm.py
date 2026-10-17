import os
from openai import OpenAI

_client = None
_client_signature = None


def _get_client() -> OpenAI:
    global _client, _client_signature

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.moonshot.cn/v1")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Go to Streamlit Cloud → Settings → Secrets and add it."
        )

    signature = (api_key, base_url)
    if _client is None or _client_signature != signature:
        _client = OpenAI(api_key=api_key, base_url=base_url)
        _client_signature = signature

    return _client


def is_llm_available() -> bool:
    """Check whether an API key is configured, without building the client."""
    return bool(os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """You are an expert medical/legal documents assistant.
Answer ONLY based on the reference texts provided below.
If the answer is not in the texts, say explicitly: "Not found in the provided documents."
Always cite the source page number.
Write the answer in English, clearly and concisely."""


def generate_answer(query: str, sources: list, system_prompt: str = None) -> str:
    if not sources:
        return "No relevant sources found."

    client = _get_client()

    context = "\n\n---\n\n".join(
        f"[Page {s.get('page', '?')} — {s.get('file', 'document')}]\n{s['text']}"
        for s in sources
    )

    user_prompt = f"""Reference texts:
{context}

User question: {query}

Answer:"""

    model = os.getenv("LLM_MODEL", "kimi-k3")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
