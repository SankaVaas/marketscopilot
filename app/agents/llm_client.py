"""
Single shared LLM client. Because both Ollama (dev) and vLLM (prod) expose an
OpenAI-compatible /v1/chat/completions endpoint, this is the ONLY place that
needs to change when moving between them -- and even then, it's just the
LLM_BASE_URL / LLM_MODEL env vars, not this code.
"""
from openai import OpenAI

from app.config import settings

client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
