"""
Single shared LLM client. Because both Ollama (dev), vLLM (prod), and Groq
(hosted) all expose an OpenAI-compatible /v1/chat/completions endpoint, this
is the ONLY place that needs to change when switching providers -- and even
then, it's just the LLM_BASE_URL / LLM_MODEL / LLM_API_KEY env vars.

Retry/backoff is included because hosted providers (Groq's free tier
especially) enforce per-minute rate limits that a purely local Ollama setup
never hits -- without this, a burst of agent calls will intermittently fail
with 429s.
"""
import time

from openai import APIStatusError, OpenAI

from app.config import settings

client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except APIStatusError as e:
            last_error = e
            if e.status_code == 429 and attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                print(f"Rate limited (429). Retrying in {wait}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raise last_error
