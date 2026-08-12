"""
Compaction strategy: once a session passes MAX_TURNS_BEFORE_COMPACTION turns,
summarize the older turns into a single rolling summary using the LLM, then
only keep the last few raw turns + the summary in context. This keeps prompt
size bounded for long-running research sessions.
"""
from openai import OpenAI

from app.config import settings
from app.memory.store import (
    MAX_TURNS_BEFORE_COMPACTION,
    count_turns,
    get_recent_turns,
    get_summary,
    upsert_summary,
)

client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def maybe_compact(session_id: str):
    if count_turns(session_id) < MAX_TURNS_BEFORE_COMPACTION:
        return

    existing_summary = get_summary(session_id) or ""
    turns = get_recent_turns(session_id, limit=MAX_TURNS_BEFORE_COMPACTION)
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)

    prompt = (
        "Summarize the following research conversation into a concise rolling "
        "summary (max 150 words) that preserves key facts, entities, and open "
        "questions. Merge with the prior summary if provided.\n\n"
        f"Prior summary: {existing_summary or '(none)'}\n\n"
        f"Recent conversation:\n{transcript}"
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    new_summary = response.choices[0].message.content.strip()
    upsert_summary(session_id, new_summary)


def build_context_messages(session_id: str, recent_turn_limit: int = 6) -> list[dict]:
    """Combine long-term summary + recent raw turns into LLM-ready messages."""
    messages = []
    summary = get_summary(session_id)
    if summary:
        messages.append({
            "role": "system",
            "content": f"Summary of earlier conversation: {summary}",
        })
    messages.extend(get_recent_turns(session_id, limit=recent_turn_limit))
    return messages
