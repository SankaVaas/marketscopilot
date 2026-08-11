"""
Two chunking strategies, selectable per document type:
  - fixed_window: simple, predictable, good baseline
  - semantic: splits on paragraph/heading boundaries, better for structured
    research notes and policy docs

Both are pure-Python, no paid API needed.
"""
import re
import uuid


def fixed_window_chunk(text: str, window_size: int = 800, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def semantic_chunk(text: str, max_chunk_chars: int = 1200) -> list[str]:
    """Split on markdown headings / blank lines, then merge small pieces up to max size."""
    raw_sections = re.split(r"\n\s*\n|\n#{1,6}\s", text)
    raw_sections = [s.strip() for s in raw_sections if s.strip()]

    chunks, buffer = [], ""
    for section in raw_sections:
        if len(buffer) + len(section) <= max_chunk_chars:
            buffer = f"{buffer}\n\n{section}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = section
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_document(text: str, strategy: str = "semantic") -> list[dict]:
    """Returns list of {chunk_id, text} dicts ready for embedding + indexing."""
    if strategy == "fixed_window":
        pieces = fixed_window_chunk(text)
    else:
        pieces = semantic_chunk(text)

    return [{"chunk_id": str(uuid.uuid4()), "text": piece} for piece in pieces]
