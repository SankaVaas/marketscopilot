"""
BM25 lexical search via rank_bm25 -- the same algorithm managed OpenSearch
uses under the hood, just running in-process. For a portfolio-scale corpus
this is plenty fast; swap for a real OpenSearch cluster in production without
changing the hybrid-search interface in hybrid.py.

The index is built in-memory but the underlying chunk registry is persisted
to disk as JSON so the FastAPI process (a separate process from the ingest
script) can load it on startup without re-ingesting. This mirrors a real
deployment where indexing is a separate batch job from the serving process.
"""
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

_bm25_index: BM25Okapi | None = None
_chunk_registry: list[dict] = []  # parallel list: {chunk_id, text, source_doc, classification}

_PERSIST_PATH = Path("data/bm25_chunk_registry.json")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_index(chunks: list[dict], persist: bool = True):
    """chunks: list of {chunk_id, text, source_doc, classification}"""
    global _bm25_index, _chunk_registry
    _chunk_registry = chunks
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)

    if persist:
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PERSIST_PATH.write_text(json.dumps(chunks))


def load_index_from_disk() -> bool:
    """Called at API startup. Returns True if an index was found and loaded."""
    if not _PERSIST_PATH.exists():
        return False
    chunks = json.loads(_PERSIST_PATH.read_text())
    if not chunks:
        return False
    build_index(chunks, persist=False)
    return True


def search(query: str, top_k: int = 10, allowed_classifications=None) -> list[dict]:
    if _bm25_index is None or not _chunk_registry:
        return []

    scores = _bm25_index.get_scores(_tokenize(query))
    ranked = sorted(
        zip(_chunk_registry, scores), key=lambda pair: pair[1], reverse=True
    )

    results = []
    for chunk, score in ranked:
        if allowed_classifications and chunk["classification"] not in allowed_classifications:
            continue
        if score <= 0:
            continue
        results.append({**chunk, "score": float(score)})
        if len(results) >= top_k:
            break
    return results
