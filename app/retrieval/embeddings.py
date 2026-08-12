"""
Open-source embeddings via sentence-transformers -- runs on CPU, no API key,
no per-call cost. BAAI/bge-large-en-v1.5 is MTEB-competitive with commercial
embedding APIs.

The model is loaded once (module-level singleton) since loading is the
expensive part; encoding individual batches is fast.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    # BGE models recommend a query instruction prefix for asymmetric search;
    # applied at query time in hybrid.py, not here (this is for documents).
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    instructed_query = f"Represent this query for retrieving relevant documents: {query}"
    embedding = model.encode([instructed_query], normalize_embeddings=True)
    return embedding[0].tolist()
