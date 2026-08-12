"""
Qdrant wrapper -- self-hosted, free, production-capable vector DB.
Swap for AWS OpenSearch Serverless (or another managed vector DB) in
production by changing only this module's client init; the interface
(`upsert_chunks`, `search`) stays identical.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings

_client = QdrantClient(url=settings.qdrant_url)

VECTOR_SIZE = 1024  # BAAI/bge-large-en-v1.5 output dimension


def ensure_collection():
    collections = [c.name for c in _client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        _client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE, distance=qmodels.Distance.COSINE
            ),
        )


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """chunks: list of {chunk_id, text, source_doc, classification}"""
    ensure_collection()
    points = [
        qmodels.PointStruct(
            id=chunk["chunk_id"],
            vector=embedding,
            payload={
                "text": chunk["text"],
                "source_doc": chunk["source_doc"],
                "classification": chunk["classification"],
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    _client.upsert(collection_name=settings.qdrant_collection, points=points)


def search(query_embedding: list[float], top_k: int = 10, allowed_classifications=None):
    query_filter = None
    if allowed_classifications:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="classification",
                    match=qmodels.MatchAny(any=list(allowed_classifications)),
                )
            ]
        )

    results = _client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=query_filter,
    )
    return [
        {
            "chunk_id": str(r.id),
            "text": r.payload["text"],
            "source_doc": r.payload["source_doc"],
            "classification": r.payload["classification"],
            "score": r.score,
        }
        for r in results
    ]
