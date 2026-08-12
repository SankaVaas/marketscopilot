"""
Hybrid retrieval: combine BM25 (lexical) and Qdrant (dense vector) results
using Reciprocal Rank Fusion (RRF) -- a simple, well-established fusion method
that needs no extra training or paid re-ranking API.
"""
from app.retrieval import bm25_index, vector_store
from app.retrieval.embeddings import embed_query

RRF_K = 60  # standard RRF smoothing constant


def hybrid_search(query: str, top_k: int = 8, allowed_classifications=None) -> list[dict]:
    lexical_results = bm25_index.search(query, top_k=20, allowed_classifications=allowed_classifications)
    query_embedding = embed_query(query)
    vector_results = vector_store.search(
        query_embedding, top_k=20, allowed_classifications=allowed_classifications
    )

    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, result in enumerate(lexical_results):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunk_lookup[cid] = result

    for rank, result in enumerate(vector_results):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunk_lookup[cid] = result

    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [{**chunk_lookup[cid], "fused_score": score} for cid, score in fused]
