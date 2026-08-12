"""
Ingestion pipeline: reads documents from a folder, chunks them, embeds them,
and loads them into both Qdrant (vector) and the in-process BM25 index.

Usage:
    python -m app.retrieval.ingest data/sample_docs

Each document's filename encodes its classification tag for the demo, e.g.:
    market_note_1.md          -> classification: public
    fo_note_rates_view.md     -> classification: front_office_only
    compliance_policy.md      -> classification: compliance_restricted
Adjust `infer_classification` for a real document-tagging pipeline
(e.g. driven by a metadata sidecar file or a document-management system).
"""
import sys
from pathlib import Path

from app.retrieval import bm25_index, vector_store
from app.retrieval.chunking import chunk_document
from app.retrieval.embeddings import embed_texts


def infer_classification(filename: str) -> str:
    name = filename.lower()
    if "compliance" in name or "restricted" in name:
        return "compliance_restricted"
    if "fo_" in name or "front_office" in name:
        return "front_office_only"
    return "public"


def ingest_folder(folder: str):
    folder_path = Path(folder)
    all_chunks = []

    for file_path in sorted(folder_path.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        classification = infer_classification(file_path.name)
        chunks = chunk_document(text, strategy="semantic")
        for chunk in chunks:
            chunk["source_doc"] = file_path.name
            chunk["classification"] = classification
        all_chunks.extend(chunks)
        print(f"Chunked {file_path.name} -> {len(chunks)} chunks ({classification})")

    if not all_chunks:
        print("No .md documents found -- nothing to ingest.")
        return

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embed_texts([c["text"] for c in all_chunks])

    print("Indexing into Qdrant...")
    vector_store.upsert_chunks(all_chunks, embeddings)

    print("Building BM25 index...")
    bm25_index.build_index(all_chunks)

    print(f"Done. Ingested {len(all_chunks)} chunks from {folder}.")


if __name__ == "__main__":
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs"
    ingest_folder(target_folder)
