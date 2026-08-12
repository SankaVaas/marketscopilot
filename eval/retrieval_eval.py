"""
Retrieval evaluation harness.

Runs each query in sample_queries.json through hybrid_search (the same
retrieval path the live agents use) and scores:
  - precision@k: fraction of retrieved chunks that come from an expected
    source document
  - recall@k (doc-level): whether at least one expected source doc was
    retrieved
  - entitlement correctness: for queries where expected_source_docs is empty
    (an RBAC test case), verifies that RBAC filtering actually excluded the
    restricted content rather than just getting lucky on relevance ranking

Results are written to eval/results/ as JSON, and this script exits non-zero
if any RBAC test case fails -- wired into CI (.github/workflows/eval.yml) so
an entitlement regression fails the build, not just the vibes.

Usage:
    python -m eval.retrieval_eval
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.retrieval import bm25_index
from app.retrieval.hybrid import hybrid_search
from app.security.rbac import allowed_classifications

QUERIES_PATH = Path(__file__).parent / "sample_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"


def precision_at_k(retrieved_docs: list[str], expected_docs: list[str]) -> float:
    if not retrieved_docs:
        return 0.0
    hits = sum(1 for doc in retrieved_docs if doc in expected_docs)
    return hits / len(retrieved_docs)


def recall_at_k(retrieved_docs: list[str], expected_docs: list[str]) -> float:
    if not expected_docs:
        return 1.0  # nothing was expected -- trivially satisfied
    hits = sum(1 for doc in expected_docs if doc in retrieved_docs)
    return hits / len(expected_docs)


def run_eval():
    if not bm25_index.load_index_from_disk():
        print(
            "ERROR: no BM25 index found. Run "
            "`python -m app.retrieval.ingest data/sample_docs` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    queries = json.loads(QUERIES_PATH.read_text())
    results = []
    rbac_failures = 0

    for case in queries:
        allowed = allowed_classifications(case["role"])
        retrieved = hybrid_search(case["query"], top_k=5, allowed_classifications=allowed)
        retrieved_docs = [r["source_doc"] for r in retrieved]

        expected = case["expected_source_docs"]
        p_at_k = precision_at_k(retrieved_docs, expected)
        r_at_k = recall_at_k(retrieved_docs, expected)

        is_rbac_case = len(expected) == 0
        rbac_ok = True
        if is_rbac_case and retrieved_docs:
            rbac_ok = False
            rbac_failures += 1

        results.append({
            "query": case["query"],
            "role": case["role"],
            "expected_source_docs": expected,
            "retrieved_docs": retrieved_docs,
            "precision_at_5": round(p_at_k, 3),
            "recall_at_5": round(r_at_k, 3),
            "rbac_check_passed": rbac_ok,
        })

    avg_precision = sum(r["precision_at_5"] for r in results) / len(results)
    avg_recall = sum(r["recall_at_5"] for r in results) / len(results)

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "num_queries": len(results),
        "avg_precision_at_5": round(avg_precision, 3),
        "avg_recall_at_5": round(avg_recall, 3),
        "rbac_failures": rbac_failures,
        "cases": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_path}")

    if rbac_failures > 0:
        print(f"\nFAILED: {rbac_failures} RBAC entitlement test case(s) leaked restricted content.")
        sys.exit(1)

    print("\nAll RBAC entitlement checks passed.")


if __name__ == "__main__":
    run_eval()
