from app.agents.llm_client import chat
from app.agents.state import AgentState

COMPOSER_SYSTEM_PROMPT = (
    "You are a Global Markets research assistant. Answer the user's question "
    "ONLY using the numbered source snippets provided. Cite sources inline "
    "using [1], [2], etc. matching the snippet numbers. If the snippets don't "
    "contain enough information to answer confidently, say so explicitly "
    "rather than guessing. Do not give direct trade advice or price-target "
    "recommendations -- describe what the research says, not what the user "
    "should do."
)

# Confidence heuristic: if fewer than this many chunks or all fused scores are
# low, flag for human review rather than silently answering.
MIN_CHUNKS_FOR_CONFIDENT_ANSWER = 2
MIN_FUSED_SCORE_FOR_CONFIDENCE = 0.02


def composer_node(state: AgentState) -> AgentState:
    if not state.get("in_scope", True):
        return {
            **state,
            "answer": (
                "This question is outside what I'm able to help with here: "
                f"{state.get('scope_reason', 'out of scope')}."
            ),
            "citations": [],
            "needs_human_review": False,
            "review_reason": None,
        }

    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            **state,
            "answer": (
                "I couldn't find any documents you're entitled to view that "
                "answer this question. Escalating to a human for follow-up."
            ),
            "citations": [],
            "needs_human_review": True,
            "review_reason": "no_retrievable_content",
        }

    numbered_context = "\n\n".join(
        f"[{i+1}] (source: {c['source_doc']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    answer = chat(
        messages=[
            {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {state['question']}\n\nSources:\n{numbered_context}",
            },
        ],
        temperature=0.3,
        max_tokens=700,
    )

    citations = [
        {
            "chunk_id": c["chunk_id"],
            "source_doc": c["source_doc"],
            "text_snippet": c["text"][:200],
            "classification": c["classification"],
        }
        for c in chunks
    ]

    low_confidence = (
        len(chunks) < MIN_CHUNKS_FOR_CONFIDENT_ANSWER
        or max((c.get("fused_score", 0) for c in chunks), default=0) < MIN_FUSED_SCORE_FOR_CONFIDENCE
    )
    compliance_flags = state.get("compliance_flags", [])

    needs_review = low_confidence or bool(compliance_flags)
    reason = None
    if compliance_flags:
        reason = f"compliance_flags: {', '.join(compliance_flags)}"
    elif low_confidence:
        reason = "low_retrieval_confidence"

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "needs_human_review": needs_review,
        "review_reason": reason,
    }
