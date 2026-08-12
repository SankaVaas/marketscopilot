"""
Compliance-check agent. Runs two layers of checks:
  1. Deterministic entitlement re-check (defense in depth -- the retriever
     already filters by classification, this re-verifies nothing slipped
     through).
  2. LLM-based policy screen: flags retrieved content that looks like it
     could produce investment-advice-like language, which Global Markets
     research assistants typically must avoid for compliance reasons.
"""
from app.agents.llm_client import chat
from app.agents.state import AgentState
from app.security.rbac import is_authorized

POLICY_SYSTEM_PROMPT = (
    "You review retrieved research snippets before they are used to answer "
    "a user's question. Flag any snippet that contains direct trade "
    "recommendations ('buy X now'), specific price targets presented as "
    "advice, or client-specific confidential information. "
    "Respond with a comma-separated list of flags from this set: "
    "trade_recommendation, price_target_advice, confidential_client_info, none."
)


def compliance_node(state: AgentState) -> AgentState:
    chunks = state.get("retrieved_chunks", [])
    role = state["role"]

    blocked_ids = [
        c["chunk_id"] for c in chunks if not is_authorized(role, c["classification"])
    ]
    allowed_chunks = [c for c in chunks if c["chunk_id"] not in blocked_ids]

    if not allowed_chunks:
        return {**state, "retrieved_chunks": [], "compliance_flags": [], "blocked_chunk_ids": blocked_ids}

    combined_text = "\n---\n".join(c["text"] for c in allowed_chunks)
    verdict = chat(
        messages=[
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": combined_text[:4000]},
        ],
        temperature=0.0,
        max_tokens=60,
    )
    flags = [f.strip() for f in verdict.split(",") if f.strip() and f.strip() != "none"]

    return {
        **state,
        "retrieved_chunks": allowed_chunks,
        "compliance_flags": flags,
        "blocked_chunk_ids": blocked_ids,
    }
