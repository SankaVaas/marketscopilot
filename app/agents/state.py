from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    username: str
    role: str
    allowed_classifications: set[str]

    # Planner output
    in_scope: bool
    scope_reason: Optional[str]

    # Retrieval output
    retrieved_chunks: list[dict]

    # Compliance output
    compliance_flags: list[str]
    blocked_chunk_ids: list[str]

    # Composer output
    answer: str
    citations: list[dict]
    needs_human_review: bool
    review_reason: Optional[str]
