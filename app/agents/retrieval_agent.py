from app.agents.state import AgentState
from app.retrieval.hybrid import hybrid_search


def retrieval_node(state: AgentState) -> AgentState:
    chunks = hybrid_search(
        query=state["question"],
        top_k=8,
        allowed_classifications=state.get("allowed_classifications"),
    )
    return {**state, "retrieved_chunks": chunks}
