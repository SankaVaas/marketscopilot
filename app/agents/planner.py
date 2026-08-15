from app.agents.llm_client import chat
from app.agents.state import AgentState

SCOPE_SYSTEM_PROMPT = (
    "You are a scope-checking assistant for an internal Global Markets "
    "research tool. Given a user question, decide if it is a reasonable "
    "market research / compliance-policy question that this tool should "
    "answer, or if it's out of scope (e.g. asks for trade execution, "
    "personal financial advice to a retail client, or unrelated small talk). "
    "Respond with exactly one line: 'IN_SCOPE' or 'OUT_OF_SCOPE: <short reason>'."
)


def planner_node(state: AgentState) -> AgentState:
    verdict = chat(
        messages=[
            {"role": "system", "content": SCOPE_SYSTEM_PROMPT},
            {"role": "user", "content": state["question"]},
        ],
        temperature=0.0,
        max_tokens=60,
    )

    if verdict.strip().upper().startswith("IN_SCOPE"):
        return {**state, "in_scope": True, "scope_reason": None}

    reason = verdict.split(":", 1)[-1].strip() if ":" in verdict else verdict
    return {**state, "in_scope": False, "scope_reason": reason}
