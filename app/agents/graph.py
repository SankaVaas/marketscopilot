"""
The orchestration backbone: a LangGraph StateGraph wiring planner ->
[retrieval -> compliance] -> composer, with conditional routing so
out-of-scope questions skip retrieval entirely.

LangGraph's checkpointing support (MemorySaver here; swap for a Postgres
checkpointer in production) is what enables resumable, multi-turn agent runs
-- the same pattern the JD calls out for long-running workflows.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.compliance_agent import compliance_node
from app.agents.composer import composer_node
from app.agents.planner import planner_node
from app.agents.retrieval_agent import retrieval_node
from app.agents.state import AgentState


def route_after_planner(state: AgentState) -> str:
    return "retrieval" if state.get("in_scope") else "composer"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("composer", composer_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner", route_after_planner, {"retrieval": "retrieval", "composer": "composer"}
    )
    graph.add_edge("retrieval", "compliance")
    graph.add_edge("compliance", "composer")
    graph.add_edge("composer", END)

    checkpointer = MemorySaver()  # swap for a Postgres checkpointer in prod
    return graph.compile(checkpointer=checkpointer)


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
