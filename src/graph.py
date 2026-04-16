from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.researcher import researcher_node
from src.agents.writer import writer_node
from src.agents.reviewer import reviewer_node


def should_revise(state: AgentState) -> str:
    """Conditional edge: route to writer for revision or END."""
    if state.get("needs_revision") and state.get("revision_count", 0) < 1:
        return "revise"
    return "done"


def increment_revision(state: AgentState) -> AgentState:
    """Increment revision counter before re-running writer."""
    return {**state, "revision_count": state.get("revision_count", 0) + 1}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("increment_revision", increment_revision)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        should_revise,
        {
            "revise": "increment_revision",
            "done": END,
        },
    )

    graph.add_edge("increment_revision", "writer")

    return graph.compile()
