import sys
from dotenv import load_dotenv
from src.graph import build_graph
from src.state import AgentState

load_dotenv()


def run(query: str) -> AgentState:
    app = build_graph()

    initial_state: AgentState = {
        "query": query,
        "research_findings": "",
        "draft_report": "",
        "review_feedback": "",
        "final_report": "",
        "quality_score": 0,
        "needs_revision": False,
        "revision_count": 0,
    }

    print(f"\n[Health Research Crew] Query: {query}\n")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n[Researcher] Findings gathered.")
    print(f"[Writer] Draft written (revision #{final_state['revision_count']}).")
    print(f"[Reviewer] Quality score: {final_state['quality_score']}/10")
    print("=" * 60)
    print("\n=== FINAL REPORT ===\n")
    print(final_state["final_report"] or final_state["draft_report"])
    print("\n=== REVIEW FEEDBACK ===\n")
    print(final_state["review_feedback"])

    return final_state


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Is magnesium effective for improving sleep quality?"
    run(query)
