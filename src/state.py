from typing import TypedDict


class AgentState(TypedDict):
    query: str
    research_findings: str
    draft_report: str
    review_feedback: str
    final_report: str
    quality_score: int
    needs_revision: bool
    revision_count: int
