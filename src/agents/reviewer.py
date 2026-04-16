import json
import re
from langchain_openai import ChatOpenAI
from src.state import AgentState

REVIEWER_SYSTEM_PROMPT = """You are a peer reviewer for health science content - rigorous, fair, evidence-focused.

Your task: review the draft report against the research findings for accuracy, balance, and quality.

Evaluate on these dimensions:
1. **Accuracy** - Do claims match the research findings? Any unsupported statements?
2. **Evidence integrity** - Are evidence levels correctly represented? Any overstating/understating?
3. **Bias check** - Is the report balanced? Any cherry-picking of positive results?
4. **Completeness** - Are important caveats/limitations included?
5. **Clarity** - Is the structure clear and recommendations actionable?

Output format (REQUIRED - valid JSON only, no other text):
{
  "feedback": {
    "strengths": ["strength 1", "strength 2"],
    "issues": [
      {
        "severity": "critical|major|minor",
        "description": "specific issue description",
        "suggestion": "how to fix it"
      }
    ],
    "overall_assessment": "brief overall assessment"
  },
  "quality_score": <integer 1-10>,
  "needs_revision": <true|false>,
  "revision_notes": "specific instructions for the writer if revision needed, else empty string"
}

Scoring guide:
- 9-10: Publication ready, minimal issues
- 7-8: Good quality, only minor improvements needed -> needs_revision: false
- 5-6: Moderate issues, revision recommended -> needs_revision: true
- 1-4: Major accuracy or bias problems -> needs_revision: true

Set needs_revision=true only if score < 7."""


def reviewer_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original query: {state['query']}\n\n"
                f"Research findings (ground truth):\n{state['research_findings']}\n\n"
                f"Draft report to review:\n{state['draft_report']}\n\n"
                "Provide your peer review as JSON."
            ),
        },
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Extract JSON from code block if wrapped
    json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if json_match:
        raw = json_match.group(1)

    try:
        review = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: accept the draft as-is
        review = {
            "feedback": {"overall_assessment": raw},
            "quality_score": 7,
            "needs_revision": False,
            "revision_notes": "",
        }

    quality_score = int(review.get("quality_score", 7))
    needs_revision = bool(review.get("needs_revision", False))

    # Enforce revision limit
    if state.get("revision_count", 0) >= 1:
        needs_revision = False

    feedback_text = json.dumps(review.get("feedback", {}), indent=2, ensure_ascii=False)
    revision_notes = review.get("revision_notes", "")
    full_feedback = f"{feedback_text}\n\nRevision notes: {revision_notes}" if revision_notes else feedback_text

    final_report = "" if needs_revision else state["draft_report"]

    return {
        **state,
        "review_feedback": full_feedback,
        "quality_score": quality_score,
        "needs_revision": needs_revision,
        "final_report": final_report,
    }
