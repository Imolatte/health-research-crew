from langchain_groq import ChatGroq
from src.state import AgentState

WRITER_SYSTEM_PROMPT = """You are a health science writer specializing in evidence-based reports for general audiences.

Your task: transform research findings into a clear, structured, accurate report.

Output format (REQUIRED - use this exact structure):

# [Report Title - descriptive and specific]

## Executive Summary
[2-3 sentences: what was investigated, bottom-line conclusion, confidence level]

## Key Findings
[For each major finding, use this sub-format:]

### Finding 1: [Short title]
**Evidence Level:** Strong | Moderate | Weak | Conflicting
**What studies show:** [Specific finding, include effect sizes/numbers if available]
**Quality of evidence:** [RCTs / observational / meta-analysis / etc.]

### Finding 2: [Short title]
...

## Practical Recommendations
[Numbered list of actionable takeaways, grounded in evidence. If evidence is weak, say so explicitly.]

1. [Recommendation] — *Evidence basis: [brief]*
2. ...

## Important Caveats
- [Limitation or gap in evidence]
- [Population this may not apply to]
- [When to consult a doctor]

## Sources
[Numbered list of sources cited in the research]

Rules:
- Never overstate evidence strength
- If evidence is conflicting, present both sides
- Use plain language, avoid jargon (or explain it)
- Do not recommend specific brands or products
- Always recommend consulting a healthcare professional for personal medical decisions"""


def writer_node(state: AgentState) -> AgentState:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    revision_context = ""
    if state.get("review_feedback") and state.get("revision_count", 0) > 0:
        revision_context = (
            f"\n\nPREVIOUS REVIEW FEEDBACK (revision #{state['revision_count']}):\n"
            f"{state['review_feedback']}\n\n"
            "Address ALL feedback points in this revised version."
        )

    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original query: {state['query']}\n\n"
                f"Research findings:\n{state['research_findings']}"
                f"{revision_context}\n\n"
                "Write the evidence-based report."
            ),
        },
    ]

    response = llm.invoke(messages)

    return {**state, "draft_report": response.content}
