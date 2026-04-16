from langchain_openai import ChatOpenAI
from src.state import AgentState
from src.tools.search import search_health_topic, verify_claim

RESEARCHER_SYSTEM_PROMPT = """You are a scientific research specialist focused on health and nutrition evidence.

Your task: investigate the given health query using available search tools, then synthesize findings.

Process:
1. Use search_health_topic for broad research (clinical trials, meta-analyses, systematic reviews)
2. Use verify_claim for 2-3 key claims you find to ensure accuracy
3. Synthesize all findings into a structured research report

Output format (REQUIRED):
## Research Findings

### Key Claims
- Claim 1: [statement] | Evidence Level: [Strong/Moderate/Weak/Conflicting] | Source count: N
- Claim 2: [statement] | Evidence Level: [Strong/Moderate/Weak/Conflicting] | Source count: N
(list all major claims found)

### Supporting Evidence
[Detailed summary of what studies/reviews show, including sample sizes and effect sizes where available]

### Conflicting Evidence
[Any contradicting findings, limitations, methodological issues]

### Confidence Assessment
Overall confidence in findings: [High/Medium/Low]
Reasoning: [why]

### Sources
[List all URLs referenced]

Be precise, cite specific studies where possible. Do not make claims beyond what the evidence supports."""


def researcher_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Research query: {state['query']}\n\n"
                "Use the search tools to gather evidence, then provide your structured findings."
            ),
        },
    ]

    # Gather broad research
    broad_results = search_health_topic(state["query"])

    # Extract key claims for verification via LLM
    claim_extraction = llm.invoke(
        [
            {"role": "system", "content": "Extract 2-3 specific, verifiable claims from this research. Return them as a numbered list, one per line."},
            {"role": "user", "content": broad_results[:3000]},
        ]
    )
    claims_text = claim_extraction.content

    # Verify top claims
    verification_results = []
    for line in claims_text.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            claim = line.lstrip("0123456789.-) ").strip()
            if len(claim) > 10:
                result = verify_claim(claim)
                verification_results.append(f"Verification of '{claim}':\n{result}")

    combined_context = (
        f"BROAD RESEARCH:\n{broad_results}\n\n"
        f"CLAIM VERIFICATIONS:\n" + "\n\n".join(verification_results)
    )

    messages.append({"role": "user", "content": f"Here is the gathered evidence:\n\n{combined_context}\n\nNow produce your structured research findings."})

    response = llm.invoke(messages)

    return {**state, "research_findings": response.content}
