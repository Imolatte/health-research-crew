from langchain_community.tools import DuckDuckGoSearchRun


def search_health_topic(query: str) -> str:
    """Broad research on a health topic."""
    tool = DuckDuckGoSearchRun()
    result = tool.run(f"scientific evidence clinical study {query} PubMed research")
    return result or "No results found."


def verify_claim(claim: str) -> str:
    """Targeted fact-check for a specific health claim."""
    tool = DuckDuckGoSearchRun()
    result = tool.run(f"systematic review meta-analysis verify: {claim}")
    return result or "No verification results found."
