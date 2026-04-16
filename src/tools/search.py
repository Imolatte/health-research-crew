from langchain_community.tools.tavily_search import TavilySearchResults


def search_health_topic(query: str) -> str:
    """Broad research on a health topic using multiple search queries."""
    tool = TavilySearchResults(max_results=5, include_answer=True)
    results = tool.invoke({"query": f"scientific evidence {query} clinical studies PubMed"})
    if not results:
        return "No results found."
    formatted = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        formatted.append(f"[{title}]({url})\n{content}")
    return "\n\n---\n\n".join(formatted)


def verify_claim(claim: str) -> str:
    """Targeted fact-check for a specific health claim."""
    tool = TavilySearchResults(max_results=3, include_answer=True)
    results = tool.invoke({"query": f"verify: {claim} systematic review meta-analysis"})
    if not results:
        return "No verification results found."
    formatted = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        formatted.append(f"[{title}]({url})\n{content}")
    return "\n\n---\n\n".join(formatted)
