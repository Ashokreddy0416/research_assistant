# tools.py
# PURPOSE: Give the agent ability to search web and read pages
# USED BY: agent.py calls these tools during the search node

import requests                       # fetches web pages
from bs4 import BeautifulSoup         # parses HTML → clean text
from ddgs import DDGS                 # free web search, no API key (renamed from duckduckgo_search)
from langchain_core.tools import tool # @tool decorator

# Browser-like headers so websites don't block our scraper
HEADERS = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}


@tool
def web_search(query: str, max_results: int = 6) -> str:
    """
    Search the web using DuckDuckGo.
    Returns: titles, URLs, and snippets of top results.
    Use this to find relevant sources for a research topic.
    The docstring is what the LLM reads to know when to use this tool.
    """
    try:
        with DDGS() as ddgs:                   # open DuckDuckGo session
            results = list(
                ddgs.text(query, max_results=max_results)
            )

        if not results:
            return "No results found for this query."

        # Format results as readable text for the LLM
        formatted = []
        for i, r in enumerate(results, 1):    # enumerate starts at 1
            formatted.append(
                f"[{i}] {r['title']}\nURL: {r['href']}\n{r['body']}\n"
            )
        return "\n".join(formatted)

    except Exception as e:
        return f"Search error: {str(e)}"       # return error as string, not crash


@tool
def scrape_page(url: str, max_chars: int = 4000) -> str:
    """
    Fetch and extract clean text from a webpage URL.
    Use after web_search to read the full content of a specific page.
    Returns the first 4000 characters of main content.
    """
    try:
        # fetch the page (timeout=10 means give up after 10 seconds)
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()                # raise error if 404/500 etc

        # parse the HTML
        soup = BeautifulSoup(resp.text, "html.parser")

        # remove noise: scripts, styles, nav bars, footers
        for tag in soup(["script", "style",
                          "nav", "footer", "header"]):
            tag.decompose()                    # remove tag from tree

        # find main content area (try main → article → body)
        main = (soup.find("main") or
                soup.find("article") or
                soup.body)

        if main:
            text = main.get_text(separator=" ", strip=True)
        else:
            text = ""

        # collapse multiple spaces into single spaces
        text = " ".join(text.split())

        # return source URL + truncated text
        return f"Source: {url}\n\n{text[:max_chars]}"

    except requests.Timeout:
        return f"Timeout: could not load {url}"
    except Exception as e:
        return f"Scrape error: {str(e)}"


# List of all tools — agent.py imports this
TOOLS = [web_search, scrape_page]