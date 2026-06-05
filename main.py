# main.py
# PURPOSE: Entry point — run this file to use the research agent
# USAGE:   uv run python main.py

from agent import agent        # our compiled LangGraph agent
from memory import clear_memory # wipe memory before new topic


def run_research(topic: str) -> str:
    """
    Run the full research agent pipeline for a given topic.
    Returns the final report as a string.
    """
    # Print a clear separator so output is easy to read
    print(f"\n{'='*55}")
    print("🔬 RESEARCH AGENT STARTING")
    print(f"📌 Topic: {topic}")
    print(f"{'='*55}\n")

    # Always clear memory before a new topic
    # Otherwise chunks from previous runs pollute results
    clear_memory()

    # thread_id lets LangGraph track this specific run
    # Useful for resuming if interrupted
    config = {"configurable": {"thread_id": topic[:30]}}

    # Initial state — fill in all required keys
    # Annotated[List] fields start as empty lists
    initial_state = {
        "topic":          topic,
        "queries":        [],
        "search_results": [],
        "scraped_pages":  [],
        "findings":       "",
        "citations":      [],
        "report":         "",
        "status":         "planning",
    }

    # invoke() runs the full graph synchronously
    # Returns final state dict when all nodes complete
    final_state = agent.invoke(initial_state, config=config)

    # Extract the report from final state
    report = final_state["report"]