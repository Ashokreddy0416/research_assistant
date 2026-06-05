# main.py
# PURPOSE: Entry point — run this file to use the research agent
# USAGE:   uv run python main.py

from agent import agent          # our compiled LangGraph agent
from memory import clear_memory  # wipe memory before new topic


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

    # Show a preview (first 500 characters)
    print(f"\n{'='*55}")
    print("✅ DONE! Report preview:")
    print(f"{'='*55}")
    print(report[:500])
    print("...\n(Full report saved in reports/ folder)")

    return report


# This block ONLY runs when you do: uv run python main.py
# It does NOT run when other files do: from main import run_research
if __name__ == "__main__":
    # Ask user for topic — strip() removes accidental spaces
    topic = input("Enter research topic: ").strip()

    # Use a default if user just presses Enter
    if not topic:
        topic = "Impact of AI on software developer jobs in India 2025"

    run_research(topic)