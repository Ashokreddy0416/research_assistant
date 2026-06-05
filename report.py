# report.py
# PURPOSE: Format and save research reports as markdown files
# USED BY: agent.py calls format_report() as the final step

import os
from datetime import datetime
from typing import List, Dict

# Create reports/ folder if it doesn't exist yet
# exist_ok=True means don't crash if folder already exists
os.makedirs("reports", exist_ok=True)


def format_report(topic: str,
                   content: str,
                   citations: List[Dict] = []) -> str:
    """
    Build a complete markdown report and save it to reports/ folder.

    topic    : the research topic (used in title + filename)
    content  : the LLM-generated report body
    citations: list of {"claim": "...", "source": "url"} dicts
    returns  : the full report string
    """
    # Format current date/time as readable string
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build the header section (markdown format)
    header = f"""# Research Report: {topic}

**Generated:** {timestamp}
**Model:** Llama 3.1 70B via Groq (free tier)

---

"""
    # Build citations section if we have any
    citation_block = ""
    if citations:
        citation_block = "\n\n---\n\n## Sources\n\n"
        for i, c in enumerate(citations, 1):
            # truncate claim to 80 chars so it's readable
            claim  = c.get("claim",  "")[:80]
            source = c.get("source", "unknown")
            citation_block += f"[{i}] {claim}... — {source}\n"

    # Combine all parts
    full_report = header + content + citation_block

    # Build a safe filename from the topic
    # e.g. "AI in Healthcare" → "ai_in_healthcare_20250605_143022.md"
    safe_name = "_".join(topic.lower().split())[:40]
    ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = f"reports/{safe_name}_{ts_str}.md"

    # Save to disk
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"✓ Report saved: {filepath}")
    return full_report


def list_reports() -> List[str]:
    """Return list of all saved report filenames, newest first."""
    return sorted(os.listdir("reports"), reverse=True)