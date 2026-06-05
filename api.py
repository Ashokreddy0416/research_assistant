# api.py
# PURPOSE: Wrap the agent in an HTTP API server
# RUN:     uv run uvicorn api:app --reload --port 8000
# DOCS:    http://localhost:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agent
from memory import clear_memory
from report import list_reports

# Create the FastAPI app
# title + description appear in auto-generated /docs page
app = FastAPI(
    title="Research Agent API",
    description="Free AI research agent — Groq + Llama 3.1 70B",
    version="1.0.0"
)


# Pydantic model defines the JSON shape of POST request body
# If request doesn't match this schema, FastAPI returns 422 automatically
class ResearchRequest(BaseModel):
    topic: str   # required field


# GET / — root endpoint, just confirms server is alive
@app.get("/")
def root():
    return {
        "message": "Research Agent is running!",
        "docs":    "/docs",
        "model":   "llama-3.1-70b-versatile (free)"
    }


# GET /health — quick health check (used by deployment platforms)
@app.get("/health")
def health():
    return {"status": "ok"}


# POST /research — main endpoint, runs the full agent
# Request body: {"topic": "your research topic"}
# Response:     {"topic": "...", "report": "...", "status": "done"}
@app.post("/research")
def research(req: ResearchRequest):
    # Validate input
    if not req.topic.strip():
        # HTTPException returns proper HTTP error response (400 Bad Request)
        raise HTTPException(status_code=400,
                            detail="Topic cannot be empty")

    # Clear previous memory and run agent
    clear_memory()
    config = {"configurable": {"thread_id": req.topic[:30]}}

    result = agent.invoke(
        {
            "topic":          req.topic,
            "queries":        [],
            "search_results": [],
            "scraped_pages":  [],
            "findings":       "",
            "citations":      [],
            "report":         "",
            "status":         "planning",
        },
        config=config
    )

    return {
        "topic":  req.topic,
        "report": result["report"],
        "status": "done"
    }


# GET /reports — list all saved report files
@app.get("/reports")
def get_reports():
    return {"reports": list_reports()}