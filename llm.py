# llm.py
# PURPOSE: Create our LLM connections to Groq
# USED BY: agent.py imports llm_large and llm_fast from here

import os                            # built-in Python: read environment variables
from dotenv import load_dotenv       # reads our .env file
from langchain_groq import ChatGroq  # Groq LLM wrapper

load_dotenv()  # loads GROQ_API_KEY from .env into environment


def get_llm(model: str = "llama-3.3-70b-versatile",
            temperature: float = 0.0):
    """
    Create and return a Groq LLM instance.

    model: which AI model to use
    temperature: 0.0 = focused/deterministic, 1.0 = creative/random
    """
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),  # reads key from .env
        model=model,
        temperature=temperature,
        max_retries=3,   # retry 3 times if API call fails
    )


# llm_large: for planning, synthesis, writing (needs more intelligence)
# Llama 3.3 70B is the current best model on Groq free tier
llm_large = get_llm("llama-3.3-70b-versatile")

# llm_fast: for quick/simple tasks like query generation (saves tokens)
# Llama 3.1 8B is fast and cheap
llm_fast = get_llm("llama-3.1-8b-instant")