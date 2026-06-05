# memory.py
# PURPOSE: Store and retrieve research content using vector similarity
# USED BY: agent.py calls remember() to store, recall() to retrieve

import chromadb    # local vector database
import hashlib     # creates unique IDs for text chunks
from sentence_transformers import SentenceTransformer  # text → vectors
from typing import List, Dict

# Load embedding model (runs locally, no API needed)
# all-MiniLM-L6-v2 is small (~80MB) but very good for semantic search
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# PersistentClient = saves to disk (survives restarts)
# path = where to save the database files
client = chromadb.PersistentClient(path="./chroma_db")

# A collection = like a table in a database
# cosine = similarity metric (best for text)
collection = client.get_or_create_collection(
    name="research_memory",
    metadata={"hnsw:space": "cosine"}
)

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split long text into overlapping chunks.
    WHY: Vector models work best on short pieces (500 words max).
    Overlapping (step=400) means no information is cut off at chunk edges.
    """
    words = text.split()                     # split into individual words
    chunks = []
    for i in range(0, len(words), 400):     # step 400, overlap 100 words
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:                            # don't add empty chunks
            chunks.append(chunk)
    return chunks

def remember(text: str, metadata: Dict = {}):
    """
    Store text in memory.
    1. Split into chunks
    2. Convert each chunk to a vector (embedding)
    3. Store vector + original text in ChromaDB
    """
    chunks = chunk_text(text)
    if not chunks:
        return                               # nothing to store

    # encode() converts list of strings into list of vectors
    embeddings = embedder.encode(chunks).tolist()

    # create unique ID for each chunk using MD5 hash
    ids = [
        hashlib.md5(f"{c}{i}".encode()).hexdigest()
        for i, c in enumerate(chunks)
    ]

    # only pass metadatas if metadata dict is non-empty
    # ChromaDB rejects empty {} metadata
    add_kwargs = {
        "embeddings": embeddings,
        "documents": chunks,
        "ids": ids,
    }
    if metadata:
        add_kwargs["metadatas"] = [metadata] * len(chunks)

    try:
        collection.add(**add_kwargs)
    except Exception as e:
        print(f"remember() failed: {e}")

def recall(query: str, n: int = 6) -> List[str]:
    """
    Retrieve most relevant stored chunks for a query.
    ChromaDB finds chunks whose meaning is closest to the query.
    """
    if collection.count() == 0:
        return []                            # nothing stored yet

    emb = embedder.encode([query]).tolist()  # convert query to vector
    results = collection.query(
        query_embeddings=emb,
        n_results=min(n, collection.count()) # don't ask for more than exists
    )
    return results["documents"][0]          # return list of text chunks

def clear_memory():
    """Wipe all stored memory. Call this before each new research topic."""
    global collection
    client.delete_collection("research_memory")
    collection = client.get_or_create_collection(
        name="research_memory",
        metadata={"hnsw:space": "cosine"}
    )