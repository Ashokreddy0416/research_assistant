# memory.py — lazy loaded for Render free tier
import chromadb
import hashlib
from typing import List, Dict

_embedder = None
_client = None
_collection = None

def _init():
    """Lazy-load heavy deps only when first used."""
    global _embedder, _client, _collection
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _client = chromadb.PersistentClient(path="./chroma_db")
        _collection = _client.get_or_create_collection(
            name="research_memory",
            metadata={"hnsw:space": "cosine"}
        )
    return _embedder, _collection

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), 400):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def remember(text: str, metadata: Dict = {}):
    embedder, collection = _init()
    chunks = chunk_text(text)
    if not chunks:
        return
    embeddings = embedder.encode(chunks).tolist()
    ids = [hashlib.md5(f"{c}{i}".encode()).hexdigest()
           for i, c in enumerate(chunks)]
    try:
        collection.add(embeddings=embeddings, documents=chunks,
                       ids=ids, metadatas=[metadata] * len(chunks))
    except Exception:
        pass

def recall(query: str, n: int = 6) -> List[str]:
    embedder, collection = _init()
    if collection.count() == 0:
        return []
    emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=emb,
                               n_results=min(n, collection.count()))
    return results["documents"][0]

def clear_memory():
    global _collection
    _init()
    _client.delete_collection("research_memory")
    _collection = _client.get_or_create_collection(
        name="research_memory",
        metadata={"hnsw:space": "cosine"}
    )