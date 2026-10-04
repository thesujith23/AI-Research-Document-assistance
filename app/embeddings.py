import os
import gc
from sentence_transformers import SentenceTransformer

# Set PyTorch/OpenBLAS thread limits for constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def _get_model():
    global _model
    if _model is None:
        print("[Embeddings] Loading SentenceTransformer model...")
        _model = SentenceTransformer(MODEL_NAME, device='cpu')
    return _model

def release_embeddings():
    """Explicitly releases the embedding model from memory."""
    global _model
    if _model is not None:
        print("[Embeddings] Releasing SentenceTransformer model...")
        del _model
        _model = None
        gc.collect()

def embed_documents(chunks: list[dict]) -> list[dict]:
    """
    Takes a list of document chunks and generates embeddings for their text.
    Adds the 'embedding' vector to each chunk dictionary while preserving metadata.
    """
    if not chunks:
        return []
        
    model = _get_model()
    texts = [chunk.get("text", "") for chunk in chunks]
    embeddings = model.encode(texts).tolist()
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
        
    return chunks

def embed_query(query: str) -> list[float]:
    """
    Takes a single query string and generates its embedding.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")
        
    model = _get_model()
    embedding = model.encode(query).tolist()
    
    return embedding

