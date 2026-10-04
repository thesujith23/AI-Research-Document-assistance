import os
import gc
from sentence_transformers import CrossEncoder

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_reranker_model = None

def _get_model():
    global _reranker_model
    if _reranker_model is None:
        print("[Reranker] Loading Cross-Encoder model...")
        # Force CPU device to prevent GPU memory spikes in restricted environments
        _reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device='cpu')
    return _reranker_model

def release_reranker():
    """Explicitly releases the reranker model from memory."""
    global _reranker_model
    if _reranker_model is not None:
        print("[Reranker] Releasing Cross-Encoder model...")
        del _reranker_model
        _reranker_model = None
        gc.collect()

def rerank(query: str, retrieved_chunks: list[dict], top_k: int = 3) -> list[dict]:
    if not retrieved_chunks:
        return []
        
    model = _get_model()
    pairs = [[query, chunk.get("text", "")] for chunk in retrieved_chunks]
    
    scores = model.predict(pairs)
    
    for i, chunk in enumerate(retrieved_chunks):
        chunk["reranker_score"] = float(scores[i])
        
    reranked_chunks = sorted(retrieved_chunks, key=lambda x: x["reranker_score"], reverse=True)
    return reranked_chunks[:top_k]

