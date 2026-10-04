import os
import sys
import json
import gc

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.document_loader import load_pdf
from app.chunker import chunk_documents
from app.embeddings import embed_documents, embed_query
from app.vector_store import VectorStore
from app.bm25_retriever import BM25Retriever
from app.hybrid_retriever import HybridRetriever
from app.reranker import rerank, release_reranker

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_real_stores(docs_dir, max_pages=None):
    if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
        print(f"Warning: No documents found in {docs_dir}")
        return None, None
        
    all_chunks = []
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None, None
        
    for filename in pdf_files:
        file_path = os.path.join(docs_dir, filename)
        print(f"Ingesting real document: {filename}...")
        pages = load_pdf(file_path)
        if max_pages:
            print(f"Limiting to first {max_pages} pages to save memory...")
            pages = pages[:max_pages]
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)
            
    if not all_chunks:
        return None, None
        
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embedded_chunks = embed_documents(all_chunks)
    
    dimension = len(embedded_chunks[0]["embedding"])
    vector_store = VectorStore(dimension=dimension)
    vector_store.add_chunks(embedded_chunks)
    
    print("Building BM25 Index...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.build(all_chunks)
    
    return vector_store, bm25_retriever

def calculate_recall(retrieved_chunks, expected_sources, expected_pages):
    for chunk in retrieved_chunks:
        if chunk.get("source") in expected_sources and chunk.get("page_number") in expected_pages:
            return 1
    return 0

def get_first_correct_rank(retrieved_chunks, expected_sources, expected_pages):
    for i, chunk in enumerate(retrieved_chunks):
        if chunk.get("source") in expected_sources and chunk.get("page_number") in expected_pages:
            return i + 1
    return 999

def run_real_evaluation(dataset_path, docs_dir):
    print("\n--- Starting REAL RAG Retrieval Evaluation (HYBRID SEARCH EXPERIMENT) ---")
    dataset = load_dataset(dataset_path)
    
    # Use full dataset and all pages
    max_pages = None
        
    print("[Evaluation] Embedding model loaded via app.embeddings")
    vector_store, bm25_retriever = build_real_stores(docs_dir, max_pages=max_pages)
    
    if not vector_store:
        return
        
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever)
    total = len(dataset)
    
    print(f"\nEvaluating {total} ground-truth questions against real PDFs...\n")
    print("[Evaluation] Retrieval Phase (FAISS & Hybrid) started...")
    
    results_cache = []
    for item in dataset:
        query = item["question"]
        query_emb = embed_query(query)
        
        # We need top_k=10 for candidates before reranking
        faiss_10 = vector_store.search(query_emb, top_k=10)
        hybrid_10 = hybrid_retriever.search(query, query_emb, top_k=10)
        
        results_cache.append({
            "item": item,
            "faiss_10": faiss_10,
            "hybrid_10": hybrid_10
        })
        
    print("[Evaluation] Retrieval completed.")
    
    # Free up Embedding memory completely before Reranking
    from app.embeddings import release_embeddings
    release_embeddings()
    gc.collect()
    
    print("[Evaluation] Reranking Phase started...\n")
    
    metrics = {
        'A': {'r1': 0, 'r3': 0, 'r5': 0}, # FAISS
        'B': {'r1': 0, 'r3': 0, 'r5': 0}, # FAISS + Reranker
        'C': {'r1': 0, 'r3': 0, 'r5': 0}, # Hybrid
        'D': {'r1': 0, 'r3': 0, 'r5': 0}  # Hybrid + Reranker
    }
    
    stats = {
        'faiss_correct_r1': 0,
        'hybrid_correct_r1': 0,
        'hybrid_rerank_correct_r1': 0,
        'faiss_to_hybrid_rerank_improved': 0,
        'faiss_to_hybrid_rerank_degraded': 0,
    }
    
    for cache in results_cache:
        item = cache["item"]
        q = item["question"]
        expected_sources = item["expected_sources"]
        expected_pages = item["expected_pages"]
        
        faiss_10 = cache["faiss_10"]
        hybrid_10 = cache["hybrid_10"]
        
        # Rerank both pipelines (model is lazy loaded)
        faiss_reranked = rerank(q, faiss_10, top_k=5)
        hybrid_reranked = rerank(q, hybrid_10, top_k=5)
        
        # Calculate Base Metrics
        for pipeline_key, chunks in [('A', faiss_10), ('B', faiss_reranked), ('C', hybrid_10), ('D', hybrid_reranked)]:
            metrics[pipeline_key]['r1'] += calculate_recall(chunks[:1], expected_sources, expected_pages)
            metrics[pipeline_key]['r3'] += calculate_recall(chunks[:3], expected_sources, expected_pages)
            metrics[pipeline_key]['r5'] += calculate_recall(chunks[:5], expected_sources, expected_pages)
            
        f_rank = get_first_correct_rank(faiss_10, expected_sources, expected_pages)
        h_rank = get_first_correct_rank(hybrid_10, expected_sources, expected_pages)
        hr_rank = get_first_correct_rank(hybrid_reranked, expected_sources, expected_pages)
        
        if f_rank == 1:
            stats['faiss_correct_r1'] += 1
        if h_rank == 1:
            stats['hybrid_correct_r1'] += 1
        if hr_rank == 1:
            stats['hybrid_rerank_correct_r1'] += 1
            
        if hr_rank < f_rank:
            stats['faiss_to_hybrid_rerank_improved'] += 1
        elif hr_rank > f_rank:
            stats['faiss_to_hybrid_rerank_degraded'] += 1
            
    print("[Evaluation] Reranking completed.")
    release_reranker()
    print("[Evaluation] Models/resources released.")
    
    print("\n==============================")
    print("   REAL EVALUATION REPORT     ")
    print("==============================")
    print(f"Total questions: {total}\n")
    
    print("A. FAISS ONLY:")
    print(f"Recall@1: {(metrics['A']['r1'] / total) * 100:.0f}%")
    print(f"Recall@3: {(metrics['A']['r3'] / total) * 100:.0f}%")
    print(f"Recall@5: {(metrics['A']['r5'] / total) * 100:.0f}%\n")
    
    print("B. FAISS + CROSS-ENCODER:")
    print(f"Recall@1: {(metrics['B']['r1'] / total) * 100:.0f}%")
    print(f"Recall@3: {(metrics['B']['r3'] / total) * 100:.0f}%")
    print(f"Recall@5: {(metrics['B']['r5'] / total) * 100:.0f}%\n")
    
    print("C. HYBRID (FAISS + BM25 + RRF):")
    print(f"Recall@1: {(metrics['C']['r1'] / total) * 100:.0f}%")
    print(f"Recall@3: {(metrics['C']['r3'] / total) * 100:.0f}%")
    print(f"Recall@5: {(metrics['C']['r5'] / total) * 100:.0f}%\n")
    
    print("D. HYBRID + CROSS-ENCODER:")
    print(f"Recall@1: {(metrics['D']['r1'] / total) * 100:.0f}%")
    print(f"Recall@3: {(metrics['D']['r3'] / total) * 100:.0f}%")
    print(f"Recall@5: {(metrics['D']['r5'] / total) * 100:.0f}%\n")
    
    print("RANKING COMPARISON STATS (FAISS vs HYBRID+RERANK):")
    print(f"- Number of questions where FAISS Rank 1 was correct: {stats['faiss_correct_r1']}")
    print(f"- Number where Hybrid+Rerank Rank 1 was correct: {stats['hybrid_rerank_correct_r1']}")
    print(f"- Number where Hybrid+Rerank improved ranking: {stats['faiss_to_hybrid_rerank_improved']}")
    print(f"- Number where Hybrid+Rerank degraded ranking: {stats['faiss_to_hybrid_rerank_degraded']}")
    print("==============================\n")

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "real_dataset.json")
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'documents'))
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    run_real_evaluation(dataset_path, docs_dir)
