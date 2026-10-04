import os
import sys
import json
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.document_loader import load_pdf
from app.chunker import chunk_documents
from app.embeddings import embed_documents, embed_query
from app.vector_store import VectorStore
from app.reranker import rerank, release_reranker

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_real_vector_store(docs_dir, max_pages=None):
    if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
        print(f"Warning: No documents found in {docs_dir}")
        return None
        
    all_chunks = []
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None
        
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
        return None
        
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embedded_chunks = embed_documents(all_chunks)
    
    dimension = len(embedded_chunks[0]["embedding"])
    store = VectorStore(dimension=dimension)
    store.add_chunks(embedded_chunks)
    
    return store

def calculate_recall(retrieved_chunks, expected_sources, expected_pages):
    for chunk in retrieved_chunks:
        if chunk.get("source") in expected_sources and chunk.get("page_number") in expected_pages:
            return 1
    return 0

def run_real_evaluation(dataset_path, docs_dir, small_mode=True):
    print("\n--- Starting REAL RAG Retrieval Evaluation ---")
    dataset = load_dataset(dataset_path)
    
    if small_mode:
        print("[Memory Constraint Mode] Using first 5 questions and first 40 pages of PDF.")
        dataset = dataset[:5]
        max_pages = 40
    else:
        max_pages = None
        
    print("[Evaluation] Embedding model loaded via app.embeddings")
    store = build_real_vector_store(docs_dir, max_pages=max_pages)
    total = len(dataset)
    
    print(f"\nEvaluating {total} ground-truth questions against real PDFs...\n")
    print("[Evaluation] FAISS retrieval started...")
    
    # 1. Run all FAISS retrievals first to batch memory usage
    results_cache = []
    for item in dataset:
        if store is None or store.index.ntotal == 0:
            results_cache.append({"item": item, "faiss_5": [], "faiss_10": []})
            continue
            
        query_emb = embed_query(item["question"])
        faiss_5 = store.search(query_emb, top_k=5)
        faiss_10 = store.search(query_emb, top_k=10)
        results_cache.append({"item": item, "faiss_5": faiss_5, "faiss_10": faiss_10})
        
    print("[Evaluation] FAISS retrieval completed.")
    
    # 2. Free up FAISS and Embedding memory completely
    from app.embeddings import release_embeddings
    release_embeddings()
    del store
    gc.collect()
    
    print("[Evaluation] Reranking started...")
    
    faiss_r1, faiss_r3, faiss_r5 = 0, 0, 0
    rerank_r1, rerank_r3, rerank_r5 = 0, 0, 0
    
    # 3. Run Reranker on cached FAISS results
    for cache in results_cache:
        item = cache["item"]
        expected_sources = item["expected_sources"]
        expected_pages = item["expected_pages"]
        
        faiss_chunks = cache["faiss_5"]
        faiss_candidates_10 = cache["faiss_10"]
        
        # rerank() will automatically lazy-load the Cross-Encoder model here
        print(f"[Reranking] Question: {item['question']}")
        rerank_chunks = rerank(item["question"], faiss_candidates_10, top_k=5)
        
        # Calculate Base Metrics
        faiss_r1 += calculate_recall(faiss_chunks[:1], expected_sources, expected_pages)
        faiss_r3 += calculate_recall(faiss_chunks[:3], expected_sources, expected_pages)
        faiss_r5 += calculate_recall(faiss_chunks[:5], expected_sources, expected_pages)
        
        # Calculate Reranked Metrics
        rerank_r1 += calculate_recall(rerank_chunks[:1], expected_sources, expected_pages)
        rerank_r3 += calculate_recall(rerank_chunks[:3], expected_sources, expected_pages)
        rerank_r5 += calculate_recall(rerank_chunks[:5], expected_sources, expected_pages)
            
    print("[Evaluation] Reranking completed.")
    release_reranker()
    print("[Evaluation] Models/resources released.")
    
    print("\n==============================")
    print("   REAL EVALUATION REPORT     ")
    print("==============================")
    print(f"Total questions: {total}\n")
    
    print("BEFORE RERANKING (FAISS Only):")
    print(f"Recall@1: {(faiss_r1 / total) * 100:.1f}%")
    print(f"Recall@3: {(faiss_r3 / total) * 100:.1f}%")
    print(f"Recall@5: {(faiss_r5 / total) * 100:.1f}%\n")
    
    print("AFTER RERANKING (Cross-Encoder):")
    print(f"Recall@1: {(rerank_r1 / total) * 100:.1f}%")
    print(f"Recall@3: {(rerank_r3 / total) * 100:.1f}%")
    print(f"Recall@5: {(rerank_r5 / total) * 100:.1f}%")
    print("==============================\n")

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "real_dataset.json")
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'documents'))
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    run_real_evaluation(dataset_path, docs_dir, small_mode=True)
