import os
import sys
import json

# Add parent directory to path so we can import the app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.embeddings import embed_documents, embed_query
from app.vector_store import VectorStore
from app.reranker import rerank

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def setup_mock_vector_store():
    """
    Creates a mock vector store loaded with documents that answer our evaluation dataset.
    """
    docs = [
        {"chunk_id": "c1", "text": "Paris is the capital and most populous city of France.", "source": "geography.pdf", "page_number": 1},
        {"chunk_id": "c2", "text": "Plants use a process called photosynthesis to make their own food using sunlight.", "source": "biology.pdf", "page_number": 5},
        {"chunk_id": "c3", "text": "Apollo 11 was the American spaceflight that first landed humans on the Moon in 1969.", "source": "history.pdf", "page_number": 12},
        {"chunk_id": "c4", "text": "A classic pizza is made with dough, tomato sauce, and mozzarella cheese.", "source": "cookbook.pdf", "page_number": 3},
        {"chunk_id": "c5", "text": "Neural networks learn by adjusting weights through backpropagation and gradient descent.", "source": "ai_book.pdf", "page_number": 25},
        {"chunk_id": "c6", "text": "Romeo and Juliet is a tragedy written by William Shakespeare.", "source": "literature.pdf", "page_number": 8},
        {"chunk_id": "c7", "text": "The boiling point of water is 100 degrees Celsius at 1 atmosphere of pressure.", "source": "physics.pdf", "page_number": 2},
        {"chunk_id": "c8", "text": "Jupiter is the largest planet in the Solar System.", "source": "astronomy.pdf", "page_number": 4},
        {"chunk_id": "c9", "text": "In Python, variables are declared simply by assigning a value to a name, like x = 5.", "source": "programming.pdf", "page_number": 15},
        {"chunk_id": "c10", "text": "The speed of light in a vacuum is approximately 299,792 kilometers per second.", "source": "physics.pdf", "page_number": 20},
        
        # Add some distractors (noise) to make the retrieval harder!
        {"chunk_id": "d1", "text": "Water freezes at 0 degrees Celsius.", "source": "physics.pdf", "page_number": 3},
        {"chunk_id": "d2", "text": "Saturn is known for its ring system.", "source": "astronomy.pdf", "page_number": 5},
        {"chunk_id": "d3", "text": "William Shakespeare also wrote Hamlet.", "source": "literature.pdf", "page_number": 9},
        {"chunk_id": "d4", "text": "Python is named after Monty Python's Flying Circus.", "source": "programming.pdf", "page_number": 1},
    ]
    
    print("Generating embeddings for mock knowledge base...")
    embedded_docs = embed_documents(docs)
    
    store = VectorStore(dimension=len(embedded_docs[0]["embedding"]))
    store.add_chunks(embedded_docs)
    
    return store

def calculate_recall(retrieved_chunks, expected_sources, expected_pages):
    for chunk in retrieved_chunks:
        if chunk.get("source") in expected_sources and chunk.get("page_number") in expected_pages:
            return 1
    return 0

def run_evaluation(dataset_path):
    print("\n--- Starting RAG Retrieval Evaluation ---")
    
    dataset = load_dataset(dataset_path)
    store = setup_mock_vector_store()
    
    total = len(dataset)
    
    # FAISS Base Metrics
    faiss_r1, faiss_r3, faiss_r5 = 0, 0, 0
    # Reranker Metrics
    rerank_r1, rerank_r3, rerank_r5 = 0, 0, 0
    
    print(f"\nEvaluating {total} questions...\n")
    
    for item in dataset:
        question = item["question"]
        expected_sources = item["expected_sources"]
        expected_pages = item["expected_pages"]
        
        query_emb = embed_query(question)
        
        # BEFORE RERANKING: FAISS top 5
        faiss_chunks = store.search(query_emb, top_k=5)
        
        # AFTER RERANKING: FAISS top 10 -> Rerank top 5
        faiss_candidates_10 = store.search(query_emb, top_k=10)
        rerank_chunks = rerank(question, faiss_candidates_10, top_k=5)
        
        # Calculate Base Metrics
        faiss_r1 += calculate_recall(faiss_chunks[:1], expected_sources, expected_pages)
        faiss_r3 += calculate_recall(faiss_chunks[:3], expected_sources, expected_pages)
        faiss_r5 += calculate_recall(faiss_chunks[:5], expected_sources, expected_pages)
        
        # Calculate Reranked Metrics
        rerank_r1 += calculate_recall(rerank_chunks[:1], expected_sources, expected_pages)
        rerank_r3 += calculate_recall(rerank_chunks[:3], expected_sources, expected_pages)
        rerank_r5 += calculate_recall(rerank_chunks[:5], expected_sources, expected_pages)
        
    print("\n==============================")
    print("      EVALUATION REPORT       ")
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
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    run_evaluation(dataset_path)

