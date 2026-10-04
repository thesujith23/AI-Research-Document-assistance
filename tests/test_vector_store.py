import os
import sys

# Add parent directory to path so we can import the app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.embeddings import embed_documents, embed_query
from app.vector_store import VectorStore

def test_vector_store():
    # 1. Setup Document Chunks
    docs = [
        {"chunk_id": "doc1", "text": "Machine learning models learn patterns from training data.", "source": "ai_book.pdf", "page_number": 10},
        {"chunk_id": "doc2", "text": "Deep learning uses neural networks to process information.", "source": "ai_book.pdf", "page_number": 15},
        {"chunk_id": "doc3", "text": "Pizza contains cheese, tomato sauce, and other toppings.", "source": "cookbook.pdf", "page_number": 5}
    ]
    
    print("Generating embeddings for documents...")
    # This adds the 'embedding' key to each doc dict
    embedded_docs = embed_documents(docs)
    
    # 2. Initialize Vector Store
    # We dynamically get the dimension from the first embedded document (384 for our model)
    dimension = len(embedded_docs[0]["embedding"])
    print(f"Initializing VectorStore with dimension {dimension}")
    store = VectorStore(dimension=dimension)
    
    # 3. Add chunks to FAISS
    store.add_chunks(embedded_docs)
    assert store.index.ntotal == 3, "Failed to add all vectors to FAISS"
    
    # 4. Perform a Semantic Search Query
    query_text = "How do machine learning systems learn?"
    print(f"\nQuery: '{query_text}'")
    
    query_emb = embed_query(query_text)
    
    # Retrieve top 2 results
    top_k = 2
    results = store.search(query_emb, top_k=top_k)
    
    # 5. Verify results
    assert len(results) == 2, f"Expected {top_k} results"
    
    print("\n--- Search Results ---")
    for i, res in enumerate(results):
        print(f"Rank {i+1} (Score: {res['score']:.4f})")
        print(f"Source: {res['source']}, Page: {res['page_number']}")
        print(f"Text: {res['text']}\n")
        
    # The highest scoring result should be doc1
    assert results[0]["chunk_id"] == "doc1", "Semantic search failed to rank the most relevant document first!"
    
    # The second highest scoring result should be doc2
    assert results[1]["chunk_id"] == "doc2", "Semantic search failed to rank the second most relevant document!"
    
    # The Pizza document should NOT be in the top 2
    for res in results:
        assert res["chunk_id"] != "doc3", "Irrelevant document was incorrectly retrieved!"
        
    # 6. Test Edge Cases and Validation
    try:
        store.search([], top_k=1)
        assert False, "Should have raised ValueError on empty query"
    except ValueError:
        pass
        
    try:
        store.search([0.0]*10, top_k=1) # Wrong dimension (10 instead of 384)
        assert False, "Should have raised ValueError on dimension mismatch"
    except ValueError:
        pass
        
    print("All vector store tests passed successfully!")

if __name__ == "__main__":
    test_vector_store()
