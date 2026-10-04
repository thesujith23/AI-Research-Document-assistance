import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.embeddings import embed_documents
from app.vector_store import VectorStore
from app.rag_pipeline import ask
import app.llm

def mock_generate_answer(question, retrieved_chunks):
    """
    A mock LLM to test the pipeline flow without spending API credits 
    or requiring an API key.
    """
    if not retrieved_chunks:
        answer = "I cannot answer this question based on the provided documents."
    else:
        answer = f"MOCK ANSWER: Based on {len(retrieved_chunks)} retrieved chunks, here is a simulated answer."
        
    sources = []
    for c in retrieved_chunks:
        src = {"source": c["source"], "page_number": c["page_number"]}
        if src not in sources:
            sources.append(src)
            
    return {
        "answer": answer,
        "sources": sources,
        "prompt_used": "MOCK PROMPT"
    }

def test_rag_pipeline():
    # 1. Override the real LLM with our mock for testing purposes
    app.llm.generate_answer = mock_generate_answer
    
    # 2. Setup mock data and initialize the vector store
    docs = [
        {"chunk_id": "c1", "text": "The capital of France is Paris.", "source": "geo.pdf", "page_number": 1},
        {"chunk_id": "c2", "text": "Python is a programming language.", "source": "tech.pdf", "page_number": 2}
    ]
    
    print("Generating embeddings for mock data...")
    embedded_docs = embed_documents(docs)
    
    store = VectorStore(dimension=len(embedded_docs[0]["embedding"]))
    store.add_chunks(embedded_docs)
    
    # 3. Test retrieving relevant chunks
    print("\n--- Test 1: Relevant Question ---")
    res1 = ask("What is the capital of France?", store, top_k=1)
    
    # Verify metadata is preserved
    assert len(res1["sources"]) > 0
    assert res1["sources"][0]["source"] == "geo.pdf"
    assert "MOCK ANSWER" in res1["answer"]
    
    # 4. Test question with no relevant context
    print("\n--- Test 2: Irrelevant Question ---")
    # Even if we ask an irrelevant question, the vector store will still return the "closest" match.
    # In a real scenario, the LLM reads that closest match, realizes it has nothing to do with the question,
    # and outputs "I cannot answer this."
    res2 = ask("How do I bake a cake?", store, top_k=1)
    assert "MOCK ANSWER" in res2["answer"]

    print("\nRAG Pipeline tests completed successfully!")

if __name__ == "__main__":
    test_rag_pipeline()
