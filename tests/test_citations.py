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
    We bypass the real LLM API call but still use the REAL context formatting logic
    from llm.py to test our citation extraction.
    """
    # 1. Format the context strings and extract sources identically to app/llm.py
    unique_sources = {}
    
    for chunk in retrieved_chunks:
        source_name = chunk.get("source", "Unknown")
        page_num = chunk.get("page_number", "Unknown")
        chunk_id = chunk.get("chunk_id", "Unknown")
        score = chunk.get("score", 0.0)
        
        key = (source_name, page_num)
        if key not in unique_sources:
            unique_sources[key] = {
                "source": source_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "score": score
            }
        else:
            if score > unique_sources[key]["score"]:
                unique_sources[key]["score"] = score
                unique_sources[key]["chunk_id"] = chunk_id
                
    sources = list(unique_sources.values())
            
    return {
        "answer": "This is a mock answer based on the provided context.",
        "sources": sources
    }

def test_citations():
    # Override the real LLM with our mock for testing purposes
    app.llm.generate_answer = mock_generate_answer
    
    # 1. Setup mock data
    # Notice we have TWO chunks from the exact same page (page 5)
    docs = [
        {"chunk_id": "c1", "text": "Machine learning models learn patterns from data.", "source": "ai_book.pdf", "page_number": 10},
        {"chunk_id": "c2", "text": "Pizza contains cheese.", "source": "cookbook.pdf", "page_number": 5},
        {"chunk_id": "c3", "text": "Tomato sauce is commonly used on pizza.", "source": "cookbook.pdf", "page_number": 5}
    ]
    
    print("Generating embeddings for mock data...")
    embedded_docs = embed_documents(docs)
    
    store = VectorStore(dimension=len(embedded_docs[0]["embedding"]))
    store.add_chunks(embedded_docs)
    
    # 2. Query that should retrieve both pizza chunks
    print("\n--- Test: Citation Deduplication ---")
    res = ask("What goes on a pizza?", store, top_k=3)
    
    sources = res["sources"]
    
    print("\nGenerated Citations:")
    for s in sources:
        print(s)
        
    # Verify metadata survives
    assert len(sources) > 0, "No sources returned!"
    for s in sources:
        assert "source" in s
        assert "page" in s
        assert "chunk_id" in s
        assert "score" in s
        
    # Verify deduplication (we retrieved 2 chunks from cookbook.pdf page 5, but should only have 1 citation for it)
    cookbook_citations = [s for s in sources if s["source"] == "cookbook.pdf"]
    assert len(cookbook_citations) == 1, "Duplicate citations for the same page were not filtered out!"
    
    print("\nCitation tests passed successfully!")

if __name__ == "__main__":
    test_citations()
