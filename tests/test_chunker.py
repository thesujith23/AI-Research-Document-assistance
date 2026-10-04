import os
import sys

# Add the parent directory to sys.path so we can import the app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.chunker import chunk_documents

def test_chunker():
    # 1. Setup mock data (simulating the output of document_loader.py)
    # The text is 10 characters long: "ABCDEFGHIJ"
    mock_documents = [
        {
            "text": "ABCDEFGHIJ",
            "page_number": 1,
            "source": "test_doc.pdf"
        }
    ]
    
    # 2. Run chunking
    # We use a very small chunk_size and overlap for testing purposes
    chunk_size = 5
    overlap = 2
    chunks = chunk_documents(mock_documents, chunk_size=chunk_size, overlap=overlap)
    
    # 3. Verify chunk creation and count
    # Expected chunks:
    # Chunk 1 (0:5) -> "ABCDE"
    # Chunk 2 (3:8) -> "DEFGH" (Starts at 5-2=3)
    # Chunk 3 (6:10) -> "GHIJ" (Starts at 8-2=6)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    
    # 4. Verify chunk size limits and content
    assert chunks[0]["text"] == "ABCDE", f"Chunk 1 text mismatch: {chunks[0]['text']}"
    assert chunks[1]["text"] == "DEFGH", f"Chunk 2 text mismatch: {chunks[1]['text']}"
    assert chunks[2]["text"] == "GHIJ", f"Chunk 3 text mismatch: {chunks[2]['text']}"
    
    # 5. Verify overlap works correctly
    # The last 2 characters of Chunk 1 ("DE") should equal the first 2 characters of Chunk 2 ("DE")
    assert chunks[0]["text"][-overlap:] == chunks[1]["text"][:overlap], "Overlap test failed between chunk 1 and 2"
    
    # 6. Verify metadata is preserved
    for chunk in chunks:
        assert chunk["source"] == "test_doc.pdf", "Source metadata lost"
        assert chunk["page_number"] == 1, "Page number metadata lost"
        assert "chunk_id" in chunk, "Chunk ID missing"
        
    # 7. Verify chunk IDs are unique
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(set(chunk_ids)) == len(chunk_ids), f"Chunk IDs are not unique: {chunk_ids}"
    
    print("All chunker tests passed successfully!\n")
    
    print("--- Example Original Text ---")
    print(mock_documents[0]["text"])
    
    print("\n--- Example Chunking Output ---")
    for c in chunks:
        print(f"[{c['chunk_id']}] Length: {len(c['text'])} | Text: '{c['text']}'")

if __name__ == "__main__":
    test_chunker()
