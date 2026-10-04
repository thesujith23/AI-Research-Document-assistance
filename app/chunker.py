def chunk_documents(documents: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Splits the text of documents into smaller chunks with a specified character overlap.
    
    Args:
        documents (list[dict]): The loaded documents, where each dict has 'text', 'source', and 'page_number'.
        chunk_size (int): The maximum number of characters in each chunk.
        overlap (int): The number of characters to overlap between consecutive chunks.
        
    Returns:
        list[dict]: A list of chunks, preserving metadata and including a unique chunk_id.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size")

    all_chunks = []
    
    for doc in documents:
        text = doc.get("text", "")
        source = doc.get("source", "unknown_source")
        page_number = doc.get("page_number", 0)
        
        # Skip empty documents/pages
        if not text:
            continue
            
        start = 0
        chunk_index = 1
        text_length = len(text)
        
        # Slide a window over the text to create chunks
        while start < text_length:
            # Determine the end index for the current chunk
            end = start + chunk_size
            
            # Extract the text for this chunk
            chunk_text = text[start:end]
            
            # Create a unique ID for the chunk (e.g., 'document.pdf_p1_c1')
            chunk_id = f"{source}_p{page_number}_c{chunk_index}"
            
            # Save the chunk along with its metadata
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": source,
                "page_number": page_number
            })
            
            # If this chunk reached or exceeded the end of the text, we're done with this document
            if end >= text_length:
                break
                
            # Move the start pointer forward
            # We subtract the overlap so the next chunk starts slightly before this one ended
            start += (chunk_size - overlap)
            chunk_index += 1
            
    return all_chunks
