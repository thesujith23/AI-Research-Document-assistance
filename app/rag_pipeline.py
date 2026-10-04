import logging
from app.embeddings import embed_query
from app.llm import generate_answer
from app.reranker import rerank

# Configure logging to display the pipeline flow to the terminal
logging.basicConfig(level=logging.INFO, format="%(message)s")

def ask(question: str, vector_store, top_k: int = 3, use_reranker: bool = False) -> dict:
    """
    The main RAG pipeline function.
    Connects the user question to retrieval, and then to generation.
    """
    logging.info("\n=== RAG PIPELINE START ===")
    logging.info(f"QUESTION\n↓\n'{question}'")
    
    # 1. Generate query embedding
    logging.info("↓\nQUERY EMBEDDING")
    query_emb = embed_query(question)
    
    # 2. Search FAISS
    logging.info("↓\nRETRIEVAL (FAISS)")
    if use_reranker:
        # Retrieve more candidates if we are going to rerank them
        retrieved_chunks = vector_store.search(query_emb, top_k=10)
    else:
        retrieved_chunks = vector_store.search(query_emb, top_k=top_k)
        
    # 3. Log FAISS chunks
    logging.info("↓\nFAISS CANDIDATES")
    if not retrieved_chunks:
        logging.info("  - (No chunks retrieved)")
    for i, chunk in enumerate(retrieved_chunks):
        logging.info(f"  - FAISS Match {i+1}: {chunk['source']} (Page {chunk['page_number']}) [FAISS Score: {chunk['score']:.4f}]")
        
    # 4. Reranking Step
    if use_reranker:
        logging.info("↓\nRERANKING (Cross-Encoder)")
        retrieved_chunks = rerank(question, retrieved_chunks, top_k=top_k)
        
        logging.info("↓\nRERANKED CANDIDATES")
        for i, chunk in enumerate(retrieved_chunks):
            logging.info(f"  - Reranked Match {i+1}: {chunk['source']} (Page {chunk['page_number']}) [Reranker Score: {chunk['reranker_score']:.4f}]")

    # 5. Pass chunks to LLM
    logging.info("↓\nLLM")
    result = generate_answer(question, retrieved_chunks)
    
    # 6. Output Answer
    logging.info("↓\nANSWER")
    logging.info(result['answer'])
    logging.info("=== RAG PIPELINE END ===\n")
    
    return result
