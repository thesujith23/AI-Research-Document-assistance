class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever):
        """
        Initializes the HybridRetriever with both FAISS and BM25 implementations.
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def search(self, query, query_embedding, top_k=5, k_rrf=60):
        """
        Performs hybrid search using Reciprocal Rank Fusion (RRF).
        
        RRF formula: RRF_score = sum( 1 / (k + rank) ) for each retriever
        """
        # 1. Retrieve candidates from both systems (get a larger set to fuse)
        faiss_candidates = self.vector_store.search(query_embedding, top_k=10)
        bm25_candidates = self.bm25_retriever.search(query, top_k=10)
        
        # 2. Combine results using RRF
        rrf_map = {} # Maps chunk_id to its combined data and score
        
        # Process FAISS results
        for rank, chunk in enumerate(faiss_candidates):
            chunk_id = chunk["chunk_id"]
            if chunk_id not in rrf_map:
                rrf_map[chunk_id] = chunk.copy()
                rrf_map[chunk_id]["faiss_score"] = chunk.get("score", 0.0)
                rrf_map[chunk_id]["bm25_score"] = 0.0
                rrf_map[chunk_id]["rrf_score"] = 0.0
            
            rrf_map[chunk_id]["faiss_score"] = chunk.get("score", 0.0)
            rrf_map[chunk_id]["rrf_score"] += 1.0 / (k_rrf + rank + 1)
            
        # Process BM25 results
        for rank, chunk in enumerate(bm25_candidates):
            chunk_id = chunk["chunk_id"]
            if chunk_id not in rrf_map:
                rrf_map[chunk_id] = chunk.copy()
                rrf_map[chunk_id]["faiss_score"] = 0.0
                rrf_map[chunk_id]["rrf_score"] = 0.0
                
            rrf_map[chunk_id]["bm25_score"] = chunk.get("bm25_score", 0.0)
            rrf_map[chunk_id]["rrf_score"] += 1.0 / (k_rrf + rank + 1)
            
        # 3. Sort by RRF score
        fused_results = list(rrf_map.values())
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Update the generic 'score' key so downstream rerankers/LLMs can see it uniformly
        for res in fused_results:
            res["score"] = res["rrf_score"]
            
        return fused_results[:top_k]
