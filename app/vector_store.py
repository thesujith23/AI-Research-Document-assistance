import numpy as np
import faiss

class VectorStore:
    """
    A simple vector database using FAISS to store and search text embeddings.
    """
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
            
        self.dimension = dimension
        
        # We use IndexFlatIP (Inner Product) for our FAISS index.
        # Cosine Similarity is simply the Inner Product of two normalized vectors.
        # By normalizing our vectors before adding/searching, IndexFlatIP computes Cosine Similarity!
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # FAISS only stores the math (vectors) and assigns them an integer ID (0, 1, 2...).
        # It does NOT store strings or metadata.
        # We use this list to map FAISS's integer IDs back to our original chunk dictionaries.
        self.chunks_metadata = []
        
    def add_chunks(self, chunks: list[dict]):
        """
        Extracts embeddings from the chunks, adds them to the FAISS index, 
        and stores the metadata.
        """
        if not chunks:
            return
            
        vectors = []
        
        for chunk in chunks:
            emb = chunk.get("embedding")
            if not emb:
                raise ValueError(f"Chunk missing embedding: {chunk.get('chunk_id')}")
            if len(emb) != self.dimension:
                raise ValueError(f"Expected dimension {self.dimension}, got {len(emb)}")
                
            vectors.append(emb)
            
            # Store a copy of the chunk WITHOUT the large embedding vector
            # We don't need to keep the vector in python memory since FAISS has it.
            metadata = {k: v for k, v in chunk.items() if k != "embedding"}
            self.chunks_metadata.append(metadata)
            
        # Convert list of vectors to a 2D numpy array of 32-bit floats (required by FAISS)
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # Normalize the vectors so Inner Product acts exactly like Cosine Similarity
        faiss.normalize_L2(vectors_np)
        
        # Add the vectors to the FAISS index
        self.index.add(vectors_np)
        
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Searches the FAISS index for the 'top_k' most similar vectors to the query.
        Returns the original chunks with a 'score' attached.
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")
            
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query dimension {len(query_embedding)} does not match index dimension {self.dimension}.")
            
        if self.index.ntotal == 0:
            return []
            
        # Limit top_k so we don't ask for more results than we have stored
        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []
            
        # Convert the single query into a 2D numpy array (1 row, D columns)
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Normalize the query vector for Cosine Similarity
        faiss.normalize_L2(query_np)
        
        # Perform the search
        # FAISS returns two 2D arrays:
        # similarities: the inner product scores
        # indices: the integer IDs of the closest vectors
        similarities, indices = self.index.search(query_np, k)
        
        results = []
        # We only did 1 query, so we look at the first row (index 0) of the results
        for j in range(k):
            idx = indices[0][j]
            score = similarities[0][j]
            
            if idx != -1 and idx < len(self.chunks_metadata):
                # Retrieve the stored metadata using the FAISS ID
                chunk_data = self.chunks_metadata[idx].copy()
                chunk_data["score"] = float(score)
                results.append(chunk_data)
                
        return results
