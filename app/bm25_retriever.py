import string
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def _tokenize(self, text):
        # A simple tokenizer: lowercases and removes punctuation
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, " ")
        return text.split()

    def build(self, chunks):
        """Builds the BM25 index from a list of chunk dictionaries."""
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5):
        """Searches the BM25 index and returns top_k chunks with their BM25 score."""
        if self.bm25 is None or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            # Create a copy so we don't mutate the original dictionary
            chunk = self.chunks[i].copy()
            chunk["bm25_score"] = float(scores[i])
            # Set a generic 'score' key for uniform processing if needed
            chunk["score"] = chunk["bm25_score"] 
            results.append(chunk)
            
        return results
