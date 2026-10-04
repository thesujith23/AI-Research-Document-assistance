import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.reranker import rerank

class TestReranker(unittest.TestCase):
    def test_reranking(self):
        query = "What is the capital of France?"
        
        # Provide chunks where the BEST answer is at the bottom of the list
        chunks = [
            {"chunk_id": "c1", "text": "France is a country in Europe.", "source": "test.pdf", "page_number": 1, "score": 0.5},
            {"chunk_id": "c2", "text": "Paris is a very nice city with the Eiffel Tower.", "source": "test.pdf", "page_number": 2, "score": 0.4},
            {"chunk_id": "c3", "text": "Paris is the capital of France.", "source": "test.pdf", "page_number": 3, "score": 0.3}
        ]
        
        reranked = rerank(query, chunks, top_k=3)
        
        self.assertEqual(len(reranked), 3)
        
        # The exact answer should now be pushed to the top
        self.assertEqual(reranked[0]["chunk_id"], "c3")
        self.assertTrue("reranker_score" in reranked[0])
        
        # Score of the top chunk should be higher than the others
        self.assertGreater(reranked[0]["reranker_score"], reranked[1]["reranker_score"])
        self.assertGreater(reranked[1]["reranker_score"], reranked[2]["reranker_score"])

if __name__ == '__main__':
    unittest.main()
