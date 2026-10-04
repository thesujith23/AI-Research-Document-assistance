# AI Research & Document Assistant

This project will eventually serve as an AI Research & Document Assistant, leveraging the power of Retrieval-Augmented Generation (RAG) to allow users to ask questions and extract insights directly from their own documents.

## Project Overview

The core purpose of this assistant is to take raw documents (like PDFs, initially), process them, and provide a conversational interface for a user to query information contained within those documents.

## The RAG Pipeline

This project will be built incrementally, following the major stages of a standard RAG pipeline:

1. **Document Loading**: Ingesting raw documents (e.g., PDFs, Text files) from the `data/documents/` directory.
2. **Text Splitting / Chunking**: Breaking down large documents into smaller, manageable chunks of text that can fit within an LLM's context window.
3. **Embedding**: Converting the text chunks into mathematical vector representations (embeddings) that capture the semantic meaning of the text.
4. **Vector Database / Storage**: Storing the embeddings in a database for efficient similarity search and retrieval.
5. **Retrieval**: Taking a user's query, embedding it, and searching the vector database to find the most relevant text chunks.
6. **Generation**: Passing the relevant text chunks along with the user's query to a Large Language Model (LLM) to generate an accurate and context-aware answer.

Currently, this repository is being built step-by-step.

### Stage 1: Document Ingestion
**What is Document Ingestion in RAG?**
In a RAG application, you can't query what the system doesn't know. Document ingestion is the first step where we take a user's raw files (like PDFs, Word docs, or Text files) and programmatically extract the unstructured text out of them. This turns files into raw text strings that Python can manipulate.

**Why extract text before chunking?**
Extraction must happen before chunking because you can't accurately split a PDF file based on its binary data. We first need the raw, human-readable text. Once we have the full text and page boundaries, we can intelligently divide it into smaller, overlapping chunks (the next stage).

**What is Metadata?**
Metadata is "data about the data." When we extract text, we don't just keep the text itself; we also save metadata like the `page_number` and the `source` (filename). This is crucial for RAG because when the LLM generates an answer, we can use the metadata to provide accurate citations to the user (e.g., "According to page 4 of annual_report.pdf...").

**Example Output Structure**
When a document is ingested, it is structured as a list of dictionaries. For example:
```python
[
    {
        "text": "The quick brown fox jumps over the lazy dog.",
        "page_number": 1,
        "source": "sample_doc.pdf"
    },
    {
        "text": "This is the content of the second page.",
        "page_number": 2,
        "source": "sample_doc.pdf"
    }
]
```

### Stage 2: Text Chunking
**What is chunking?**
Chunking is the process of breaking down long pieces of text (like entire document pages) into smaller, more manageable blocks or "chunks" of text.

**Why is it required in RAG?**
1. **LLM Context Windows:** Large Language Models have a strict limit on how much text they can process at once (the context window). You cannot feed an entire 100-page PDF into an LLM all at once.
2. **Retrieval Accuracy:** If you store an entire page as one vector, the semantic meaning gets diluted. By splitting text into smaller chunks (e.g., a few paragraphs), vector similarity search becomes much more precise, returning only the specific snippets that answer the user's query.

**What is Chunk Size?**
Chunk size is the maximum length of a text chunk. In a character-based approach, a chunk size of 500 means the chunk will contain at most 500 characters. In more advanced approaches, this is measured in "tokens."

**What is Chunk Overlap?**
Overlap means the end of one chunk is repeated at the beginning of the next chunk. This prevents critical context or sentences from being cut exactly in half. For example, if a sentence spans the boundary between Chunk 1 and Chunk 2, an overlap ensures the full sentence is preserved in at least one of the chunks.

**Limitations of Character-Based Chunking**
Character-based chunking is the simplest approach, but it is "dumb". It might slice a chunk right in the middle of a word or sentence because it strictly counts characters. More advanced chunkers (like Recursive Character or Semantic Chunkers) try to split on logical boundaries like periods (`.`), newlines (`\n`), or even shifts in topic.

### Stage 3: Embeddings
**What is an Embedding?**
An embedding is a mathematical representation of text. It converts a chunk of text into an array (or vector) of floating-point numbers. Think of it as a set of coordinates mapping the "meaning" of the text into a high-dimensional space.

**Why does RAG need Embeddings?**
RAG needs to find chunks of text that answer the user's query. If the user asks "How do computers learn?", RAG needs to find chunks talking about "machine learning" or "AI training". Embeddings make this possible because texts with similar meanings will have similar numerical coordinates.

**Why is normal keyword search not enough?**
Keyword search (like `Ctrl+F` or SQL `LIKE`) only finds exact word matches. If you search for "computers", it won't find a document that only uses the word "laptops". Embeddings capture *semantic meaning*, so they understand that "computers" and "laptops" are related.

**What does Semantic Similarity mean?**
It means two pieces of text have the same underlying concept or intent, even if they use completely different words. For example, "I am happy" and "I feel joyful" are semantically similar.

**What is Cosine Similarity?**
Cosine similarity is a mathematical formula used to measure how close two vectors (embeddings) are to each other. It calculates the angle between the two vectors. A score of 1.0 means they are pointing in the exact same direction (highly similar), while a lower score means they are unrelated.

**Why do embedding dimensions matter?**
The dimension refers to the length of the number array (e.g., 384 numbers). A higher dimension means the model can capture more complex nuances of meaning, but it takes more memory and storage space.

**Why use the same model for documents and queries?**
If you embed your documents using Model A (which maps "apple" to coordinate [1, 2, 3]) and embed the user's query using Model B (which maps "apple" to coordinate [9, 8, 7]), the coordinates won't match up. You must use the exact same model so that the query and the documents are placed in the *same mathematical space*.

**The Model Being Used**
We are using `all-MiniLM-L6-v2` via the `sentence-transformers` library. It is lightweight, runs well on a CPU, and produces 384-dimensional embeddings, making it perfect for learning and local development.

### Stage 4: Vector Store
**What is a Vector Store?**
A vector database (or store) is a specialized database designed to store and quickly search through mathematical vectors (embeddings) rather than rows and columns of text.

**Why do we need a Vector Store?**
Generating embeddings for thousands of pages takes time. We can't generate them every single time a user asks a question. We generate them *once*, store them in the vector store, and then we only have to generate an embedding for the user's short query. Furthermore, comparing a query vector against millions of document vectors one-by-one is incredibly slow. Vector stores use specialized indexes to do this search almost instantly.

**What is FAISS?**
FAISS (Facebook AI Similarity Search) is an open-source library that allows developers to quickly search for embeddings that are similar to each other. It is an extremely fast local vector index, perfectly suited for building RAG applications on your own machine without paying for a cloud database.

**How FAISS Similarity Search Works Conceptually**
FAISS organizes the vectors you give it. When you give it a new query vector, it calculates the mathematical distance (or similarity) between the query vector and the stored vectors. The index we use (`IndexFlatIP` combined with vector normalization) perfectly replicates Cosine Similarity mathematically.

**What does Top-K Retrieval mean?**
"Top-K" simply means the top *number* (K) of results you want back. If K=3, the vector store will return the 3 document chunks whose vectors are mathematically closest to the query vector.

**Storing vs Searching vs Retrieving**
1. **Storing**: Taking the 384 numbers from your embedding model and inserting them into the FAISS index. FAISS assigns an integer ID (like 0, 1, 2) to each vector.
2. **Searching**: FAISS compares the query vector to the stored vectors and returns the integer IDs of the closest ones (e.g., IDs 2 and 5).
3. **Retrieving Original Chunks**: FAISS only knows about numbers, not text. So we maintain a Python list that acts as a dictionary. When FAISS says "ID 2 is the best match", we look up index 2 in our list to grab the original text chunk and its metadata to show the user.

**Why preserve metadata alongside vectors?**
When the vector store returns the best match, it only gives us a score. Without metadata, we wouldn't know which PDF file the text came from, what page it was on, or even what the text itself said! Metadata allows us to trace the mathematical result back to the real-world document.
### Stage 4: Vector Store
**What is a Vector Store?**
A vector database (or store) is a specialized database designed to store and quickly search through mathematical vectors (embeddings) rather than rows and columns of text.

**Why do we need a Vector Store?**
Generating embeddings for thousands of pages takes time. We can't generate them every single time a user asks a question. We generate them *once*, store them in the vector store, and then we only have to generate an embedding for the user's short query. Furthermore, comparing a query vector against millions of document vectors one-by-one is incredibly slow. Vector stores use specialized indexes to do this search almost instantly.

**What is FAISS?**
FAISS (Facebook AI Similarity Search) is an open-source library that allows developers to quickly search for embeddings that are similar to each other. It is an extremely fast local vector index, perfectly suited for building RAG applications on your own machine without paying for a cloud database.

**How FAISS Similarity Search Works Conceptually**
FAISS organizes the vectors you give it. When you give it a new query vector, it calculates the mathematical distance (or similarity) between the query vector and the stored vectors. The index we use (`IndexFlatIP` combined with vector normalization) perfectly replicates Cosine Similarity mathematically.

**What does Top-K Retrieval mean?**
"Top-K" simply means the top *number* (K) of results you want back. If K=3, the vector store will return the 3 document chunks whose vectors are mathematically closest to the query vector.

**Storing vs Searching vs Retrieving**
1. **Storing**: Taking the 384 numbers from your embedding model and inserting them into the FAISS index. FAISS assigns an integer ID (like 0, 1, 2) to each vector.
2. **Searching**: FAISS compares the query vector to the stored vectors and returns the integer IDs of the closest ones (e.g., IDs 2 and 5).
3. **Retrieving Original Chunks**: FAISS only knows about numbers, not text. So we maintain a Python list that acts as a dictionary. When FAISS says "ID 2 is the best match", we look up index 2 in our list to grab the original text chunk and its metadata to show the user.

**Why preserve metadata alongside vectors?**
When the vector store returns the best match, it only gives us a score. Without metadata, we wouldn't know which PDF file the text came from, what page it was on, or even what the text itself said! Metadata allows us to trace the mathematical result back to the real-world document.

### Stage 5: RAG Generation
**What makes a system a RAG system?**
A system is considered "RAG" (Retrieval-Augmented Generation) when it actively retrieves outside information from a database (Retrieval) and injects that information into a prompt for an LLM to read (Augmented) before the LLM generates a response (Generation). Without retrieval, it's just a normal chatbot.

**What happens after retrieval?**
After FAISS retrieves the top-K document chunks, we take the text from those chunks and paste them directly into the instructions we send to the LLM (like GPT-3.5 or Gemini).

**What is "context" in RAG?**
The retrieved chunks form the "context". The LLM is given this context and instructed to pretend it is the only information that exists in the world for the purpose of answering the user's question.

**Why should retrieved chunks be included in the LLM prompt?**
LLMs are trained on past data. They don't know your private PDF files, your company's latest data, or recent news. By pasting your specific chunks into the prompt, you temporarily give the LLM the exact knowledge it needs to answer the question accurately right now.

**What is Prompt Grounding?**
Grounding means restricting the LLM to only use the facts provided in the prompt. We explicitly write instructions like: "Answer using ONLY the provided context. Do not invent information."

**Why can an LLM still hallucinate even when using RAG?**
LLMs are fundamentally prediction engines designed to sound confident and conversational. If the retrieved context doesn't actually contain the answer, the LLM might try to "help" by guessing or relying on its general training data, rather than admitting it doesn't know. 

**Why instruct the LLM to answer only from the provided context?**
This minimizes hallucinations. If we don't strictly forbid the LLM from using outside knowledge, it might give a seemingly correct answer that conflicts with the actual contents of your specific documents. We want the assistant to act as a window into *your* documents, not a general trivia bot.

### Stage 6A: Citations and Source-Awareness
**What are RAG Citations?**
Citations are explicit references telling the user exactly which document and page the AI used to formulate its answer.

**Why are citations important?**
Trust. Even with RAG, users should not blindly trust an AI's output. Citations allow the user to easily click a link or open a PDF to verify the AI's claims. If the AI hallucinates, the user can catch it by checking the cited source.

**The Flow of Metadata**
Metadata (like filename and page number) travels a long path:
1. **Extraction**: `document_loader.py` grabs the filename and page number while reading the PDF.
2. **Chunking**: `chunker.py` copies this metadata to every small chunk it creates from that page.
3. **Embeddings & FAISS**: The text is converted to numbers, but we store the metadata safely in a Python dictionary.
4. **Retrieval**: FAISS finds the best math vectors, and we fetch the corresponding metadata from our dictionary.
5. **LLM**: The metadata is formatted into the prompt, AND passed back to the user alongside the final answer.

**Why shouldn't the LLM invent citations?**
LLMs are terrible at tracking exact quotes or page numbers. If you ask an LLM "What page did you get this from?", it will often guess or confidently lie. 

**Retrieved Source Metadata vs LLM-Generated Citations**

### Stage 6B: The Frontend UI (Streamlit)
**Putting it all together**
The backend RAG pipeline is powerful, but a command-line interface isn't great for reading long PDFs and chatting. We built a simple Web UI using **Streamlit**. 

**Clean Separation of Concerns**
Notice that `app/ui.py` does *not* contain the complex logic for chunking or similarity search. It simply acts as a bridge. It takes the PDF the user uploaded, saves it temporarily, and then passes it through our existing modular pipeline:
`load_pdf` -> `chunk_documents` -> `embed_documents` -> `VectorStore` -> `ask()`

**Running the Application**
To start the web interface, open a terminal in the project directory, activate your virtual environment, and run:
```bash
python -m streamlit run app/ui.py
```
This will start a local server and open the RAG assistant in your default web browser!

### Stage 7: RAG Evaluation
**Why do we need evaluation?**
In software engineering, you write unit tests. In AI, you write evaluations. As your document base grows from 10 pages to 10,000 pages, the vector search gets harder. If you change your chunk size from 500 to 1000, how do you know if the system got better or worse?

**Why "the answer looks correct" is not enough**
Eyeballing the LLM's response is dangerous. The LLM might give a correct answer from its own internal memory, even if the retriever failed to find the document! An evaluation system proves that the math (vector search) is actually working.

**What is Retrieval Quality?**
Retrieval quality strictly measures how good FAISS is at finding the right text chunks. It ignores the LLM entirely. If FAISS finds the right page, retrieval is good.

**What is a Golden Dataset?**
A golden dataset is a manually created list of questions and their *known correct sources*. For example: Question: "What is X?", Expected Source: "File Y, Page Z". You run this dataset automatically whenever you update your code.

**What is Recall@K?**
Recall asks: *"Did we find the needle in the haystack?"*
`K` is the number of results we look at. 
If `K=3` (Recall@3), we look at the top 3 chunks returned by FAISS. If the golden source is inside those 3 chunks, Recall = 1 (Success!). If the golden source is not in the top 3, Recall = 0 (Failure).
*Example*: The answer is on Page 10. FAISS returns pages 3, 7, 10. Recall@3 = 1 because 10 is in the list!

**What is Precision@K?**
Precision asks: *"How much garbage did we return with the needle?"*
If FAISS returns 5 chunks, and only 1 chunk has the answer, your precision is 1/5 (20%). In RAG, high Recall is far more important than high Precision, because the LLM is smart enough to ignore the garbage chunks as long as the correct chunk is present.
**Mock Evaluation vs Real Evaluation**
A mock evaluation tests the theoretical capability of the system by feeding it perfectly clean, predefined text snippets. A real evaluation hooks directly into the actual `document_loader.py` and `chunker.py` and attempts to parse *real* PDFs. Real PDFs contain awful formatting, missing spaces, tables, and headers. Evaluating against real documents shows you how your chunking and extraction strategies hold up in the messy real world.

**Ground Truth (Manual Verification)**
Ground Truth is a manually verified set of answers. You MUST NOT use an LLM to generate the "Expected Pages" for your evaluation. If you use the LLM to write the test, and the LLM to take the test, you will get false confidence. An actual human must read the PDF and write: "Question X is answered on Page Y."

**The Process for Creating Real Evaluations:**
1. Open a real PDF you care about.
2. Read it manually.
3. Write 15-20 questions based on different pages.
4. Record the exact source file and page number.
5. Save this as a JSON file and run the evaluation script.
6. When a question FAILS (Recall = 0), read the logs. Did the retriever grab a chunk that *looked* right but was wrong? Did the extraction step garble the text so it couldn't be found? This is how you systematically improve RAG!

### Stage 8: Reranking
**Retrieval vs Reranking**
FAISS (Retrieval) is incredibly fast but less accurate because it uses Bi-Encoders. A Bi-Encoder processes the question and the document completely separately and just measures the angle between them. It's like judging a book by its cover.
Reranking is slow but highly accurate because it uses Cross-Encoders. A Cross-Encoder processes the question and the document *together*, allowing it to see how every word in the question interacts with every word in the document. It's like actually reading the book.

**Why retrieve more candidates first?**
Because Cross-Encoders are so computationally expensive, you cannot run them on all 10,000 pages of your database. Instead, you use the fast FAISS retriever to find the Top 10 best pages, and then you use the slow Cross-Encoder to deeply analyze those 10 pages and pick the true Top 3.

**The Impact**
By adding a lightweight Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), we proved mathematically that our `Recall@1` (getting the perfect answer on the very first try) improved drastically without changing our underlying FAISS index!
