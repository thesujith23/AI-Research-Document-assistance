import sys
import os
import tempfile
import streamlit as st

# Add parent directory to path so python can find the 'app' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.document_loader import load_pdf
from app.chunker import chunk_documents
from app.embeddings import embed_documents
from app.vector_store import VectorStore
from app.rag_pipeline import ask

# Configure the Streamlit page
st.set_page_config(page_title="RAG Research Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Research & Document Assistant")

# --- Session State Management ---
# We use session state so the vector store persists while the user interacts with the UI
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []

# --- Sidebar: Document Upload & Processing ---
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if not uploaded_files:
            st.error("Please upload at least one PDF document first.")
        else:
            # Show a progress spinner while processing
            with st.spinner("Processing documents... This may take a moment."):
                try:
                    all_chunks = []
                    indexed_names = []
                    
                    # Create a temporary directory to save the uploaded files
                    # This allows our existing load_pdf function to work seamlessly
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            
                            # --- BACKEND RAG PIPELINE CALLS ---
                            
                            # Stage 1: Document Ingestion
                            pages = load_pdf(file_path)
                            
                            # Stage 2: Text Chunking
                            chunks = chunk_documents(pages)
                            all_chunks.extend(chunks)
                            
                            indexed_names.append(uploaded_file.name)
                            
                    if not all_chunks:
                        st.error("No extractable text found in the provided documents.")
                    else:
                        # Stage 3: Embeddings
                        embedded_chunks = embed_documents(all_chunks)
                        
                        # Stage 4: Vector Store
                        dimension = len(embedded_chunks[0]["embedding"])
                        store = VectorStore(dimension=dimension)
                        store.add_chunks(embedded_chunks)
                        
                        # Save the fully loaded store into the session state
                        st.session_state.vector_store = store
                        st.session_state.indexed_documents = indexed_names
                        
                        st.success(f"Successfully processed {len(indexed_names)} document(s)!")
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

    st.divider()
    
    st.header("Indexed Documents")
    if st.session_state.indexed_documents:
        for doc in st.session_state.indexed_documents:
            st.markdown(f"- `{doc}`")
    else:
        st.write("No documents currently indexed.")

# --- Main Area: Query & Answer ---
st.header("2. Ask Questions")

# Check for API key (Required for LLM generation)
if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ API Key is missing! The LLM will not be able to generate answers until this is set in the `.env` file.")

question = st.text_input("Ask a question about your documents:")

if st.button("Submit Question"):
    if not st.session_state.vector_store:
        st.error("Please upload and process documents first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Stage 5 & 6: Run the RAG Pipeline (Retrieval + LLM + Citations)
                result = ask(question, st.session_state.vector_store, top_k=3)
                
                # Display the Answer
                st.subheader("Answer")
                st.write(result["answer"])
                
                st.divider()
                
                # Display the Citations
                st.subheader("Sources (Citations)")
                if result.get("sources"):
                    for idx, source in enumerate(result["sources"]):
                        score_str = f" (Similarity Score: {source['score']:.4f})" if 'score' in source else ""
                        st.markdown(f"**{idx+1}. {source['source']}** - Page {source['page']}{score_str}")
                else:
                    st.write("No relevant sources found in the documents.")
                    
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
