import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Constructs a grounded prompt using retrieved chunks and queries the LLM.
    
    Args:
        question (str): The user's query.
        retrieved_chunks (list[dict]): The most relevant chunks from FAISS.
        
    Returns:
        dict: A dictionary containing the 'answer', the 'sources' used, and the 'prompt_used'.
    """
    
    # 1. Format the context strings
    context_parts = []
    unique_sources = {}
    
    for chunk in retrieved_chunks:
        source_name = chunk.get("source", "Unknown")
        page_num = chunk.get("page_number", "Unknown")
        chunk_id = chunk.get("chunk_id", "Unknown")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        
        # Keep track of unique sources based on source name and page number
        key = (source_name, page_num)
        if key not in unique_sources:
            unique_sources[key] = {
                "source": source_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "score": score
            }
        else:
            # If there are multiple chunks from the same page, keep the one with the highest score
            if score > unique_sources[key]["score"]:
                unique_sources[key]["score"] = score
                unique_sources[key]["chunk_id"] = chunk_id
                
        # Format the chunk clearly for the LLM
        formatted_chunk = f"SOURCE: {source_name}\nPAGE: {page_num}\nTEXT:\n{text}\n"
        context_parts.append(formatted_chunk)
        
    sources = list(unique_sources.values())
        
    # Join all chunks together separated by newlines
    context_string = "\n".join(context_parts)
    
    # 2. Construct the Grounded Prompt
    prompt = f"""
You are a helpful Research Assistant. 
You have been provided with some context extracted from documents.
Your task is to answer the user's question based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
- Answer using ONLY the provided context.
- Do not invent information or hallucinate.
- If the answer cannot be found in the context, say "I cannot answer this question based on the provided documents."
- Do not rely on outside knowledge.
- Keep the answer clear and concise.

CONTEXT:
{context_string}

USER QUESTION:
{question}
"""
    
    # 3. Call the LLM
    try:
        # Point the OpenAI SDK to OpenRouter's API URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct", # Highly capable LLM on OpenRouter
            messages=[
                {"role": "system", "content": "You are a precise and helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0 # Set to 0.0 to make it strictly factual and deterministic
        )
        answer = response.choices[0].message.content
        
    except Exception as e:
        # Handle cases where the API key is missing or the API call fails
        answer = f"[LLM Error or API Key Missing]: {e}"
        
    # 4. Return structured response
    return {
        "answer": answer,
        "sources": sources,
        "prompt_used": prompt
    }
