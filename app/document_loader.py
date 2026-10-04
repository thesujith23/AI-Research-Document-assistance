import os
from pypdf import PdfReader
from pypdf.errors import PdfReadError

def load_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF file and extracts text page by page.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a page.
                    It contains the extracted 'text', 'page_number' (1-indexed), 
                    and 'source' (the file name).
    """
    # 1. Handle missing files gracefully
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return []

    # Extract the file name to use as metadata
    file_name = os.path.basename(file_path)
    extracted_pages = []

    try:
        # 2. Open the PDF file (PdfReader handles the file reading)
        reader = PdfReader(file_path)
        
        # 3. Iterate through every page in the PDF
        for page_num, page in enumerate(reader.pages):
            
            # Extract text from the current page
            text = page.extract_text()
            
            # 4. Handle pages with no extractable text
            # Some PDF pages are just images or empty; extract_text() might return None or whitespace
            if text:
                text = text.strip()
            else:
                text = "" # Ensure it's always a string, even if empty
                
            # 5. Create a structured dictionary for the page (Text + Metadata)
            page_data = {
                "text": text,
                "page_number": page_num + 1,  # Adding 1 because enumerate is 0-indexed
                "source": file_name
            }
            extracted_pages.append(page_data)
            
    except PdfReadError as e:
        # Handle invalid or corrupted PDF files specifically
        print(f"Error: '{file_path}' is an invalid or corrupted PDF file. Details: {e}")
        return []
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred while reading '{file_path}': {e}")
        return []

    # Return the structured data
    return extracted_pages
