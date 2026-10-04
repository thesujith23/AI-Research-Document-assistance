import os
import sys
from pypdf import PdfWriter

# Add the parent directory to sys.path so we can import the app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.document_loader import load_pdf

def create_dummy_pdf(file_path: str):
    """Creates a simple, blank PDF for testing purposes."""
    writer = PdfWriter()
    # Add two blank pages
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with open(file_path, "wb") as f:
        writer.write(f)

def run_tests():
    test_pdf_path = os.path.join("data", "documents", "test_sample.pdf")
    os.makedirs(os.path.dirname(test_pdf_path), exist_ok=True)
    
    # 1. Test loading a valid PDF
    print("\n--- Testing Valid PDF ---")
    create_dummy_pdf(test_pdf_path) # Generate a dummy pdf first
    pages = load_pdf(test_pdf_path)
    
    for page in pages:
        print(f"Page {page['page_number']} from {page['source']}:")
        print(f"Extracted Text: {repr(page['text'])} (Length: {len(page['text'])})")
        print("-" * 30)
        
    # 2. Test loading a non-existent file
    print("\n--- Testing Missing File ---")
    load_pdf("data/documents/does_not_exist.pdf")
    
    # 3. Test loading an invalid file (e.g., plain text file disguised as PDF)
    invalid_pdf_path = os.path.join("data", "documents", "invalid.pdf")
    with open(invalid_pdf_path, "w") as f:
        f.write("This is just plain text, not a real PDF binary format.")
        
    print("\n--- Testing Invalid PDF ---")
    load_pdf(invalid_pdf_path)

if __name__ == "__main__":
    run_tests()
