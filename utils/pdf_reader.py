import fitz  # PyMuPDF
import os

def extract_text(file_path: str) -> str:
    """Extract full text for chunking."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def extract_pages(file_path: str) -> list:
    """
    Extract text page by page.
    Returns list of dicts with page number and text.
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages = []

    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({
                    "page": page_num + 1,
                    "text": text
                })
        doc.close()

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        pages.append({"page": 1, "text": text})

    return pages