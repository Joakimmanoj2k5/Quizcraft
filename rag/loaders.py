import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import pypdf

@dataclass
class Page:
    page_number: Optional[int]
    text: str

@dataclass
class Document:
    source: str
    file_type: str
    pages: List[Page]

def load_document(file_path: str | Path) -> Document:
    """
    Loads a document (PDF or TXT) and returns a Document object containing pages.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    if path.stat().st_size == 0:
        raise ValueError("The uploaded file is empty.")
        
    ext = path.suffix.lower()
    source_name = path.name
    
    if ext == ".txt":
        return _load_txt(path, source_name)
    elif ext == ".pdf":
        return _load_pdf(path, source_name)
    else:
        raise ValueError("Unsupported file type. Please upload PDF or TXT.")

def _load_txt(path: Path, source_name: str) -> Document:
    """Helper to load a txt file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fallback to other encodings if needed, but utf-8 is standard
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()
            
    if not text.strip():
        raise ValueError("The uploaded file is empty.")
        
    return Document(
        source=source_name,
        file_type="txt",
        pages=[Page(page_number=None, text=text)]
    )

def _load_pdf(path: Path, source_name: str) -> Document:
    """Helper to load a pdf file."""
    try:
        reader = pypdf.PdfReader(path)
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {e}")
        
    if not reader.pages:
        raise ValueError("No readable text could be extracted from this PDF.")
        
    pages = []
    has_text = False
    
    for i, pdf_page in enumerate(reader.pages):
        text = pdf_page.extract_text()
        if text:
            # page_number is 1-indexed
            pages.append(Page(page_number=i + 1, text=text))
            has_text = True
            
    if not has_text:
        raise ValueError("No readable text could be extracted from this PDF.")
        
    return Document(
        source=source_name,
        file_type="pdf",
        pages=pages
    )
