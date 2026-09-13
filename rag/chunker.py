from typing import List, Dict, Any
from .loaders import Document
from .text_cleaner import clean_text

def recursive_split(text: str, chunk_size: int, overlap: int, separators: List[str]) -> List[str]:
    """Recursively split text into overlapping chunks."""
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    separator = separators[0] if separators else ""

    if separator:
        if separator == ". ":
            splits = [s + "." for s in text.split(separator) if s]
            if splits and not text.endswith(". "):
                splits[-1] = splits[-1].rstrip(".")
            join_str = " "
        else:
            splits = text.split(separator)
            join_str = separator

        chunks = []
        current_chunk = []
        current_len = 0

        for s in splits:
            if not s:
                continue

            s_len = len(s)

            # Split oversized sections recursively
            if s_len > chunk_size:
                if current_chunk:
                    chunks.append(join_str.join(current_chunk))
                    current_chunk = []
                    current_len = 0

                if len(separators) > 1:
                    chunks.extend(
                        recursive_split(s, chunk_size, overlap, separators[1:])
                    )
                else:
                    step = chunk_size - overlap
                    for i in range(0, len(s), step):
                        chunks.append(s[i:i + chunk_size])
                continue

            # Current chunk full → finalize it
            extra = s_len + (len(join_str) if current_chunk else 0)

            if current_len + extra > chunk_size:
                completed = join_str.join(current_chunk)
                chunks.append(completed)

                # Guaranteed character overlap
                overlap_text = completed[-overlap:] if overlap > 0 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current_chunk.append(s)
            current_len += s_len + (len(join_str) if current_len > 0 else 0)

        if current_chunk:
            chunks.append(join_str.join(current_chunk))

        return chunks

    # Final fallback: character slicing
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])

    return chunks

def chunk_document(document: Document, chunk_size: int = 4000, overlap: int = 600) -> List[Dict[str, Any]]:
    """
    Splits a document into overlapping chunks while preserving metadata.
    Operates at the document level to allow chunks to span across page boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be strictly greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")
        
    full_text = ""
    page_mapping = []
    
    # 1. Combine pages into a logically continuous document and maintain boundary mappings
    for page in document.pages:
        cleaned_page = clean_text(page.text)
        if not cleaned_page:
            continue
            
        # Add space between pages to avoid fusing words across boundaries
        if full_text and not full_text[-1].isspace() and not cleaned_page[0].isspace():
            full_text += " "
            
        start_idx = len(full_text)
        full_text += cleaned_page
        end_idx = len(full_text)
        
        page_mapping.append({
            "start": start_idx,
            "end": end_idx,
            "page_number": page.page_number
        })
        
    if not full_text:
        return []
        
    # 2. Perform document-level semantic/paragraph-aware chunking
    separators = ["\n\n", "\n", ". ", " "]
    text_chunks = recursive_split(full_text, chunk_size, overlap, separators)
    
    # 3. Attach page metadata back to every chunk
    all_chunks = []
    search_start = 0
    chunk_counter = 1
    
    for text_chunk in text_chunks:
        if not text_chunk.strip():
            continue
            
        # Find exact position in full_text to determine page boundaries
        chunk_idx = full_text.find(text_chunk, search_start)
        if chunk_idx == -1:
            chunk_idx = full_text.find(text_chunk)
            if chunk_idx == -1:
                chunk_idx = search_start
                
        chunk_end = chunk_idx + len(text_chunk)
        search_start = chunk_idx + 1
        
        start_page = None
        end_page = None
        
        for mapping in page_mapping:
            if mapping["end"] > chunk_idx and mapping["start"] < chunk_end:
                if start_page is None:
                    start_page = mapping["page_number"]
                end_page = mapping["page_number"]
                
        chunk_dict = {
            "chunk_id": f"{document.source}_chunk_{chunk_counter:03d}",
            "text": text_chunk.strip(),
            "source": document.source,
            "start_page": start_page,
            "end_page": end_page
        }
        all_chunks.append(chunk_dict)
        chunk_counter += 1
        
    return all_chunks
