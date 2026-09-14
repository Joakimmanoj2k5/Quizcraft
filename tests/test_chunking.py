import pytest
from rag.text_cleaner import clean_text
from rag.chunker import chunk_document
from rag.loaders import Document, Page

def test_clean_text():
    # TEST 6: Text is cleaned correctly.
    dirty = "This   has \n\n\n\ntoo much \t\t whitespace.\nAnd a broken \nline."
    cleaned = clean_text(dirty)
    
    assert "This has \n\ntoo much whitespace." in cleaned
    assert "broken line." in cleaned
    
    # Should not destroy meaning
    assert clean_text("1. Heading") == "1. Heading"

def test_long_document_multiple_chunks():
    # TEST 7: Long document produces multiple chunks.
    text = "Word. " * 2000  # A long string
    doc = Document(source="long.txt", file_type="txt", pages=[Page(page_number=None, text=text)])
    
    chunks = chunk_document(doc, chunk_size=1000, overlap=100)
    
    assert len(chunks) > 1
    
    # TEST 9: Every chunk contains source metadata.
    for i, chunk in enumerate(chunks):
        assert chunk["source"] == "long.txt"
        assert chunk["start_page"] is None
        assert chunk["end_page"] is None
        assert "chunk_id" in chunk
        
        # TEST 11: No chunk is unexpectedly empty.
        assert len(chunk["text"]) > 0

def test_chunks_overlap():
    # TEST 8: Chunks overlap verification.
    text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
    doc = Document(source="alphabet.txt", file_type="txt", pages=[Page(page_number=None, text=text)])
    
    # Very small chunk size to force overlap logic
    chunks = chunk_document(doc, chunk_size=20, overlap=10)
    
    assert len(chunks) > 1
    
    # Check that some text from chunk 0 is in chunk 1
    chunk_0 = chunks[0]["text"]
    chunk_1 = chunks[1]["text"]
    
    # They should share some words/letters due to overlap
    words_0 = set(chunk_0.split())
    words_1 = set(chunk_1.split())
    
    intersection = words_0.intersection(words_1)
    assert len(intersection) > 0, f"No overlap found between consecutive chunks:\nChunk 1: {chunk_0}\nChunk 2: {chunk_1}"

def test_cross_page_chunking():
    # TEST: Cross-Page Testing
    page1 = Page(page_number=1, text="Artificial intelligence is a field that focuses on")
    page2 = Page(page_number=2, text="building systems capable of performing tasks that normally require human intelligence.")
    
    doc = Document(source="ai.pdf", file_type="pdf", pages=[page1, page2])
    
    # Large enough chunk to fit both pages
    chunks = chunk_document(doc, chunk_size=4000, overlap=600)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    
    # Verify both pages' content exists in one chunk
    assert "Artificial intelligence is a field" in chunk["text"]
    assert "building systems capable" in chunk["text"]
    
    # Verify metadata ranges span correctly
    assert chunk["start_page"] == 1
    assert chunk["end_page"] == 2
    assert chunk["source"] == "ai.pdf"

def test_changing_chunk_size_changes_behavior():
    # TEST 12: Changing chunk_size/overlap changes chunking behavior.
    text = "This is a simple text that we will chunk differently to see if the lengths change." * 50
    doc = Document(source="test.txt", file_type="txt", pages=[Page(page_number=None, text=text)])
    
    chunks_small = chunk_document(doc, chunk_size=200, overlap=50)
    chunks_large = chunk_document(doc, chunk_size=500, overlap=50)
    
    assert len(chunks_small) > len(chunks_large)

def test_chunking_validations():
    # Validation tests for sizes
    doc = Document(source="test.txt", file_type="txt", pages=[Page(page_number=None, text="test")])
    
    with pytest.raises(ValueError, match="chunk_size must be strictly greater than 0"):
        chunk_document(doc, chunk_size=0, overlap=0)
        
    with pytest.raises(ValueError, match="overlap must be greater than or equal to 0"):
        chunk_document(doc, chunk_size=100, overlap=-1)
        
    with pytest.raises(ValueError, match="overlap must be strictly less than chunk_size"):
        chunk_document(doc, chunk_size=100, overlap=100)
