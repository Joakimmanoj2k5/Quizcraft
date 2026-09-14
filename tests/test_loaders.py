import pytest
from pathlib import Path
from rag.loaders import load_document

def test_load_txt(tmp_path):
    # TEST 1: Valid TXT file -> text extracted.
    file_path = tmp_path / "valid.txt"
    file_path.write_text("Hello TXT World. This is a test.")
    
    doc = load_document(file_path)
    
    assert doc.file_type == "txt"
    assert doc.source == "valid.txt"
    assert len(doc.pages) == 1
    assert "Hello TXT World" in doc.pages[0].text
    assert doc.pages[0].page_number is None

def test_load_pdf():
    # TEST 2: Valid PDF -> text extracted.
    pdf_path = Path("tests/dummy.pdf")
    if not pdf_path.exists():
        pytest.skip("dummy.pdf not found. Run PDF generation first.")
        
    doc = load_document(pdf_path)
    
    assert doc.file_type == "pdf"
    assert doc.source == "dummy.pdf"
    assert len(doc.pages) >= 1
    assert "Hello PDF World" in doc.pages[0].text
    assert doc.pages[0].page_number == 1

def test_load_empty_txt(tmp_path):
    # TEST 3: Empty TXT -> meaningful error.
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    
    with pytest.raises(ValueError, match="The uploaded file is empty."):
        load_document(file_path)

def test_load_empty_pdf(tmp_path):
    # TEST 4: Invalid/corrupt PDF -> graceful error.
    file_path = tmp_path / "empty.pdf"
    file_path.write_text("")
    
    with pytest.raises(ValueError, match="The uploaded file is empty."):
        load_document(file_path)

def test_load_corrupt_pdf(tmp_path):
    # TEST 4 (continued): Corrupt PDF
    file_path = tmp_path / "corrupt.pdf"
    file_path.write_text("This is not a real PDF content")
    
    with pytest.raises(ValueError, match="Failed to read PDF file"):
        load_document(file_path)

def test_unsupported_extension(tmp_path):
    # TEST 5: Unsupported extension -> graceful error.
    file_path = tmp_path / "file.docx"
    file_path.write_text("dummy content")
    
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)
