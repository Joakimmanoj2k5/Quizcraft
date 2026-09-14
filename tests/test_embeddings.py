import pytest
from rag.embedder import Embedder

def test_embedder_initialization_and_properties():
    # Load model
    embedder = Embedder("all-MiniLM-L6-v2")
    
    # Requirements specify we must report/check dimension and max length
    assert embedder.embedding_dimension == 384, f"Expected 384, got {embedder.embedding_dimension}"
    assert embedder.max_seq_length == 256, f"Expected max seq len 256, got {embedder.max_seq_length}"

def test_embed_texts():
    embedder = Embedder("all-MiniLM-L6-v2")
    texts = ["This is a test document.", "Another short sentence."]
    
    embeddings = embedder.embed_texts(texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert isinstance(embeddings[0][0], float)

def test_embed_empty_texts():
    embedder = Embedder("all-MiniLM-L6-v2")
    assert embedder.embed_texts([]) == []

def test_embed_texts_type_error():
    embedder = Embedder("all-MiniLM-L6-v2")
    with pytest.raises(TypeError, match="Expected a list of strings"):
        embedder.embed_texts("Not a list") # type: ignore

def test_embed_query():
    embedder = Embedder("all-MiniLM-L6-v2")
    query = "What is artificial intelligence?"
    
    embedding = embedder.embed_query(query)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert isinstance(embedding[0], float)

def test_embed_query_validation_errors():
    embedder = Embedder("all-MiniLM-L6-v2")
    
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedder.embed_query("   ")
        
    with pytest.raises(TypeError, match="Query must be a string"):
        embedder.embed_query(["Not a string"]) # type: ignore
