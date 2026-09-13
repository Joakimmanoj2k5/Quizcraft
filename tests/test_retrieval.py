import pytest
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import DocumentRetriever

def test_embedder():
    # TEST 1: Embedding generation
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    texts = ["This is a test sentence.", "This is another test sentence."]
    
    embeddings = embedder.embed_texts(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384  # Dimension of all-MiniLM-L6-v2
    
    query_embedding = embedder.embed_query("test query")
    assert len(query_embedding) == 384

def test_vector_store():
    # TEST 2: Vector store insertion and retrieval
    vs = VectorStore(collection_name="test_collection") # Ephemeral by default
    
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "The capital of France is Paris.",
            "source": "geography.txt",
            "start_page": None,
            "end_page": None
        },
        {
            "chunk_id": "chunk_2",
            "text": "The capital of Japan is Tokyo.",
            "source": "geography.txt",
            "start_page": None,
            "end_page": None
        }
    ]
    
    # Mock embeddings (dimension doesn't matter for pure ChromaDB test as long as consistent)
    embeddings = [[0.1, 0.2, 0.3], [0.8, 0.9, 1.0]]
    
    vs.add_chunks(chunks, embeddings)
    
    # Search for something closer to chunk_1
    query_embedding = [0.1, 0.2, 0.4] 
    results = vs.search(query_embedding, top_k=1)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_1"
    assert results[0]["text"] == "The capital of France is Paris."
    assert results[0]["source"] == "geography.txt"

def test_document_retriever():
    # TEST 3: High-level retriever integration
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = DocumentRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )
    
    chunks = [
        {
            "chunk_id": "pdf_chunk_1",
            "text": "Artificial intelligence is a fascinating field.",
            "source": "ai.pdf",
            "start_page": 1,
            "end_page": 1
        },
        {
            "chunk_id": "pdf_chunk_2",
            "text": "Photosynthesis is how plants make food.",
            "source": "biology.pdf",
            "start_page": 5,
            "end_page": 6
        }
    ]
    
    retriever.index_documents(chunks)
    
    # Query related to biology
    results = retriever.get_relevant_context("How do plants eat?", top_k=1)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "pdf_chunk_2"
    assert "Photosynthesis" in results[0]["text"]
    assert results[0]["start_page"] == 5
    assert results[0]["end_page"] == 6
