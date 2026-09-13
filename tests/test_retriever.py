import pytest
from unittest.mock import MagicMock, patch
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import DocumentRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder():
    """Creates a mock Embedder that returns deterministic vectors."""
    embedder = MagicMock(spec=Embedder)
    # embed_texts returns one 3-dim vector per input text
    embedder.embed_texts.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    # embed_query returns a vector close to the first chunk
    embedder.embed_query.return_value = [0.9, 0.1, 0.0]
    return embedder


@pytest.fixture
def mock_vector_store(tmp_path):
    """Creates a real (but temporary) VectorStore."""
    return VectorStore(
        collection_name="test_retriever",
        persist_directory=str(tmp_path / "chroma_retriever"),
    )


@pytest.fixture
def sample_chunks():
    """Three chunks following the exact Person 1 contract."""
    return [
        {
            "chunk_id": "doc_chunk_001",
            "text": "CPU scheduling algorithms determine process execution order.",
            "source": "os.pdf",
            "start_page": 1,
            "end_page": 1,
        },
        {
            "chunk_id": "doc_chunk_002",
            "text": "Database indexing uses B-trees for efficient retrieval.",
            "source": "db.pdf",
            "start_page": 5,
            "end_page": 6,
        },
        {
            "chunk_id": "doc_chunk_003",
            "text": "Binary search trees have logarithmic complexity.",
            "source": "dsa.txt",
            "start_page": None,
            "end_page": None,
        },
    ]


@pytest.fixture
def retriever(mock_embedder, mock_vector_store):
    """Retriever wired with mock embedder and real (temp) vector store."""
    return DocumentRetriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )


# -----------------------------------------------------------------------
# A. Retriever initialization
# -----------------------------------------------------------------------
def test_initialization(mock_embedder, mock_vector_store):
    r = DocumentRetriever(embedder=mock_embedder, vector_store=mock_vector_store)
    assert r.embedder is mock_embedder
    assert r.vector_store is mock_vector_store


def test_initialization_rejects_wrong_types():
    with pytest.raises(TypeError, match="embedder must be an instance"):
        DocumentRetriever(embedder="not_an_embedder", vector_store=MagicMock(spec=VectorStore))

    with pytest.raises(TypeError, match="vector_store must be an instance"):
        DocumentRetriever(embedder=MagicMock(spec=Embedder), vector_store="not_a_store")


# -----------------------------------------------------------------------
# B. index_documents()
# -----------------------------------------------------------------------
def test_index_documents(retriever, sample_chunks):
    count = retriever.index_documents(sample_chunks)
    assert count == 3
    assert retriever.vector_store.count() == 3


def test_index_documents_rejects_non_list(retriever):
    with pytest.raises(TypeError, match="chunks must be a list"):
        retriever.index_documents({"not": "a list"})


def test_index_documents_rejects_empty(retriever):
    with pytest.raises(ValueError, match="chunks list is empty"):
        retriever.index_documents([])


# -----------------------------------------------------------------------
# C. Query embedding is performed
# -----------------------------------------------------------------------
def test_query_embedding_called(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)
    retriever.get_relevant_context("test query", top_k=1)
    retriever.embedder.embed_query.assert_called_once_with("test query")


# -----------------------------------------------------------------------
# D. VectorStore search is called
# -----------------------------------------------------------------------
def test_vectorstore_search_called(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)
    results = retriever.get_relevant_context("test query", top_k=2)
    # The mock embedder returns [0.9, 0.1, 0.0] for the query,
    # which is closest to [1.0, 0.0, 0.0] (chunk_001).
    assert len(results) > 0


# -----------------------------------------------------------------------
# E. Returned metadata is preserved
# -----------------------------------------------------------------------
def test_returned_metadata_preserved(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)
    results = retriever.get_relevant_context("test query", top_k=3)

    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "source" in r
        assert "start_page" in r
        assert "end_page" in r
        assert "score" in r


def test_none_pages_preserved_in_results(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)
    results = retriever.get_relevant_context("test query", top_k=3)

    # Find the TXT chunk
    txt_results = [r for r in results if r["source"] == "dsa.txt"]
    assert len(txt_results) == 1
    assert txt_results[0]["start_page"] is None
    assert txt_results[0]["end_page"] is None


# -----------------------------------------------------------------------
# F. top_k behavior
# -----------------------------------------------------------------------
def test_top_k_limits_results(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)
    results = retriever.get_relevant_context("test query", top_k=1)
    assert len(results) == 1


def test_top_k_larger_than_collection(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)
    results = retriever.get_relevant_context("test query", top_k=100)
    assert len(results) == 3  # only 3 chunks exist, no padding


# -----------------------------------------------------------------------
# G. Empty query handling
# -----------------------------------------------------------------------
def test_empty_query_raises(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.get_relevant_context("")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.get_relevant_context("   ")


def test_non_string_query_raises(retriever, sample_chunks):
    with pytest.raises(TypeError, match="query must be a string"):
        retriever.get_relevant_context(123)


# -----------------------------------------------------------------------
# H. Invalid top_k handling
# -----------------------------------------------------------------------
def test_invalid_top_k_raises(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        retriever.get_relevant_context("test", top_k=0)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        retriever.get_relevant_context("test", top_k=-5)


# -----------------------------------------------------------------------
# I. Empty collection handling
# -----------------------------------------------------------------------
def test_empty_collection_returns_empty(retriever):
    results = retriever.get_relevant_context("test query", top_k=5)
    assert results == []


# -----------------------------------------------------------------------
# J. Result ordering (closest first)
# -----------------------------------------------------------------------
def test_results_ordered_by_distance(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)
    results = retriever.get_relevant_context("test query", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores), f"Results not sorted by distance: {scores}"


# -----------------------------------------------------------------------
# K. Same Embedder is used for indexing and querying
# -----------------------------------------------------------------------
def test_same_embedder_used(retriever, sample_chunks):
    retriever.index_documents(sample_chunks)
    retriever.get_relevant_context("test query", top_k=1)

    # Both embed_texts (indexing) and embed_query (retrieval) must
    # have been called on the SAME mock object
    retriever.embedder.embed_texts.assert_called_once()
    retriever.embedder.embed_query.assert_called_once()


# =======================================================================
# SEMANTIC INTEGRATION TESTS (real model, small dataset)
# =======================================================================

@pytest.fixture(scope="module")
def real_embedder():
    """Load the real model ONCE for the entire module."""
    return Embedder("all-MiniLM-L6-v2")


SEMANTIC_CHUNKS = [
    {
        "chunk_id": "scheduling_001",
        "text": "Round Robin CPU scheduling assigns each process a fixed time quantum and cycles through the ready queue.",
        "source": "os_notes.pdf",
        "start_page": 3,
        "end_page": 3,
    },
    {
        "chunk_id": "database_001",
        "text": "Database indexing uses structures such as B-trees to make data retrieval more efficient.",
        "source": "db_notes.pdf",
        "start_page": 10,
        "end_page": 11,
    },
    {
        "chunk_id": "networking_001",
        "text": "TCP provides reliable, ordered delivery of data between network endpoints.",
        "source": "networking.txt",
        "start_page": None,
        "end_page": None,
    },
]


def test_semantic_scheduling_query(real_embedder, tmp_path):
    vs = VectorStore(collection_name="sem_sched", persist_directory=str(tmp_path / "sem1"))
    retriever = DocumentRetriever(embedder=real_embedder, vector_store=vs)
    retriever.index_documents(SEMANTIC_CHUNKS)

    results = retriever.get_relevant_context(
        "What happens when a process receives a time quantum in Round Robin scheduling?",
        top_k=3,
    )
    assert results[0]["chunk_id"] == "scheduling_001"
    assert results[0]["source"] == "os_notes.pdf"


def test_semantic_database_query(real_embedder, tmp_path):
    vs = VectorStore(collection_name="sem_db", persist_directory=str(tmp_path / "sem2"))
    retriever = DocumentRetriever(embedder=real_embedder, vector_store=vs)
    retriever.index_documents(SEMANTIC_CHUNKS)

    results = retriever.get_relevant_context(
        "How do database indexes improve data retrieval?",
        top_k=3,
    )
    assert results[0]["chunk_id"] == "database_001"
    assert results[0]["source"] == "db_notes.pdf"


def test_semantic_networking_query(real_embedder, tmp_path):
    vs = VectorStore(collection_name="sem_net", persist_directory=str(tmp_path / "sem3"))
    retriever = DocumentRetriever(embedder=real_embedder, vector_store=vs)
    retriever.index_documents(SEMANTIC_CHUNKS)

    results = retriever.get_relevant_context(
        "How does TCP provide reliable communication?",
        top_k=3,
    )
    assert results[0]["chunk_id"] == "networking_001"
    assert results[0]["start_page"] is None
    assert results[0]["end_page"] is None
