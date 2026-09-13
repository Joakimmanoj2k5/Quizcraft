import pytest
from rag.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Deterministic test vectors (small, no real model needed)
# ---------------------------------------------------------------------------
TEST_VECTORS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

# Person 1 contract – PDF chunk (has page numbers)
PDF_CHUNK_1 = {
    "chunk_id": "notes.pdf_chunk_001",
    "text": "CPU scheduling decisions take place under four conditions.",
    "source": "notes.pdf",
    "start_page": 1,
    "end_page": 2,
}

PDF_CHUNK_2 = {
    "chunk_id": "notes.pdf_chunk_002",
    "text": "Round Robin scheduling uses a time quantum of 10 to 100ms.",
    "source": "notes.pdf",
    "start_page": 3,
    "end_page": 3,
}

# Person 1 contract – TXT chunk (page metadata is None)
TXT_CHUNK = {
    "chunk_id": "revision.txt_chunk_001",
    "text": "Binary search trees have O(log n) average complexity.",
    "source": "revision.txt",
    "start_page": None,
    "end_page": None,
}


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore with an isolated temp directory."""
    return VectorStore(
        collection_name="test_collection",
        persist_directory=str(tmp_path / "chroma"),
    )


# -----------------------------------------------------------------------
# A. Collection creation
# -----------------------------------------------------------------------
def test_collection_creation(store):
    assert store.collection_name == "test_collection"
    assert store.count() == 0


# -----------------------------------------------------------------------
# B. Adding chunks
# -----------------------------------------------------------------------
def test_add_chunks(store):
    store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[0]])
    assert store.count() == 1


# -----------------------------------------------------------------------
# C. Correct collection count
# -----------------------------------------------------------------------
def test_count_after_multiple_adds(store):
    store.add_chunks(
        [PDF_CHUNK_1, PDF_CHUNK_2, TXT_CHUNK],
        TEST_VECTORS,
    )
    assert store.count() == 3


# -----------------------------------------------------------------------
# D. Stored text is preserved
# -----------------------------------------------------------------------
def test_stored_text_preserved(store):
    store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[0]])
    results = store.search(TEST_VECTORS[0], top_k=1)
    assert results[0]["text"] == PDF_CHUNK_1["text"]


# -----------------------------------------------------------------------
# E. chunk_id is preserved
# -----------------------------------------------------------------------
def test_chunk_id_preserved(store):
    store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[0]])
    results = store.search(TEST_VECTORS[0], top_k=1)
    assert results[0]["chunk_id"] == "notes.pdf_chunk_001"


# -----------------------------------------------------------------------
# F. source is preserved
# -----------------------------------------------------------------------
def test_source_preserved(store):
    store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[0]])
    results = store.search(TEST_VECTORS[0], top_k=1)
    assert results[0]["source"] == "notes.pdf"


# -----------------------------------------------------------------------
# G. start_page / end_page are preserved (PDF)
# -----------------------------------------------------------------------
def test_page_metadata_preserved_pdf(store):
    store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[0]])
    results = store.search(TEST_VECTORS[0], top_k=1)
    assert results[0]["start_page"] == 1
    assert results[0]["end_page"] == 2


# -----------------------------------------------------------------------
# H. TXT-style None page metadata is handled correctly
# -----------------------------------------------------------------------
def test_none_page_metadata_roundtrip(store):
    store.add_chunks([TXT_CHUNK], [TEST_VECTORS[2]])
    results = store.search(TEST_VECTORS[2], top_k=1)
    assert results[0]["start_page"] is None
    assert results[0]["end_page"] is None
    assert results[0]["source"] == "revision.txt"


# -----------------------------------------------------------------------
# I. Search returns the expected nearest vector
# -----------------------------------------------------------------------
def test_search_nearest_vector(store):
    store.add_chunks(
        [PDF_CHUNK_1, PDF_CHUNK_2, TXT_CHUNK],
        TEST_VECTORS,
    )
    # Query with vector identical to PDF_CHUNK_2
    results = store.search(TEST_VECTORS[1], top_k=1)
    assert results[0]["chunk_id"] == "notes.pdf_chunk_002"
    assert results[0]["score"] < 1e-5  # distance ~0


# -----------------------------------------------------------------------
# J. top_k works
# -----------------------------------------------------------------------
def test_top_k(store):
    store.add_chunks(
        [PDF_CHUNK_1, PDF_CHUNK_2, TXT_CHUNK],
        TEST_VECTORS,
    )
    results = store.search(TEST_VECTORS[0], top_k=2)
    assert len(results) == 2


# -----------------------------------------------------------------------
# K. Results are ordered correctly (closest first)
# -----------------------------------------------------------------------
def test_results_ordered_by_distance(store):
    store.add_chunks(
        [PDF_CHUNK_1, PDF_CHUNK_2, TXT_CHUNK],
        TEST_VECTORS,
    )
    results = store.search(TEST_VECTORS[0], top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores), f"Results not sorted by distance: {scores}"


# -----------------------------------------------------------------------
# L. Empty collection behavior
# -----------------------------------------------------------------------
def test_search_empty_collection(store):
    results = store.search([1.0, 0.0, 0.0], top_k=5)
    assert results == []


# -----------------------------------------------------------------------
# M. Invalid top_k
# -----------------------------------------------------------------------
def test_invalid_top_k(store):
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        store.search([1.0, 0.0, 0.0], top_k=0)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        store.search([1.0, 0.0, 0.0], top_k=-1)


# -----------------------------------------------------------------------
# N. Mismatched chunks and embeddings
# -----------------------------------------------------------------------
def test_mismatched_chunks_embeddings(store):
    with pytest.raises(ValueError, match="Mismatch"):
        store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[0], TEST_VECTORS[1]])

    with pytest.raises(ValueError, match="Mismatch"):
        store.add_chunks([PDF_CHUNK_1, PDF_CHUNK_2], [TEST_VECTORS[0]])


# -----------------------------------------------------------------------
# O. Duplicate IDs
# -----------------------------------------------------------------------
def test_duplicate_ids(store):
    store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[0]])
    assert store.count() == 1

    # ChromaDB v1.x silently ignores duplicate IDs on add().
    # The count should remain 1 and the original data should be preserved.
    store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[1]])
    assert store.count() == 1

    # Verify the original embedding is still stored (vector 0, not vector 1)
    results = store.search(TEST_VECTORS[0], top_k=1)
    assert results[0]["chunk_id"] == "notes.pdf_chunk_001"
    assert results[0]["score"] < 1e-5  # exact match to original vector


# -----------------------------------------------------------------------
# P. Persistence across VectorStore instances
# -----------------------------------------------------------------------
def test_persistence(tmp_path):
    persist_dir = str(tmp_path / "persist_test")

    # Instance 1: create and add data
    vs1 = VectorStore(collection_name="persist_coll", persist_directory=persist_dir)
    vs1.add_chunks(
        [PDF_CHUNK_1, TXT_CHUNK],
        [TEST_VECTORS[0], TEST_VECTORS[2]],
    )
    assert vs1.count() == 2

    # Force the client reference to be dropped
    del vs1

    # Instance 2: point to the same directory
    vs2 = VectorStore(collection_name="persist_coll", persist_directory=persist_dir)
    assert vs2.count() == 2

    results = vs2.search(TEST_VECTORS[0], top_k=1)
    assert results[0]["chunk_id"] == "notes.pdf_chunk_001"
    assert results[0]["text"] == PDF_CHUNK_1["text"]


# -----------------------------------------------------------------------
# Extra: clear() works
# -----------------------------------------------------------------------
def test_clear(store):
    store.add_chunks([PDF_CHUNK_1, PDF_CHUNK_2], [TEST_VECTORS[0], TEST_VECTORS[1]])
    assert store.count() == 2

    store.clear()
    assert store.count() == 0


# -----------------------------------------------------------------------
# Extra: empty query embedding raises
# -----------------------------------------------------------------------
def test_empty_query_embedding(store):
    with pytest.raises(ValueError, match="query_embedding cannot be empty"):
        store.search([], top_k=1)


# -----------------------------------------------------------------------
# Extra: missing chunk_id raises
# -----------------------------------------------------------------------
def test_missing_chunk_id(store):
    bad_chunk = {"text": "hello", "source": "x.txt", "start_page": None, "end_page": None}
    with pytest.raises(ValueError, match="missing a 'chunk_id'"):
        store.add_chunks([bad_chunk], [TEST_VECTORS[0]])


# -----------------------------------------------------------------------
# Extra: missing text raises
# -----------------------------------------------------------------------
def test_missing_text(store):
    bad_chunk = {"chunk_id": "c1", "source": "x.txt", "start_page": None, "end_page": None}
    with pytest.raises(ValueError, match="missing 'text'"):
        store.add_chunks([bad_chunk], [TEST_VECTORS[0]])


# -----------------------------------------------------------------------
# Extra: top_k larger than collection returns all items
# -----------------------------------------------------------------------
def test_top_k_greater_than_collection_size(store):
    store.add_chunks([PDF_CHUNK_1], [TEST_VECTORS[0]])
    results = store.search(TEST_VECTORS[0], top_k=100)
    assert len(results) == 1
