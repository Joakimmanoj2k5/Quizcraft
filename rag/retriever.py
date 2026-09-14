from typing import List, Dict, Any, Optional
from .embedder import Embedder
from .vector_store import VectorStore


class DocumentRetriever:
    """
    Coordinates semantic indexing and retrieval by combining the Embedder
    and VectorStore components.

    The SAME Embedder instance is used for both document indexing and query
    embedding, ensuring vector-space consistency.

    Score convention:
        Results include a 'score' field representing L2 (Euclidean) distance,
        inherited from the VectorStore. LOWER values = MORE relevant.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        """
        Initializes the retriever with explicit Embedder and VectorStore
        instances. Both must be provided — no hidden defaults.

        Args:
            embedder:     The Embedder instance for generating dense vectors.
            vector_store: The VectorStore instance for persistent storage/search.

        Raises:
            TypeError: If embedder or vector_store are not the expected types.
        """
        if not isinstance(embedder, Embedder):
            raise TypeError("embedder must be an instance of Embedder.")
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must be an instance of VectorStore.")

        self.embedder = embedder
        self.vector_store = vector_store

    def index_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds chunk text using the Embedder and stores the chunks with their
        embeddings in the VectorStore.

        Args:
            chunks: A list of chunk dictionaries following the Person 1 contract:
                    {chunk_id, text, source, start_page, end_page}.

        Returns:
            The number of chunks successfully indexed.

        Raises:
            TypeError:  If chunks is not a list.
            ValueError: If chunks is empty.
        """
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list of dictionaries.")

        if not chunks:
            raise ValueError("chunks list is empty. Nothing to index.")

        # Extract text for embedding
        texts = [chunk.get("text", "") for chunk in chunks]

        # Generate embeddings using the SAME model that will be used for queries
        embeddings = self.embedder.embed_texts(texts)

        # Store in the vector database
        self.vector_store.add_chunks(chunks, embeddings)

        return len(chunks)

    def get_relevant_context(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Embeds the query using the SAME Embedder used for indexing, then
        searches the VectorStore for the most relevant chunks.

        Args:
            query:  The semantic search query string.
            top_k:  Maximum number of results to return. Must be >= 1.

        Returns:
            A list of chunk dictionaries ordered by relevance (closest first).
            Each dictionary contains:
                chunk_id, text, source, start_page, end_page, score.
            The 'score' is L2 distance — LOWER = MORE RELEVANT.

            If the collection contains fewer chunks than top_k, only the
            available chunks are returned (no padding or duplication).

        Raises:
            ValueError: If query is empty/whitespace or top_k < 1.
            TypeError:  If query is not a string.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")

        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        # Check if the collection has any data
        if self.vector_store.count() == 0:
            return []

        # Embed the query with the SAME model used during indexing
        query_embedding = self.embedder.embed_query(query)

        # Search the vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results
