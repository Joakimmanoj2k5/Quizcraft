import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb

# Sentinel value used to represent None in ChromaDB metadata.
# ChromaDB does not allow None values in metadata fields.
# We use -1 as a sentinel for missing page numbers (e.g., TXT files),
# and convert it back to None when returning results.
_PAGE_NONE_SENTINEL = -1


class VectorStore:
    """
    Manages persistent storage and retrieval of vector embeddings and their
    associated metadata using ChromaDB.

    This class is responsible ONLY for storage and retrieval. It does NOT
    perform embedding — that responsibility belongs to the Embedder class.

    Metadata handling:
        ChromaDB does not support None values in metadata. For TXT-sourced
        chunks where start_page and end_page are None, we store the sentinel
        value -1 internally and convert it back to None when returning results.

    Score semantics:
        The 'score' field in search results represents L2 (Euclidean) distance.
        Lower values indicate higher similarity (closer vectors).
    """

    def __init__(
        self,
        collection_name: str = "quizcraft",
        persist_directory: str = "data/chroma",
    ):
        """
        Initializes the ChromaDB persistent client and retrieves or creates
        the specified collection.

        Args:
            collection_name: The name of the ChromaDB collection.
            persist_directory: The local directory for persistent storage.
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name cannot be empty.")

        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(
        self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """
        Adds text chunks and their corresponding embeddings to the vector store.

        Args:
            chunks: A list of chunk dictionaries following the Person 1 contract:
                    {chunk_id, text, source, start_page, end_page}.
            embeddings: The corresponding dense embedding vectors.

        Raises:
            ValueError: If chunks/embeddings are mismatched, empty, or missing
                        required fields (chunk_id, text).
        """
        if not chunks and not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks and {len(embeddings)} embeddings provided."
            )

        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        valid_embeddings: List[List[float]] = []

        for i, chunk in enumerate(chunks):
            # Validate chunk_id
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Chunk at index {i} is missing a 'chunk_id'.")

            # Validate text
            text = chunk.get("text")
            if not text:
                raise ValueError(
                    f"Chunk at index {i} ('{chunk_id}') is missing 'text'."
                )

            # Validate embedding
            if not embeddings[i]:
                raise ValueError(
                    f"Embedding at index {i} ('{chunk_id}') is empty."
                )

            # Build metadata, converting None page values to sentinel
            start_page = chunk.get("start_page")
            end_page = chunk.get("end_page")

            meta = {
                "source": chunk.get("source", "unknown"),
                "start_page": start_page if start_page is not None else _PAGE_NONE_SENTINEL,
                "end_page": end_page if end_page is not None else _PAGE_NONE_SENTINEL,
            }

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(meta)
            valid_embeddings.append(embeddings[i])

        self.collection.add(
            ids=ids,
            embeddings=valid_embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches the vector store for the most similar chunks.

        Args:
            query_embedding: The embedding vector of the search query.
            top_k: Number of results to return. Must be >= 1.

        Returns:
            A list of dictionaries ordered by relevance (closest first).
            Each dictionary contains:
                chunk_id:   The original Person 1 chunk ID.
                text:       The stored document text.
                source:     The source filename.
                start_page: The starting page (int or None for TXT).
                end_page:   The ending page (int or None for TXT).
                score:      The L2 distance (lower = more similar).

        Raises:
            ValueError: If top_k < 1 or query_embedding is empty.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        # If the collection is empty, return an empty list
        collection_count = self.collection.count()
        if collection_count == 0:
            return []

        # ChromaDB raises an error if top_k > collection size, so we clamp it
        effective_top_k = min(top_k, collection_count)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k,
        )

        retrieved_chunks: List[Dict[str, Any]] = []

        if not results.get("ids") or not results["ids"][0]:
            return retrieved_chunks

        r_ids = results["ids"][0]
        r_docs = results["documents"][0] if results.get("documents") else []
        r_metas = results["metadatas"][0] if results.get("metadatas") else []
        r_distances = results["distances"][0] if results.get("distances") else []

        for i in range(len(r_ids)):
            meta = r_metas[i] if i < len(r_metas) and r_metas[i] else {}

            # Convert sentinel values back to None
            start_page = meta.get("start_page")
            end_page = meta.get("end_page")
            if start_page == _PAGE_NONE_SENTINEL:
                start_page = None
            if end_page == _PAGE_NONE_SENTINEL:
                end_page = None

            chunk = {
                "chunk_id": r_ids[i],
                "text": r_docs[i] if i < len(r_docs) else "",
                "source": meta.get("source", ""),
                "start_page": start_page,
                "end_page": end_page,
                "score": r_distances[i] if i < len(r_distances) else 0.0,
            }
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def count(self) -> int:
        """Returns the number of chunks currently stored in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """
        Deletes and recreates the collection, removing all stored data.
        The persistence directory and client remain intact.
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
