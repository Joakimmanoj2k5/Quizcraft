from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Responsible for generating dense vector embeddings from text chunks or queries
    using a local SentenceTransformer model.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedder with the specified sentence-transformers model.
        Loads the model into memory only once per instance.
        
        Args:
            model_name (str): The name of the sentence-transformers model to load.
                              Defaults to 'all-MiniLM-L6-v2' which is lightweight and fast.
        """
        self.model_name = model_name
        # The model is downloaded (if not cached) and loaded into memory here
        self.model = SentenceTransformer(model_name)

    @property
    def max_seq_length(self) -> int:
        """Returns the maximum sequence length (in tokens) supported by the loaded model."""
        return self.model.max_seq_length
        
    @property
    def embedding_dimension(self) -> int:
        """Returns the embedding vector dimension of the loaded model."""
        return self.model.get_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings.
        
        Args:
            texts (List[str]): A list of string chunks.
            
        Returns:
            List[List[float]]: A list of lists of floats, representing the dense vector embeddings.
            
        Raises:
            TypeError: If the input is not a list of strings.
        """
        if not isinstance(texts, list):
            raise TypeError("Expected a list of strings.")
            
        if not texts:
            return []
            
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
        
    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single search query.
        
        Args:
            query (str): The search query string.
            
        Returns:
            List[float]: A list of floats representing the query embedding.
            
        Raises:
            ValueError: If the query is empty or just whitespace.
            TypeError: If the query is not a string.
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
            
        if not query.strip():
            raise ValueError("Query cannot be empty.")
            
        embedding = self.model.encode(query, show_progress_bar=False)
        return embedding.tolist()
