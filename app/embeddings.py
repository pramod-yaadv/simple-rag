from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingsUtils:
    """
    Utility class for generating text embeddings using Sentence Transformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingsUtils with a specific model.
        
        Args:
            model_name (str): The name of the sentence-transformers model to use.
                              Defaults to "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text string.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        embedding = self.model.encode(text)
        return embedding.tolist()

    def batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts (List[str]): A list of input texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
