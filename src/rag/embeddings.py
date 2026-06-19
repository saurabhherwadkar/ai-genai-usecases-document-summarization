# Embeddings module - generates vector embeddings using sentence-transformers.
# Wraps the sentence-transformers library for batch and single text embedding.

from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

# Module logger for tracking embedding operations
logger = get_logger(__name__)


class EmbeddingsGenerator:
    """Generates vector embeddings from text using sentence-transformers.

    Lazily loads the embedding model on first use to avoid startup cost
    when embeddings are not immediately needed.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the embeddings generator with the model configuration.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to the configured model in settings.
        """
        # Load model name from settings if not provided
        settings = get_settings()
        # Store the model name for lazy loading
        self._model_name = model_name or settings.embeddings.model
        # Model instance will be loaded on first use
        self._model: SentenceTransformer | None = None

        # Log initialization with the configured model name
        logger.debug("EmbeddingsGenerator configured with model: %s", self._model_name)

    def _get_model(self) -> SentenceTransformer:
        """Get or lazily load the sentence-transformers model.

        Loads the model on first call and caches it for subsequent uses.

        Returns:
            SentenceTransformer: The loaded embedding model instance.

        Raises:
            RetrievalError: If the model fails to load.
        """
        # Return cached model if already loaded
        if self._model is not None:
            return self._model

        try:
            # Load the sentence-transformers model
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded successfully")
            return self._model
        except Exception as error:
            # Wrap model loading errors in RetrievalError
            raise RetrievalError(
                f"Failed to load embedding model: {self._model_name}",
                details={"model": self._model_name, "error": str(error)},
            ) from error

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: The text to embed.

        Returns:
            list[float]: The embedding vector as a list of floats.

        Raises:
            RetrievalError: If embedding generation fails.
        """
        try:
            # Get the model instance (loads if needed)
            model = self._get_model()
            # Generate the embedding and convert to list
            embedding = model.encode(text, convert_to_numpy=True)
            # Convert numpy array to Python list of floats
            return embedding.tolist()
        except RetrievalError:
            # Re-raise RetrievalErrors without wrapping
            raise
        except Exception as error:
            # Wrap unexpected errors
            raise RetrievalError(
                "Failed to generate embedding for text",
                details={"text_length": len(text), "error": str(error)},
            ) from error

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of text strings.

        Processes multiple texts in a single batch for efficiency,
        leveraging GPU parallelism when available.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors, one per input text.

        Raises:
            RetrievalError: If batch embedding generation fails.
        """
        # Return empty list for empty input
        if not texts:
            return []

        try:
            # Get the model instance (loads if needed)
            model = self._get_model()
            # Log the batch size for monitoring
            logger.debug("Generating embeddings for batch of %d texts", len(texts))
            # Generate embeddings for all texts in a single batch call
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # Convert each numpy array to a Python list of floats
            result = [embedding.tolist() for embedding in embeddings]
            # Log successful batch embedding
            logger.debug("Generated %d embeddings (dimension: %d)", len(result), len(result[0]) if result else 0)
            return result
        except RetrievalError:
            # Re-raise RetrievalErrors without wrapping
            raise
        except Exception as error:
            # Wrap unexpected errors
            raise RetrievalError(
                "Failed to generate batch embeddings",
                details={"batch_size": len(texts), "error": str(error)},
            ) from error
