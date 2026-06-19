# Retriever module - provides high-level document retrieval from the vector store.
# Combines embedding generation and vector search into a single retrieval interface.

from src.config.settings import get_settings
from src.rag.embeddings import EmbeddingsGenerator
from src.rag.vector_store import VectorStore
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

# Module logger for tracking retrieval operations
logger = get_logger(__name__)


class Retriever:
    """High-level retriever that combines embedding and vector search.

    Provides a simple interface for querying documents by natural language,
    handling embedding generation and distance filtering internally.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embeddings_generator: EmbeddingsGenerator | None = None,
    ) -> None:
        """Initialize the retriever with its dependencies.

        Args:
            vector_store: VectorStore instance for document search.
                         Defaults to new instance with configured settings.
            embeddings_generator: EmbeddingsGenerator instance for query embedding.
                                Defaults to new instance with configured model.
        """
        # Load settings for retrieval configuration
        settings = get_settings()

        # Initialize the vector store for document search
        self._vector_store = vector_store or VectorStore()
        # Initialize the embeddings generator for query embedding
        self._embeddings_generator = embeddings_generator or EmbeddingsGenerator()
        # Maximum number of results to retrieve
        self._top_k = settings.rag.top_k
        # Maximum distance threshold for filtering irrelevant results
        self._max_distance = settings.rag.max_distance

        # Log the retriever configuration
        logger.debug("Retriever initialized: top_k=%d, max_distance=%.2f", self._top_k, self._max_distance)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve relevant document chunks for a natural language query.

        Embeds the query, searches the vector store, and filters results
        by distance threshold before returning.

        Args:
            query: The natural language search query.
            top_k: Optional override for number of results. Defaults to configured value.

        Returns:
            list[dict]: List of result dictionaries with keys:
                - content: The chunk text content.
                - metadata: Chunk metadata dictionary.
                - distance: Cosine distance from the query.

        Raises:
            RetrievalError: If embedding or vector search fails.
        """
        # Use configured top_k if not overridden
        num_results = top_k or self._top_k

        # Log the retrieval query
        logger.info("Retrieving documents for query: '%s' (top_k=%d)", query[:100], num_results)

        try:
            # Generate the embedding vector for the query text
            query_embedding = self._embeddings_generator.embed_text(query)

            # Query the vector store with the embedding
            raw_results = self._vector_store.query(
                query_embedding=query_embedding,
                top_k=num_results,
            )

            # Parse and filter the raw ChromaDB results
            filtered_results = self._parse_and_filter_results(raw_results)

            # Log the retrieval outcome
            logger.info("Retrieved %d relevant chunks (filtered from raw results)", len(filtered_results))

            return filtered_results

        except RetrievalError:
            # Re-raise RetrievalErrors without wrapping
            raise
        except Exception as error:
            # Wrap unexpected errors
            raise RetrievalError(
                "Failed to retrieve documents",
                details={"query": query[:200], "error": str(error)},
            ) from error

    def _parse_and_filter_results(self, raw_results: dict) -> list[dict]:
        """Parse ChromaDB results and filter by distance threshold.

        Converts raw ChromaDB format into clean dictionaries and removes
        results that exceed the maximum distance threshold.

        Args:
            raw_results: Raw results dictionary from ChromaDB query.

        Returns:
            list[dict]: Filtered and parsed result dictionaries.
        """
        # Initialize the filtered results list
        filtered = []

        # Check if results are present
        if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
            logger.debug("No results returned from vector store")
            return filtered

        # Extract the first (and only) query's results
        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        # Iterate through results and filter by distance
        for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            # Skip results that exceed the distance threshold
            if distance > self._max_distance:
                logger.debug("Filtering out result %s (distance=%.3f > %.3f)", doc_id, distance, self._max_distance)
                continue

            # Add the result to the filtered list
            filtered.append(
                {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return filtered

    def build_context_string(self, results: list[dict]) -> str:
        """Build a combined context string from retrieval results.

        Concatenates all retrieved chunk contents with separators and
        source attribution for use in LLM prompts.

        Args:
            results: List of retrieval result dictionaries.

        Returns:
            str: Combined context string with source attributions.
        """
        # Return empty string if no results
        if not results:
            return ""

        # Build context sections from each result
        context_parts = []
        for index, result in enumerate(results, start=1):
            # Extract the source document name from metadata
            source = result.get("metadata", {}).get("source", "Unknown")
            # Format the chunk with its source attribution
            context_parts.append(f"[Source: {source}]\n{result['content']}")

        # Join all context sections with double newline separators
        return "\n\n---\n\n".join(context_parts)
