# Vector store module - manages ChromaDB persistent storage for document embeddings.
# Provides methods for adding, querying, and managing document chunks in the vector store.

import chromadb

from src.config.settings import get_settings
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

# Module logger for tracking vector store operations
logger = get_logger(__name__)


class VectorStore:
    """Manages ChromaDB persistent vector store for document embeddings.

    Provides a high-level interface for storing and querying document
    chunk embeddings with metadata filtering support.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialize the vector store with ChromaDB connection settings.

        Args:
            persist_directory: Directory for ChromaDB persistent storage.
                             Defaults to configured path.
            collection_name: Name of the ChromaDB collection.
                           Defaults to configured name.
        """
        # Load settings for default values
        settings = get_settings()

        # Set the persistence directory for ChromaDB data files
        self._persist_directory = persist_directory or settings.vector_store.persist_directory
        # Set the collection name within ChromaDB
        self._collection_name = collection_name or settings.vector_store.collection_name
        # ChromaDB client instance (lazily initialized)
        self._client: chromadb.PersistentClient | None = None
        # ChromaDB collection instance (lazily initialized)
        self._collection = None

        # Log the configuration
        logger.debug(
            "VectorStore configured: directory=%s, collection=%s",
            self._persist_directory,
            self._collection_name,
        )

    def _get_collection(self):
        """Get or create the ChromaDB collection.

        Lazily initializes the ChromaDB client and collection on first access.

        Returns:
            chromadb.Collection: The ChromaDB collection instance.

        Raises:
            RetrievalError: If ChromaDB client or collection initialization fails.
        """
        # Return cached collection if already initialized
        if self._collection is not None:
            return self._collection

        try:
            # Initialize the persistent ChromaDB client
            logger.info("Initializing ChromaDB client at: %s", self._persist_directory)
            self._client = chromadb.PersistentClient(path=self._persist_directory)

            # Get or create the named collection
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Log the collection status
            logger.info(
                "ChromaDB collection '%s' ready with %d documents",
                self._collection_name,
                self._collection.count(),
            )

            return self._collection
        except Exception as error:
            # Wrap initialization errors
            raise RetrievalError(
                "Failed to initialize ChromaDB collection",
                details={
                    "directory": self._persist_directory,
                    "collection": self._collection_name,
                    "error": str(error),
                },
            ) from error

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add document chunks with their embeddings to the vector store.

        Args:
            ids: Unique identifiers for each document chunk.
            embeddings: Embedding vectors for each chunk.
            documents: Text content of each chunk.
            metadatas: Metadata dictionaries for each chunk.

        Raises:
            RetrievalError: If adding documents to ChromaDB fails.
        """
        try:
            # Get the collection instance
            collection = self._get_collection()

            # Log the add operation
            logger.info("Adding %d documents to vector store", len(ids))

            # Add the documents in batch to ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            # Log successful addition
            logger.info("Successfully added %d documents. Total count: %d", len(ids), collection.count())

        except RetrievalError:
            # Re-raise RetrievalErrors without wrapping
            raise
        except Exception as error:
            # Wrap unexpected errors
            raise RetrievalError(
                "Failed to add documents to vector store",
                details={"count": len(ids), "error": str(error)},
            ) from error

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict | None = None,
    ) -> dict:
        """Query the vector store for similar documents.

        Performs a nearest-neighbor search using the query embedding vector
        and returns the most similar document chunks with distances.

        Args:
            query_embedding: The embedding vector of the search query.
            top_k: Maximum number of results to return.
            where: Optional metadata filter for restricting results.

        Returns:
            dict: ChromaDB query results with keys: ids, documents, metadatas, distances.

        Raises:
            RetrievalError: If the vector store query fails.
        """
        try:
            # Get the collection instance
            collection = self._get_collection()

            # Log the query operation
            logger.debug("Querying vector store: top_k=%d, filter=%s", top_k, where)

            # Build the query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }

            # Add metadata filter if provided
            if where:
                query_params["where"] = where

            # Execute the query against ChromaDB
            results = collection.query(**query_params)

            # Log the number of results returned
            result_count = len(results["ids"][0]) if results["ids"] else 0
            logger.debug("Vector store query returned %d results", result_count)

            return results

        except RetrievalError:
            # Re-raise RetrievalErrors without wrapping
            raise
        except Exception as error:
            # Wrap unexpected errors
            raise RetrievalError(
                "Failed to query vector store",
                details={"top_k": top_k, "error": str(error)},
            ) from error

    def get_document_count(self) -> int:
        """Get the total number of documents stored in the collection.

        Returns:
            int: Total document count in the vector store.
        """
        try:
            # Get the collection and return its count
            collection = self._get_collection()
            return collection.count()
        except Exception:
            # Return 0 if we cannot access the collection
            return 0

    def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB.

        Removes all documents and their embeddings. Use with caution
        as this operation is irreversible.

        Raises:
            RetrievalError: If collection deletion fails.
        """
        try:
            # Ensure the client is initialized
            if self._client is None:
                self._get_collection()

            # Delete the collection from ChromaDB
            logger.warning("Deleting collection: %s", self._collection_name)
            self._client.delete_collection(name=self._collection_name)

            # Reset the cached collection reference
            self._collection = None

            # Log successful deletion
            logger.info("Collection '%s' deleted successfully", self._collection_name)

        except Exception as error:
            # Wrap deletion errors
            raise RetrievalError(
                f"Failed to delete collection: {self._collection_name}",
                details={"collection": self._collection_name, "error": str(error)},
            ) from error
