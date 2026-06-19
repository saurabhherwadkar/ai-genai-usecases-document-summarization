# Ingestion service module - orchestrates the full document ingestion workflow.
# Coordinates the pipeline, embeddings generation, and vector store storage.

from src.ingestion.pipeline import IngestionPipeline
from src.models.lead_schemas import DocumentChunk
from src.rag.embeddings import EmbeddingsGenerator
from src.rag.vector_store import VectorStore
from src.utils.exceptions import IngestionError
from src.utils.logger import get_logger

# Module logger for tracking service operations
logger = get_logger(__name__)


class IngestionService:
    """Orchestrates the complete document ingestion workflow.

    Manages the end-to-end process of loading documents, splitting them
    into chunks, generating embeddings, and storing in the vector store.
    """

    def __init__(
        self,
        pipeline: IngestionPipeline | None = None,
        embeddings_generator: EmbeddingsGenerator | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize the ingestion service with its dependencies.

        Args:
            pipeline: The ingestion pipeline for document loading and splitting.
            embeddings_generator: The embeddings generator for creating vectors.
            vector_store: The vector store for persisting embeddings.
        """
        # Initialize the ingestion pipeline
        self._pipeline = pipeline or IngestionPipeline()
        # Initialize the embeddings generator
        self._embeddings_generator = embeddings_generator or EmbeddingsGenerator()
        # Initialize the vector store
        self._vector_store = vector_store or VectorStore()

        # Log initialization
        logger.info("IngestionService initialized")

    def ingest_file(self, file_path: str) -> dict:
        """Ingest a single file: load, chunk, embed, and store.

        Args:
            file_path: Path to the document file to ingest.

        Returns:
            dict: Result with status, documents_processed, and chunks_created.

        Raises:
            IngestionError: If any step of the ingestion process fails.
        """
        # Log the ingestion start
        logger.info("Ingesting file: %s", file_path)

        try:
            # Load and chunk the document through the pipeline
            chunks = self._pipeline.ingest_file(file_path)

            # Generate embeddings and store in vector store
            stored_count = self._embed_and_store_chunks(chunks)

            # Log the result
            logger.info("File ingestion complete: %d chunks stored", stored_count)

            return {
                "status": "success",
                "documents_processed": 1,
                "chunks_created": stored_count,
            }

        except IngestionError:
            raise
        except Exception as error:
            raise IngestionError(
                f"File ingestion failed: {str(error)}",
                details={"path": file_path, "error": str(error)},
            ) from error

    def ingest_directory(self, directory_path: str) -> dict:
        """Ingest all supported files from a directory.

        Args:
            directory_path: Path to the directory containing documents.

        Returns:
            dict: Result with status, documents_processed, and chunks_created.

        Raises:
            IngestionError: If the directory cannot be processed.
        """
        # Log the directory ingestion start
        logger.info("Ingesting directory: %s", directory_path)

        try:
            # Load and chunk all documents in the directory
            chunks = self._pipeline.ingest_directory(directory_path)

            # Generate embeddings and store in vector store
            stored_count = self._embed_and_store_chunks(chunks)

            # Count unique source documents
            unique_documents = set(chunk.document_name for chunk in chunks)

            # Log the result
            logger.info(
                "Directory ingestion complete: %d documents, %d chunks stored",
                len(unique_documents),
                stored_count,
            )

            return {
                "status": "success",
                "documents_processed": len(unique_documents),
                "chunks_created": stored_count,
            }

        except IngestionError:
            raise
        except Exception as error:
            raise IngestionError(
                f"Directory ingestion failed: {str(error)}",
                details={"path": directory_path, "error": str(error)},
            ) from error

    def _embed_and_store_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Generate embeddings for chunks and store them in the vector store.

        Processes chunks in batches for efficient embedding generation.

        Args:
            chunks: List of DocumentChunk objects to embed and store.

        Returns:
            int: Number of chunks successfully stored.

        Raises:
            IngestionError: If embedding or storage fails.
        """
        # Return 0 if no chunks to process
        if not chunks:
            logger.warning("No chunks to embed and store")
            return 0

        # Extract text content from chunks for batch embedding
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings for all chunks in batch
        logger.info("Generating embeddings for %d chunks", len(chunks))
        embeddings = self._embeddings_generator.embed_batch(texts)

        # Prepare data for vector store insertion
        ids = [chunk.chunk_id for chunk in chunks]
        documents = texts
        metadatas = [chunk.metadata for chunk in chunks]

        # Store in the vector store
        logger.info("Storing %d chunks in vector store", len(chunks))
        self._vector_store.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)
