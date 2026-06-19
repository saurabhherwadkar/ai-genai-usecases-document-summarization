# Ingestion pipeline module - orchestrates the full document ingestion workflow.
# Coordinates loading, splitting, and preparing documents for embedding and storage.

from pathlib import Path

from src.config.settings import get_settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.excel_loader import ExcelLoader
from src.ingestion.text_splitter import TextSplitter
from src.models.lead_schemas import DocumentChunk
from src.utils.exceptions import IngestionError
from src.utils.logger import get_logger

# Module logger for tracking pipeline operations
logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates the complete document ingestion workflow.

    Coordinates document loading, text splitting, and chunk preparation
    for downstream embedding and vector store storage.
    """

    def __init__(
        self,
        document_loader: DocumentLoader | None = None,
        excel_loader: ExcelLoader | None = None,
        text_splitter: TextSplitter | None = None,
    ) -> None:
        """Initialize the ingestion pipeline with its component dependencies.

        Uses dependency injection for testability. Falls back to default
        instances if no dependencies are provided.

        Args:
            document_loader: Loader for PDF, DOCX, TXT files. Defaults to new instance.
            excel_loader: Loader for Excel files. Defaults to new instance.
            text_splitter: Splitter for chunking text. Defaults to configured instance.
        """
        # Load application settings for chunk configuration
        settings = get_settings()

        # Initialize the document loader (PDF, DOCX, TXT)
        self._document_loader = document_loader or DocumentLoader()
        # Initialize the Excel-specific loader
        self._excel_loader = excel_loader or ExcelLoader()
        # Initialize the text splitter with configured chunk size and overlap
        self._text_splitter = text_splitter or TextSplitter(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
        )
        # Store supported formats from settings for validation
        self._supported_formats = settings.ingestion.supported_formats
        # Store max file size limit in bytes
        self._max_file_size_bytes = settings.ingestion.max_file_size_mb * 1024 * 1024

        # Log pipeline initialization
        logger.info(
            "IngestionPipeline initialized with chunk_size=%d, overlap=%d",
            settings.rag.chunk_size,
            settings.rag.chunk_overlap,
        )

    def ingest_file(self, file_path: str) -> list[DocumentChunk]:
        """Ingest a single file through the pipeline.

        Loads the file content, splits it into chunks, and returns
        the prepared DocumentChunk objects.

        Args:
            file_path: Path to the document file to ingest.

        Returns:
            list[DocumentChunk]: List of chunks ready for embedding.

        Raises:
            IngestionError: If the file cannot be loaded, is too large, or format is unsupported.
        """
        # Convert to Path object for file operations
        path = Path(file_path)

        # Validate the file exists
        if not path.exists():
            raise IngestionError(f"File not found: {file_path}", details={"path": file_path})

        # Validate file size against configured maximum
        self._validate_file_size(path)

        # Get the file extension for format routing
        extension = path.suffix.lower()

        # Validate that the format is supported
        if extension not in self._supported_formats:
            raise IngestionError(
                f"Unsupported format: {extension}",
                details={"path": file_path, "supported": self._supported_formats},
            )

        # Log the ingestion start
        logger.info("Ingesting file: %s", path.name)

        # Route to the appropriate loader based on file extension
        content = self._load_file_content(path, extension)

        # Split the loaded content into chunks
        chunks = self._text_splitter.split(content, path.name)

        # Log the ingestion result
        logger.info("Ingested %s: produced %d chunks", path.name, len(chunks))

        return chunks

    def ingest_directory(self, directory_path: str) -> list[DocumentChunk]:
        """Ingest all supported files from a directory.

        Scans the directory for files with supported extensions and
        ingests each one through the pipeline.

        Args:
            directory_path: Path to the directory containing documents.

        Returns:
            list[DocumentChunk]: Combined list of chunks from all ingested files.

        Raises:
            IngestionError: If the directory does not exist or contains no supported files.
        """
        # Convert to Path object for directory operations
        dir_path = Path(directory_path)

        # Validate the directory exists
        if not dir_path.exists():
            raise IngestionError(f"Directory not found: {directory_path}", details={"path": directory_path})

        # Validate that the path is a directory
        if not dir_path.is_dir():
            raise IngestionError(f"Path is not a directory: {directory_path}", details={"path": directory_path})

        # Collect all supported files from the directory
        supported_files = self._find_supported_files(dir_path)

        # Verify that at least one supported file was found
        if not supported_files:
            raise IngestionError(
                f"No supported files found in directory: {directory_path}",
                details={"path": directory_path, "supported_formats": self._supported_formats},
            )

        # Log the number of files to process
        logger.info("Found %d supported files in %s", len(supported_files), directory_path)

        # Ingest each file and collect all chunks
        all_chunks = []
        for file_path in supported_files:
            try:
                # Ingest the individual file
                chunks = self.ingest_file(str(file_path))
                all_chunks.extend(chunks)
            except IngestionError as error:
                # Log the error but continue processing remaining files
                logger.error("Failed to ingest %s: %s", file_path.name, error.message)

        # Log the total ingestion result
        logger.info(
            "Directory ingestion complete: %d total chunks from %d files", len(all_chunks), len(supported_files)
        )

        return all_chunks

    def _load_file_content(self, path: Path, extension: str) -> str:
        """Route file loading to the appropriate loader based on extension.

        Args:
            path: Path object for the file to load.
            extension: Lowercase file extension string.

        Returns:
            str: The text content extracted from the file.
        """
        # Route Excel files to the dedicated Excel loader
        if extension in (".xlsx", ".xls"):
            return self._excel_loader.load(str(path))

        # Route all other supported formats to the document loader
        return self._document_loader.load(str(path))

    def _validate_file_size(self, path: Path) -> None:
        """Validate that a file does not exceed the configured size limit.

        Args:
            path: Path object for the file to check.

        Raises:
            IngestionError: If the file exceeds the maximum allowed size.
        """
        # Get the file size in bytes
        file_size = path.stat().st_size

        # Compare against the configured maximum
        if file_size > self._max_file_size_bytes:
            raise IngestionError(
                f"File exceeds maximum size: {path.name} ({file_size / 1024 / 1024:.1f}MB > {self._max_file_size_bytes / 1024 / 1024:.0f}MB)",
                details={"path": str(path), "size_mb": file_size / 1024 / 1024},
            )

    def _find_supported_files(self, dir_path: Path) -> list[Path]:
        """Find all files with supported extensions in a directory.

        Searches only the immediate directory level (non-recursive).

        Args:
            dir_path: Path to the directory to scan.

        Returns:
            list[Path]: List of Path objects for supported files, sorted by name.
        """
        # Collect files matching any supported extension
        supported_files = []
        for item in sorted(dir_path.iterdir()):
            # Check if the item is a file with a supported extension
            if item.is_file() and item.suffix.lower() in self._supported_formats:
                supported_files.append(item)

        return supported_files
