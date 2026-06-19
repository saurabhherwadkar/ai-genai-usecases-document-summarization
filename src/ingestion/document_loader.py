# Document loader module - loads text content from PDF, DOCX, and TXT files.
# Uses format-specific parsers with a unified interface for the ingestion pipeline.

from pathlib import Path

import docx
from PyPDF2 import PdfReader

from src.utils.exceptions import IngestionError
from src.utils.logger import get_logger

# Module logger for tracking document loading events
logger = get_logger(__name__)


class DocumentLoader:
    """Loads text content from various document formats.

    Supports PDF, DOCX, and TXT file formats. Each format has a dedicated
    parsing method that extracts raw text content from the file.
    """

    # Mapping of file extensions to their respective loader methods
    SUPPORTED_FORMATS = {
        ".pdf": "_load_pdf",
        ".docx": "_load_docx",
        ".txt": "_load_txt",
    }

    def load(self, file_path: str) -> str:
        """Load text content from a document file.

        Determines the file format from the extension and delegates to
        the appropriate format-specific loading method.

        Args:
            file_path: Absolute or relative path to the document file.

        Returns:
            str: The extracted text content from the document.

        Raises:
            IngestionError: If the file does not exist, format is unsupported, or parsing fails.
        """
        # Convert the path string to a Path object for validation
        path = Path(file_path)

        # Verify that the file exists on disk
        if not path.exists():
            raise IngestionError(f"File not found: {file_path}", details={"path": file_path})

        # Verify that the path points to a file, not a directory
        if not path.is_file():
            raise IngestionError(f"Path is not a file: {file_path}", details={"path": file_path})

        # Get the file extension in lowercase for format matching
        extension = path.suffix.lower()

        # Check that the file format is supported by this loader
        if extension not in self.SUPPORTED_FORMATS:
            raise IngestionError(
                f"Unsupported file format: {extension}",
                details={"path": file_path, "extension": extension, "supported": list(self.SUPPORTED_FORMATS.keys())},
            )

        # Get the loader method name from the format mapping
        loader_method_name = self.SUPPORTED_FORMATS[extension]

        # Get the actual method reference from this instance
        loader_method = getattr(self, loader_method_name)

        # Log the loading attempt at info level
        logger.info("Loading document: %s (format: %s)", path.name, extension)

        try:
            # Execute the format-specific loader and return the content
            content = loader_method(path)
            # Log successful loading with content length
            logger.info("Successfully loaded %d characters from %s", len(content), path.name)
            return content
        except IngestionError:
            # Re-raise IngestionErrors without wrapping
            raise
        except Exception as error:
            # Wrap unexpected errors in IngestionError for consistent handling
            raise IngestionError(
                f"Failed to load document: {path.name}",
                details={"path": file_path, "error": str(error)},
            ) from error

    def _load_pdf(self, path: Path) -> str:
        """Extract text content from a PDF file using PyPDF2.

        Iterates through all pages and concatenates their text content
        with newlines as separators.

        Args:
            path: Path object pointing to the PDF file.

        Returns:
            str: Concatenated text from all PDF pages.

        Raises:
            IngestionError: If the PDF has no extractable text.
        """
        # Initialize the PDF reader with the file path
        reader = PdfReader(str(path))

        # Extract text from each page and collect into a list
        pages_text = []
        for page_index, page in enumerate(reader.pages):
            # Extract text content from the current page
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
            # Log progress for large documents
            logger.debug("Extracted text from page %d of %s", page_index + 1, path.name)

        # Verify that at least some text was extracted
        if not pages_text:
            raise IngestionError(f"No text content found in PDF: {path.name}", details={"path": str(path)})

        # Join all page texts with newline separators
        return "\n".join(pages_text)

    def _load_docx(self, path: Path) -> str:
        """Extract text content from a DOCX file using python-docx.

        Iterates through all paragraphs and concatenates their text content
        with newlines as separators.

        Args:
            path: Path object pointing to the DOCX file.

        Returns:
            str: Concatenated text from all document paragraphs.

        Raises:
            IngestionError: If the DOCX has no extractable text.
        """
        # Open the DOCX file using python-docx
        document = docx.Document(str(path))

        # Extract text from each paragraph in the document
        paragraphs_text = []
        for paragraph in document.paragraphs:
            # Only include non-empty paragraphs
            if paragraph.text.strip():
                paragraphs_text.append(paragraph.text)

        # Verify that at least some text was extracted
        if not paragraphs_text:
            raise IngestionError(f"No text content found in DOCX: {path.name}", details={"path": str(path)})

        # Log the number of paragraphs extracted
        logger.debug("Extracted %d paragraphs from %s", len(paragraphs_text), path.name)

        # Join all paragraph texts with newline separators
        return "\n".join(paragraphs_text)

    def _load_txt(self, path: Path) -> str:
        """Load text content from a plain text file.

        Reads the entire file content using UTF-8 encoding.

        Args:
            path: Path object pointing to the text file.

        Returns:
            str: The complete file content as a string.

        Raises:
            IngestionError: If the file is empty.
        """
        # Read the entire file content with UTF-8 encoding
        content = path.read_text(encoding="utf-8")

        # Verify that the file is not empty
        if not content.strip():
            raise IngestionError(f"Empty text file: {path.name}", details={"path": str(path)})

        # Log successful text file reading
        logger.debug("Read %d characters from text file %s", len(content), path.name)

        return content
