# Text splitter module - splits document text into overlapping chunks for RAG.
# Uses sentence-aware splitting to avoid breaking in the middle of sentences.

from src.models.lead_schemas import DocumentChunk
from src.utils.logger import get_logger

# Module logger for tracking text splitting operations
logger = get_logger(__name__)


class TextSplitter:
    """Splits document text into overlapping chunks for embedding and retrieval.

    Uses a sentence-aware approach that prefers splitting at sentence boundaries
    (periods, newlines) to maintain semantic coherence within each chunk.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize the text splitter with chunk size and overlap configuration.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
        """
        # Maximum character count for each text chunk
        self._chunk_size = chunk_size
        # Number of overlapping characters between adjacent chunks
        self._chunk_overlap = chunk_overlap

        # Log the configuration
        logger.debug("TextSplitter initialized: chunk_size=%d, overlap=%d", chunk_size, chunk_overlap)

    def split(self, text: str, document_name: str) -> list[DocumentChunk]:
        """Split a document's text content into overlapping chunks.

        Creates DocumentChunk objects with metadata about their position
        within the source document.

        Args:
            text: The full text content of the document to split.
            document_name: Name of the source document for metadata.

        Returns:
            list[DocumentChunk]: List of chunks with content and metadata.
        """
        # Return empty list if text is empty or whitespace-only
        if not text or not text.strip():
            logger.warning("Empty text provided for splitting, document: %s", document_name)
            return []

        # Split the text into sentence-aware chunks
        raw_chunks = self._create_chunks(text)

        # Convert raw text chunks into DocumentChunk model instances
        document_chunks = []
        for index, chunk_text in enumerate(raw_chunks):
            # Create a DocumentChunk with position metadata
            chunk = DocumentChunk(
                document_name=document_name,
                content=chunk_text,
                chunk_index=index,
                metadata={
                    "source": document_name,
                    "chunk_index": index,
                    "total_chunks": len(raw_chunks),
                    "char_count": len(chunk_text),
                },
            )
            document_chunks.append(chunk)

        # Log the splitting result
        logger.info("Split '%s' into %d chunks", document_name, len(document_chunks))

        return document_chunks

    def _create_chunks(self, text: str) -> list[str]:
        """Create overlapping text chunks with sentence-aware boundaries.

        Attempts to split at sentence boundaries (periods followed by space,
        double newlines) when possible, falling back to hard splits at chunk_size.

        Args:
            text: The full text to split into chunks.

        Returns:
            list[str]: List of text chunk strings.
        """
        # Initialize the list to hold chunk strings
        chunks = []
        # Track the current position in the text
        start = 0
        # Get the total length of the text
        text_length = len(text)

        while start < text_length:
            # Calculate the end position for this chunk
            end = start + self._chunk_size

            # If we've reached or passed the end of text, take the remainder
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to find a sentence boundary near the end of the chunk
            split_position = self._find_sentence_boundary(text, start, end)

            # Extract the chunk text and strip whitespace
            chunk_text = text[start:split_position].strip()

            # Only add non-empty chunks
            if chunk_text:
                chunks.append(chunk_text)

            # Move start forward by chunk_size minus overlap for the next chunk
            start = split_position - self._chunk_overlap

            # Ensure forward progress to prevent infinite loops
            if start <= (split_position - self._chunk_size):
                start = split_position

        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary position near the target end position.

        Searches backwards from the end position looking for sentence-ending
        punctuation followed by whitespace, preferring natural break points.

        Args:
            text: The full document text.
            start: The start position of the current chunk.
            end: The target end position for the chunk.

        Returns:
            int: The position to split at (a sentence boundary or the original end).
        """
        # Define the search window: look back up to 20% of chunk_size for a boundary
        search_start = max(start + (self._chunk_size // 2), end - (self._chunk_size // 5))

        # Search backwards from end for sentence boundaries
        best_boundary = end

        # Look for double newline (paragraph boundary) - strongest signal
        double_newline_pos = text.rfind("\n\n", search_start, end)
        if double_newline_pos != -1:
            best_boundary = double_newline_pos + 2
            return best_boundary

        # Look for period followed by space (sentence boundary)
        period_pos = text.rfind(". ", search_start, end)
        if period_pos != -1:
            best_boundary = period_pos + 2
            return best_boundary

        # Look for single newline as fallback
        newline_pos = text.rfind("\n", search_start, end)
        if newline_pos != -1:
            best_boundary = newline_pos + 1
            return best_boundary

        # No good boundary found, use the original end position
        return end
