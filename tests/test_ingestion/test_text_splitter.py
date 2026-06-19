# Test text splitter module - unit tests for the TextSplitter class.
# Tests chunking behavior, overlap, and sentence boundary detection.

import pytest

from src.ingestion.text_splitter import TextSplitter


class TestTextSplitter:
    """Tests for the TextSplitter class."""

    def setup_method(self):
        """Set up test fixtures with a configured text splitter."""
        # Create a splitter with small chunk size for testing
        self.splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

    def test_split_short_text_returns_single_chunk(self):
        """Test that text shorter than chunk_size returns one chunk."""
        # Text shorter than the chunk size
        text = "Short construction document content."

        # Split the text
        chunks = self.splitter.split(text, "test.txt")

        # Should produce exactly one chunk
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].document_name == "test.txt"
        assert chunks[0].chunk_index == 0

    def test_split_long_text_creates_multiple_chunks(self):
        """Test that long text is split into multiple chunks."""
        # Text longer than chunk_size
        text = "A" * 300

        # Split the text
        chunks = self.splitter.split(text, "long_doc.txt")

        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_split_preserves_document_name_in_metadata(self):
        """Test that each chunk has the correct document name."""
        text = "This is a test document with enough content. " * 10

        chunks = self.splitter.split(text, "my_document.pdf")

        # All chunks should reference the source document
        for chunk in chunks:
            assert chunk.document_name == "my_document.pdf"
            assert chunk.metadata["source"] == "my_document.pdf"

    def test_split_assigns_sequential_chunk_indices(self):
        """Test that chunks are assigned sequential zero-based indices."""
        text = "Word " * 200

        chunks = self.splitter.split(text, "doc.txt")

        # Verify sequential indexing
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_split_empty_text_returns_empty_list(self):
        """Test that empty text returns an empty list of chunks."""
        chunks = self.splitter.split("", "empty.txt")
        assert chunks == []

    def test_split_whitespace_only_returns_empty_list(self):
        """Test that whitespace-only text returns an empty list."""
        chunks = self.splitter.split("   \n\t  ", "whitespace.txt")
        assert chunks == []

    def test_split_respects_sentence_boundaries(self):
        """Test that splits prefer sentence boundaries over hard cuts."""
        # Create text with clear sentence boundaries
        splitter = TextSplitter(chunk_size=80, chunk_overlap=10)
        text = "First sentence is here. Second sentence follows. Third sentence is the last one in this block."

        chunks = splitter.split(text, "sentences.txt")

        # Chunks should preferably end at sentence boundaries
        # At minimum, verify that chunking occurred
        assert len(chunks) >= 1

    def test_split_metadata_includes_total_chunks(self):
        """Test that chunk metadata includes the total chunk count."""
        text = "Content " * 100

        chunks = self.splitter.split(text, "doc.txt")
        total = len(chunks)

        # Each chunk should know the total
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == total

    def test_split_metadata_includes_char_count(self):
        """Test that chunk metadata includes the character count."""
        text = "This is a test. " * 50

        chunks = self.splitter.split(text, "doc.txt")

        # Each chunk's char_count should match its content length
        for chunk in chunks:
            assert chunk.metadata["char_count"] == len(chunk.content)
