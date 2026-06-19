# Test pipeline module - unit tests for the IngestionPipeline class.
# Tests file and directory ingestion orchestration.

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.pipeline import IngestionPipeline
from src.models.lead_schemas import DocumentChunk
from src.utils.exceptions import IngestionError


class TestIngestionPipeline:
    """Tests for the IngestionPipeline class."""

    def test_ingest_file_returns_chunks(self, tmp_path):
        """Test that ingesting a valid file returns document chunks."""
        # Create a sample text file
        txt_file = tmp_path / "project.txt"
        txt_file.write_text("Construction project details. Budget is $5M.", encoding="utf-8")

        # Create pipeline and ingest
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_file(str(txt_file))

        # Verify chunks were created
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert chunks[0].document_name == "project.txt"

    def test_ingest_file_nonexistent_raises_error(self):
        """Test that ingesting a nonexistent file raises IngestionError."""
        pipeline = IngestionPipeline()

        with pytest.raises(IngestionError) as exc_info:
            pipeline.ingest_file("/nonexistent/file.txt")

        assert "File not found" in exc_info.value.message

    def test_ingest_directory_processes_all_supported_files(self, tmp_documents_dir):
        """Test that directory ingestion processes all supported files."""
        # The tmp_documents_dir fixture provides a directory with 2 txt files
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_directory(str(tmp_documents_dir))

        # Should have chunks from both files
        assert len(chunks) > 0

        # Verify chunks come from different source documents
        sources = set(chunk.document_name for chunk in chunks)
        assert len(sources) == 2

    def test_ingest_directory_nonexistent_raises_error(self):
        """Test that ingesting a nonexistent directory raises IngestionError."""
        pipeline = IngestionPipeline()

        with pytest.raises(IngestionError) as exc_info:
            pipeline.ingest_directory("/nonexistent/directory")

        assert "Directory not found" in exc_info.value.message

    def test_ingest_directory_empty_directory_raises_error(self, tmp_path):
        """Test that ingesting an empty directory raises IngestionError."""
        pipeline = IngestionPipeline()

        with pytest.raises(IngestionError) as exc_info:
            pipeline.ingest_directory(str(tmp_path))

        assert "No supported files" in exc_info.value.message

    def test_ingest_file_unsupported_format_raises_error(self, tmp_path):
        """Test that unsupported file formats raise IngestionError."""
        # Create a file with unsupported extension
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2", encoding="utf-8")

        pipeline = IngestionPipeline()

        with pytest.raises(IngestionError) as exc_info:
            pipeline.ingest_file(str(csv_file))

        assert "Unsupported format" in exc_info.value.message
