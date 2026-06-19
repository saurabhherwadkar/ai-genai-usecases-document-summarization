# Test document loader module - unit tests for the DocumentLoader class.
# Tests loading of PDF, DOCX, and TXT files including error cases.

import pytest

from src.ingestion.document_loader import DocumentLoader
from src.utils.exceptions import IngestionError


class TestDocumentLoader:
    """Tests for the DocumentLoader class."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create a fresh loader instance for each test
        self.loader = DocumentLoader()

    def test_load_txt_file_successfully(self, tmp_path):
        """Test that a valid TXT file is loaded correctly."""
        # Create a temporary text file with construction content
        txt_file = tmp_path / "test_doc.txt"
        txt_file.write_text("Construction permit for new building.", encoding="utf-8")

        # Load the file
        content = self.loader.load(str(txt_file))

        # Verify the content was loaded correctly
        assert content == "Construction permit for new building."

    def test_load_txt_file_preserves_newlines(self, tmp_path):
        """Test that newlines are preserved in loaded text files."""
        # Create a multi-line text file
        txt_file = tmp_path / "multiline.txt"
        txt_file.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")

        # Load the file
        content = self.loader.load(str(txt_file))

        # Verify newlines are preserved
        assert "Line 1\nLine 2\nLine 3" in content

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading a nonexistent file raises IngestionError."""
        with pytest.raises(IngestionError) as exc_info:
            self.loader.load("/nonexistent/path/file.txt")

        assert "File not found" in exc_info.value.message

    def test_load_unsupported_format_raises_error(self, tmp_path):
        """Test that loading an unsupported format raises IngestionError."""
        # Create a file with unsupported extension
        unsupported_file = tmp_path / "data.csv"
        unsupported_file.write_text("col1,col2\nval1,val2", encoding="utf-8")

        with pytest.raises(IngestionError) as exc_info:
            self.loader.load(str(unsupported_file))

        assert "Unsupported file format" in exc_info.value.message

    def test_load_empty_txt_file_raises_error(self, tmp_path):
        """Test that loading an empty TXT file raises IngestionError."""
        # Create an empty text file
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(IngestionError) as exc_info:
            self.loader.load(str(empty_file))

        assert "Empty text file" in exc_info.value.message

    def test_load_directory_path_raises_error(self, tmp_path):
        """Test that passing a directory path raises IngestionError."""
        with pytest.raises(IngestionError) as exc_info:
            self.loader.load(str(tmp_path))

        assert "not a file" in exc_info.value.message

    def test_supported_formats_includes_expected_types(self):
        """Test that the supported formats mapping includes PDF, DOCX, and TXT."""
        assert ".pdf" in self.loader.SUPPORTED_FORMATS
        assert ".docx" in self.loader.SUPPORTED_FORMATS
        assert ".txt" in self.loader.SUPPORTED_FORMATS
