# Test retriever module - unit tests for the Retriever class.
# Tests retrieval, filtering, and context string building.

import pytest
from unittest.mock import MagicMock, patch

from src.rag.retriever import Retriever


class TestRetriever:
    """Tests for the Retriever class."""

    def test_build_context_string_with_results(self, sample_retrieval_results):
        """Test that context string is built correctly from retrieval results."""
        # Create retriever with mocked dependencies
        mock_vector_store = MagicMock()
        mock_embeddings = MagicMock()
        retriever = Retriever(vector_store=mock_vector_store, embeddings_generator=mock_embeddings)

        # Build context string
        context = retriever.build_context_string(sample_retrieval_results)

        # Verify the context includes content from results
        assert "Building permit" in context
        assert "Highway Bridge" in context
        # Verify source attribution is included
        assert "permits_2025.pdf" in context
        assert "rfp_notice.txt" in context

    def test_build_context_string_empty_results_returns_empty(self):
        """Test that empty results produce an empty context string."""
        mock_vector_store = MagicMock()
        mock_embeddings = MagicMock()
        retriever = Retriever(vector_store=mock_vector_store, embeddings_generator=mock_embeddings)

        context = retriever.build_context_string([])

        assert context == ""

    def test_retrieve_calls_embeddings_and_vector_store(self):
        """Test that retrieve correctly chains embedding and query calls."""
        # Set up mocks
        mock_vector_store = MagicMock()
        mock_embeddings = MagicMock()

        # Configure mock returns
        mock_embeddings.embed_text.return_value = [0.1] * 384
        mock_vector_store.query.return_value = {
            "ids": [["id1"]],
            "documents": [["Test document content"]],
            "metadatas": [[{"source": "test.txt"}]],
            "distances": [[0.3]],
        }

        # Create retriever with mocks
        retriever = Retriever(vector_store=mock_vector_store, embeddings_generator=mock_embeddings)

        # Perform retrieval
        results = retriever.retrieve("find construction projects")

        # Verify the chain was called correctly
        mock_embeddings.embed_text.assert_called_once_with("find construction projects")
        mock_vector_store.query.assert_called_once()

        # Verify results are parsed
        assert len(results) == 1
        assert results[0]["content"] == "Test document content"

    def test_retrieve_filters_by_max_distance(self):
        """Test that results exceeding max_distance are filtered out."""
        mock_vector_store = MagicMock()
        mock_embeddings = MagicMock()

        # Return one close and one distant result
        mock_embeddings.embed_text.return_value = [0.1] * 384
        mock_vector_store.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["Close doc", "Far doc"]],
            "metadatas": [[{"source": "close.txt"}, {"source": "far.txt"}]],
            "distances": [[0.5, 2.0]],
        }

        retriever = Retriever(vector_store=mock_vector_store, embeddings_generator=mock_embeddings)
        results = retriever.retrieve("test query")

        # Only the close result should pass the distance filter
        assert len(results) == 1
        assert results[0]["content"] == "Close doc"

    def test_retrieve_empty_vector_store_returns_empty(self):
        """Test that retrieval from empty vector store returns empty list."""
        mock_vector_store = MagicMock()
        mock_embeddings = MagicMock()

        mock_embeddings.embed_text.return_value = [0.1] * 384
        mock_vector_store.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever = Retriever(vector_store=mock_vector_store, embeddings_generator=mock_embeddings)
        results = retriever.retrieve("query with no results")

        assert results == []
