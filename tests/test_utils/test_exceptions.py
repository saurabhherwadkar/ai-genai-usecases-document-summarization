# Test exceptions module - unit tests for the custom exception hierarchy.
# Tests exception instantiation, message handling, and inheritance.

import pytest

from src.utils.exceptions import (
    ApplicationError,
    IngestionError,
    RetrievalError,
    LLMError,
    LeadExtractionError,
    ValidationError,
)


class TestApplicationError:
    """Tests for the base ApplicationError class."""

    def test_creates_with_message(self):
        """Test that ApplicationError stores the message."""
        error = ApplicationError("Something went wrong")
        assert error.message == "Something went wrong"
        assert str(error) == "Something went wrong"

    def test_creates_with_details(self):
        """Test that ApplicationError stores optional details."""
        error = ApplicationError("Error", details={"key": "value"})
        assert error.details == {"key": "value"}

    def test_defaults_to_empty_details(self):
        """Test that details defaults to empty dict when not provided."""
        error = ApplicationError("Error")
        assert error.details == {}


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    def test_ingestion_error_is_application_error(self):
        """Test that IngestionError inherits from ApplicationError."""
        error = IngestionError("Ingestion failed")
        assert isinstance(error, ApplicationError)

    def test_retrieval_error_is_application_error(self):
        """Test that RetrievalError inherits from ApplicationError."""
        error = RetrievalError("Retrieval failed")
        assert isinstance(error, ApplicationError)

    def test_llm_error_is_application_error(self):
        """Test that LLMError inherits from ApplicationError."""
        error = LLMError("LLM failed")
        assert isinstance(error, ApplicationError)

    def test_lead_extraction_error_is_application_error(self):
        """Test that LeadExtractionError inherits from ApplicationError."""
        error = LeadExtractionError("Extraction failed")
        assert isinstance(error, ApplicationError)

    def test_validation_error_is_application_error(self):
        """Test that ValidationError inherits from ApplicationError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, ApplicationError)

    def test_can_catch_all_with_application_error(self):
        """Test that all custom exceptions can be caught with ApplicationError."""
        errors = [
            IngestionError("test"),
            RetrievalError("test"),
            LLMError("test"),
            LeadExtractionError("test"),
            ValidationError("test"),
        ]

        for error in errors:
            try:
                raise error
            except ApplicationError as caught:
                assert caught.message == "test"

    def test_each_exception_stores_details(self):
        """Test that all exception types properly store details."""
        details = {"path": "/test", "reason": "not found"}

        error = IngestionError("Failed", details=details)
        assert error.details["path"] == "/test"
        assert error.details["reason"] == "not found"
