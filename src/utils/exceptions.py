# Custom exceptions module - defines the application exception hierarchy.
# Each exception type represents a specific failure domain for targeted error handling.


class ApplicationError(Exception):
    """Base exception for all application-specific errors.

    All custom exceptions inherit from this class to allow catching
    any application error with a single except clause.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize the application error with a message and optional details.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional error context.
        """
        # Store the error message for display and logging
        self.message = message
        # Store additional context details for debugging
        self.details = details or {}
        super().__init__(self.message)


class IngestionError(ApplicationError):
    """Raised when document ingestion fails.

    Covers failures in loading, parsing, chunking, or storing documents.
    """

    pass


class RetrievalError(ApplicationError):
    """Raised when RAG retrieval from the vector store fails.

    Covers failures in embedding queries, searching ChromaDB, or assembling context.
    """

    pass


class LLMError(ApplicationError):
    """Raised when the LLM API call fails.

    Covers failures in Anthropic Claude API calls including rate limits,
    timeouts, and invalid responses.
    """

    pass


class LeadExtractionError(ApplicationError):
    """Raised when construction lead extraction fails.

    Covers failures in parsing LLM responses into structured lead data
    or when the extracted data fails validation.
    """

    pass


class ValidationError(ApplicationError):
    """Raised when input validation fails.

    Covers failures in user input sanitization, query validation,
    or request parameter validation.
    """

    pass
