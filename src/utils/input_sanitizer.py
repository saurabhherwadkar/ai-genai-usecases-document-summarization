# Input sanitizer module - validates and sanitizes user inputs to prevent injection attacks.
# Provides methods for query validation, path sanitization, and injection pattern detection.

import re

from src.utils.exceptions import ValidationError
from src.utils.logger import get_logger

# Module logger for tracking sanitization events
logger = get_logger(__name__)

# Maximum allowed query length in characters
MAX_QUERY_LENGTH = 5000

# Maximum allowed file path length in characters
MAX_PATH_LENGTH = 500

# Regex patterns that indicate potential prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts|rules)",
    r"disregard\s+(previous|above|all)",
    r"system\s*:\s*you\s+are",
    r"<\s*script\s*>",
    r"javascript\s*:",
    r"on(error|load|click)\s*=",
    r"\{\{.*\}\}",
    r"<%.*%>",
]

# Compiled regex patterns for efficient repeated matching
COMPILED_INJECTION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]

# Patterns indicating path traversal attempts
PATH_TRAVERSAL_PATTERNS = [
    r"\.\.",
    r"~",
    r"%2e%2e",
    r"%252e%252e",
]

# Compiled path traversal patterns for efficient matching
COMPILED_PATH_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PATH_TRAVERSAL_PATTERNS]


class InputSanitizer:
    """Validates and sanitizes user inputs for security.

    Provides static methods for sanitizing queries, file paths, and detecting
    potential injection attacks before they reach the application logic.
    """

    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitize a user query by removing control characters and validating length.

        Args:
            query: The raw user query string to sanitize.

        Returns:
            str: The sanitized query string with control characters removed.

        Raises:
            ValidationError: If the query is empty, too long, or contains injection patterns.
        """
        # Check that the query is not empty or whitespace-only
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty", details={"query": query})

        # Remove control characters that could cause display or processing issues
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)

        # Strip leading and trailing whitespace
        sanitized = sanitized.strip()

        # Enforce maximum query length to prevent resource exhaustion
        if len(sanitized) > MAX_QUERY_LENGTH:
            raise ValidationError(
                f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters",
                details={"length": len(sanitized), "max_length": MAX_QUERY_LENGTH},
            )

        # Check for prompt injection patterns in the sanitized query
        InputSanitizer._detect_injection(sanitized)

        # Log the successful sanitization at debug level
        logger.debug("Query sanitized successfully, length: %d", len(sanitized))

        return sanitized

    @staticmethod
    def sanitize_file_path(file_path: str) -> str:
        """Sanitize a file path by checking for traversal attacks and length.

        Args:
            file_path: The raw file path string to validate.

        Returns:
            str: The validated file path string.

        Raises:
            ValidationError: If the path is empty, too long, or contains traversal patterns.
        """
        # Check that the file path is not empty
        if not file_path or not file_path.strip():
            raise ValidationError("File path cannot be empty", details={"path": file_path})

        # Strip whitespace from the path
        sanitized = file_path.strip()

        # Enforce maximum path length
        if len(sanitized) > MAX_PATH_LENGTH:
            raise ValidationError(
                f"File path exceeds maximum length of {MAX_PATH_LENGTH} characters",
                details={"length": len(sanitized), "max_length": MAX_PATH_LENGTH},
            )

        # Check for path traversal attack patterns
        for pattern in COMPILED_PATH_PATTERNS:
            if pattern.search(sanitized):
                logger.warning("Path traversal attempt detected: %s", sanitized)
                raise ValidationError(
                    "Path traversal detected in file path",
                    details={"path": sanitized, "pattern": pattern.pattern},
                )

        # Log successful path validation at debug level
        logger.debug("File path sanitized successfully: %s", sanitized)

        return sanitized

    @staticmethod
    def _detect_injection(text: str) -> None:
        """Check text for known prompt injection patterns.

        Args:
            text: The text to scan for injection patterns.

        Raises:
            ValidationError: If an injection pattern is detected in the text.
        """
        # Iterate through compiled injection patterns and check for matches
        for pattern in COMPILED_INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning("Potential prompt injection detected matching pattern: %s", pattern.pattern)
                raise ValidationError(
                    "Potential prompt injection detected",
                    details={"pattern": pattern.pattern},
                )
