# Middleware module - provides CORS, global exception handling, and request logging.
# Configures cross-cutting concerns for the FastAPI application.

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.exceptions import ApplicationError
from src.utils.logger import get_logger

# Module logger for tracking middleware operations
logger = get_logger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """Configure all middleware for the FastAPI application.

    Sets up CORS, request logging, and global exception handling.

    Args:
        app: The FastAPI application instance to configure.
    """
    # Add CORS middleware for cross-origin requests from the Streamlit UI
    _setup_cors(app)

    # Add request logging middleware
    _setup_request_logging(app)

    # Add global exception handler for application errors
    _setup_exception_handlers(app)

    # Log successful middleware setup
    logger.info("Middleware configured successfully")


def _setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware for the application.

    Allows requests from the Streamlit UI and local development origins.

    Args:
        app: The FastAPI application instance.
    """
    # Add CORS middleware with permissive settings for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _setup_request_logging(app: FastAPI) -> None:
    """Configure request logging middleware.

    Logs request method, path, and response time for all requests.

    Args:
        app: The FastAPI application instance.
    """

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log each HTTP request with method, path, and elapsed time.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            Response: The HTTP response from the handler.
        """
        # Record the start time for elapsed time calculation
        start_time = time.time()

        # Process the request through the handler chain
        response = await call_next(request)

        # Calculate the elapsed time in milliseconds
        elapsed_ms = (time.time() - start_time) * 1000

        # Log the request details at info level
        logger.info(
            "%s %s - %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response


def _setup_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers for the application.

    Catches ApplicationError and unhandled exceptions, returning
    structured JSON error responses.

    Args:
        app: The FastAPI application instance.
    """

    @app.exception_handler(ApplicationError)
    async def handle_application_error(request: Request, error: ApplicationError) -> JSONResponse:
        """Handle known application errors with structured JSON responses.

        Args:
            request: The HTTP request that triggered the error.
            error: The ApplicationError instance.

        Returns:
            JSONResponse: Structured error response with message and details.
        """
        # Log the application error
        logger.error("Application error on %s %s: %s", request.method, request.url.path, error.message)

        # Return a structured error response
        return JSONResponse(
            status_code=400,
            content={
                "error": error.__class__.__name__,
                "message": error.message,
                "details": error.details,
            },
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_error(request: Request, error: Exception) -> JSONResponse:
        """Handle unexpected errors with a generic error response.

        Args:
            request: The HTTP request that triggered the error.
            error: The unhandled exception.

        Returns:
            JSONResponse: Generic 500 error response without internal details.
        """
        # Log the full error details for debugging
        logger.error(
            "Unhandled error on %s %s: %s",
            request.method,
            request.url.path,
            str(error),
            exc_info=True,
        )

        # Return a generic error response (avoid leaking internals)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )
