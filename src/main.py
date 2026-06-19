# Main module - FastAPI application factory and entry point.
# Configures the application, registers routes, and sets up middleware.

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from src.api.middleware import setup_middleware
from src.api.routes import chat, health, ingest, leads
from src.config.settings import get_settings
from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Module logger for tracking application lifecycle
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events.

    Performs initialization on startup and cleanup on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    # Startup: log application start
    settings = get_settings()
    logger.info("Starting %s v%s (env: %s)", settings.app.name, settings.app.version, settings.app.env)
    yield
    # Shutdown: log application stop
    logger.info("Shutting down %s", settings.app.name)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Registers routes, middleware, and sets up the application metadata.

    Returns:
        FastAPI: The fully configured FastAPI application.
    """
    # Load application settings
    settings = get_settings()

    # Create the FastAPI application with metadata
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="RAG + LangGraph agent for finding construction leads from ingested documents",
        lifespan=lifespan,
    )

    # Register route modules
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(leads.router)

    # Configure middleware (CORS, logging, exception handling)
    setup_middleware(app)

    # Log successful application creation
    logger.info("FastAPI application created with %d routes", len(app.routes))

    return app


# Create the application instance for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Load settings for server configuration
    settings = get_settings()

    # Run the application with uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.env == "development",
    )
