# Health route module - provides the health check endpoint for monitoring.
# Reports application status and key metrics.

from fastapi import APIRouter, Depends

from src.api.models.schemas import HealthResponse
from src.config.settings import get_settings
from src.rag.vector_store import VectorStore
from src.services.lead_service import LeadService
from src.utils.logger import get_logger

# Module logger for tracking health check requests
logger = get_logger(__name__)

# Create the router for health endpoints
router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the application health and return status metrics.

    Returns the application status, version, vector store document count,
    and total extracted leads count.

    Returns:
        HealthResponse: Health status with key metrics.
    """
    # Load application settings for version info
    settings = get_settings()

    # Get the vector store document count
    vector_store = VectorStore()
    vector_count = vector_store.get_document_count()

    # Get the lead count
    lead_service = LeadService()
    leads_count = lead_service.get_lead_count()

    # Log the health check
    logger.debug("Health check: vector_store=%d, leads=%d", vector_count, leads_count)

    # Return the health response
    return HealthResponse(
        status="healthy",
        version=settings.app.version,
        vector_store_count=vector_count,
        leads_count=leads_count,
    )
