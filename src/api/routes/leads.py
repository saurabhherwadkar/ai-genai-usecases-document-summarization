# Leads route module - provides endpoints for browsing and managing extracted leads.
# Supports listing, filtering, and deletion of construction leads.

from fastapi import APIRouter, HTTPException, Query

from src.api.models.schemas import LeadsListResponse
from src.models.lead_schemas import ScoredLead
from src.services.lead_service import LeadService
from src.utils.logger import get_logger

# Module logger for tracking leads requests
logger = get_logger(__name__)

# Create the router for leads endpoints
router = APIRouter(prefix="/api", tags=["leads"])


@router.get("/leads", response_model=LeadsListResponse)
async def list_leads(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results per page"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum score filter"),
    project_type: str | None = Query(default=None, description="Filter by project type"),
) -> LeadsListResponse:
    """List all extracted construction leads with filtering and pagination.

    Args:
        limit: Maximum number of leads to return per page.
        offset: Number of leads to skip for pagination.
        min_score: Minimum quality score threshold.
        project_type: Optional filter by project type category.

    Returns:
        LeadsListResponse: Paginated list of scored leads with total count.
    """
    # Initialize the lead service
    lead_service = LeadService()

    # Log the leads request
    logger.info(
        "Leads list request: limit=%d, offset=%d, min_score=%.2f, type=%s", limit, offset, min_score, project_type
    )

    # Retrieve filtered and paginated leads
    leads, total = lead_service.get_all_leads(
        limit=limit,
        offset=offset,
        min_score=min_score,
        project_type=project_type,
    )

    # Return the paginated response
    return LeadsListResponse(
        leads=leads,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/leads/{lead_id}", response_model=ScoredLead)
async def get_lead(lead_id: str) -> ScoredLead:
    """Get a single construction lead by its unique identifier.

    Args:
        lead_id: The unique identifier of the lead to retrieve.

    Returns:
        ScoredLead: The matching scored lead.

    Raises:
        HTTPException: 404 if the lead is not found.
    """
    # Initialize the lead service
    lead_service = LeadService()

    # Look up the lead by ID
    lead = lead_service.get_lead_by_id(lead_id)

    # Raise 404 if not found
    if lead is None:
        logger.warning("Lead not found: %s", lead_id)
        raise HTTPException(status_code=404, detail=f"Lead not found: {lead_id}")

    # Log the successful retrieval
    logger.debug("Lead retrieved: %s", lead_id)

    return lead


@router.delete("/leads")
async def clear_leads() -> dict:
    """Delete all stored construction leads.

    Clears the leads storage completely. This action is irreversible.

    Returns:
        dict: Confirmation of the deletion.
    """
    # Initialize the lead service
    lead_service = LeadService()

    # Clear all leads
    lead_service.clear_all_leads()

    # Log the deletion
    logger.info("All leads cleared via API")

    # Return confirmation
    return {"status": "cleared", "message": "All leads have been deleted."}
