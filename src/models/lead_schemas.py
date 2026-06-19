# Lead schemas module - defines Pydantic models for construction leads, contacts, and documents.
# These models represent the core domain entities extracted from construction documents.

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Contact information associated with a construction lead.

    Stores details about individuals or organizations involved in
    a construction project for business development outreach.
    """

    # Full name of the contact person
    name: str = Field(default="", description="Contact person name")
    # Company or organization the contact belongs to
    company: str = Field(default="", description="Company or organization name")
    # Email address for reaching the contact
    email: str = Field(default="", description="Email address")
    # Phone number for reaching the contact
    phone: str = Field(default="", description="Phone number")
    # Job title or role of the contact within their organization
    role: str = Field(default="", description="Role or title of the contact")


class ProjectDetails(BaseModel):
    """Detailed information about a construction project.

    Captures all relevant attributes of a construction project
    that constitute a viable business lead.
    """

    # Official name or title of the construction project
    project_name: str = Field(description="Name or title of the construction project")
    # Category of construction (commercial, residential, infrastructure, industrial)
    project_type: str = Field(default="", description="Type: commercial, residential, infrastructure, etc.")
    # Brief summary of the project scope and purpose
    description: str = Field(default="", description="Brief project description")
    # Physical location including address, city, and state
    location: str = Field(default="", description="Project location (address/city/state)")
    # Entity that owns or is developing the project
    owner: str = Field(default="", description="Project owner or developer")
    # Estimated budget or contract value as a string
    budget: str = Field(default="", description="Estimated budget or contract value")
    # Expected start and end dates or duration
    timeline: str = Field(default="", description="Project timeline or expected dates")
    # Current phase of the project lifecycle
    project_phase: str = Field(default="", description="Phase: planning, bidding, under construction, etc.")
    # Description of the work scope and deliverables
    scope_of_work: str = Field(default="", description="Scope description")
    # Size of the project in square feet if available
    square_footage: str = Field(default="", description="Project size in sq ft if available")


class ConstructionLead(BaseModel):
    """A single construction lead extracted from ingested documents.

    Represents a complete business lead combining project details,
    contacts, and metadata about the extraction source.
    """

    # Unique identifier for this lead
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique lead identifier")
    # Detailed project information for this lead
    project: ProjectDetails = Field(description="Project details")
    # List of contacts associated with this lead
    contacts: list[ContactInfo] = Field(default_factory=list, description="Associated contacts")
    # Names of the source documents from which this lead was extracted
    source_documents: list[str] = Field(default_factory=list, description="Source document filenames")
    # Timestamp when this lead was extracted
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Extraction timestamp"
    )
    # Original text passage from which the lead was derived
    raw_context: str = Field(default="", description="Original text context the lead was extracted from")


class ScoredLead(BaseModel):
    """A construction lead with associated quality and relevance scores.

    Wraps a ConstructionLead with scoring information used for
    ranking and filtering results returned to the user.
    """

    # The underlying construction lead data
    lead: ConstructionLead = Field(description="The construction lead")
    # Overall quality score from 0 to 1
    score: float = Field(default=0.0, description="Overall quality score 0-1")
    # Score based on how many fields are populated
    completeness_score: float = Field(default=0.0, description="Data completeness 0-1")
    # Score based on relevance to the user's query
    relevance_score: float = Field(default=0.0, description="Query relevance 0-1")
    # Breakdown of individual scoring factor values
    score_breakdown: dict = Field(default_factory=dict, description="Individual scoring factors")


class DocumentChunk(BaseModel):
    """A single chunk of text from an ingested document.

    Represents a segment of a document after text splitting,
    ready for embedding and storage in the vector store.
    """

    # Unique identifier for this chunk
    chunk_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chunk identifier")
    # Name of the source document file
    document_name: str = Field(description="Source document filename")
    # The text content of this chunk
    content: str = Field(description="Chunk text content")
    # Zero-based index of this chunk within the source document
    chunk_index: int = Field(description="Position index within the document")
    # Additional metadata about the chunk (page number, section, etc.)
    metadata: dict = Field(default_factory=dict, description="Additional chunk metadata")
