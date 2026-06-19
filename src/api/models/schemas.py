# API schemas module - defines Pydantic request and response models for all API endpoints.
# Provides input validation and output serialization for the FastAPI routes.

from pydantic import BaseModel, Field

from src.models.lead_schemas import ScoredLead


class ChatRequest(BaseModel):
    """Request model for the chat endpoint.

    Validates the user query and optional conversation tracking.
    """

    # The user's natural language query (required, 1-2000 characters)
    query: str = Field(min_length=1, max_length=2000, description="User's chat query")
    # Optional conversation identifier for session tracking
    conversation_id: str | None = Field(default=None, description="Optional conversation ID for session tracking")


class ChatResponse(BaseModel):
    """Response model for the chat endpoint.

    Contains the agent's response, extracted leads, and source documents.
    """

    # The natural language response text from the agent
    response: str = Field(description="Agent's natural language response")
    # List of scored construction leads extracted (if any)
    leads: list[ScoredLead] = Field(default_factory=list, description="Extracted construction leads")
    # List of source document names referenced in the response
    sources: list[str] = Field(default_factory=list, description="Source documents referenced")
    # The classified intent of the user's query
    query_intent: str = Field(default="", description="Classified query intent")


class IngestRequest(BaseModel):
    """Request model for the document ingestion endpoint.

    Accepts either a directory path or a list of file paths for ingestion.
    """

    # Optional directory path containing documents to ingest
    directory_path: str | None = Field(default=None, description="Directory path to ingest")
    # Optional list of specific file paths to ingest
    file_paths: list[str] | None = Field(default=None, description="Specific file paths to ingest")


class IngestResponse(BaseModel):
    """Response model for the document ingestion endpoint.

    Reports the outcome of the ingestion operation.
    """

    # Status of the ingestion operation
    status: str = Field(description="Ingestion status (success/failed)")
    # Number of documents successfully processed
    documents_processed: int = Field(description="Number of documents ingested")
    # Number of text chunks created and stored
    chunks_created: int = Field(description="Number of chunks stored in vector store")


class LeadsListResponse(BaseModel):
    """Response model for the leads listing endpoint.

    Provides paginated access to stored construction leads.
    """

    # List of scored leads for the current page
    leads: list[ScoredLead] = Field(description="List of scored construction leads")
    # Total number of leads matching the filter criteria
    total: int = Field(description="Total leads matching filter")
    # The limit used for this page
    limit: int = Field(description="Page size limit")
    # The offset used for this page
    offset: int = Field(description="Page offset")


class HealthResponse(BaseModel):
    """Response model for the health check endpoint.

    Reports application status and key metrics.
    """

    # Overall application health status
    status: str = Field(description="Application health status")
    # Application version string
    version: str = Field(description="Application version")
    # Number of documents in the vector store
    vector_store_count: int = Field(default=0, description="Documents in vector store")
    # Number of extracted leads in storage
    leads_count: int = Field(default=0, description="Total extracted leads")
