# Ingest route module - provides endpoints for document ingestion.
# Handles directory-based ingestion and individual file uploads.

import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from src.api.models.schemas import IngestRequest, IngestResponse
from src.services.ingestion_service import IngestionService
from src.utils.input_sanitizer import InputSanitizer
from src.utils.logger import get_logger

# Module logger for tracking ingestion requests
logger = get_logger(__name__)

# Create the router for ingestion endpoints
router = APIRouter(prefix="/api", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Ingest documents from a directory or specific file paths.

    Processes all supported document formats in the specified location
    and stores their content in the vector store.

    Args:
        request: The ingestion request with directory or file paths.

    Returns:
        IngestResponse: Result with status and counts.
    """
    # Initialize the ingestion service
    service = IngestionService()

    # Log the ingestion request
    logger.info("Ingest request received: directory=%s, files=%s", request.directory_path, request.file_paths)

    # Handle directory-based ingestion
    if request.directory_path:
        # Sanitize the directory path for security
        sanitized_path = InputSanitizer.sanitize_file_path(request.directory_path)
        # Perform directory ingestion
        result = service.ingest_directory(sanitized_path)
        return IngestResponse(**result)

    # Handle file-paths-based ingestion
    if request.file_paths:
        total_documents = 0
        total_chunks = 0

        # Process each file path
        for file_path in request.file_paths:
            # Sanitize each file path
            sanitized_path = InputSanitizer.sanitize_file_path(file_path)
            # Ingest the individual file
            result = service.ingest_file(sanitized_path)
            total_documents += result["documents_processed"]
            total_chunks += result["chunks_created"]

        return IngestResponse(
            status="success",
            documents_processed=total_documents,
            chunks_created=total_chunks,
        )

    # No path provided - return error response
    return IngestResponse(
        status="failed",
        documents_processed=0,
        chunks_created=0,
    )


@router.post("/ingest/upload", response_model=IngestResponse)
async def upload_and_ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a file and ingest it into the vector store.

    Saves the uploaded file to a temporary location, processes it
    through the ingestion pipeline, then cleans up.

    Args:
        file: The uploaded file from the multipart form request.

    Returns:
        IngestResponse: Result with status and counts.
    """
    # Log the upload request
    logger.info("File upload received: %s (%s)", file.filename, file.content_type)

    # Create a temporary file to store the upload
    suffix = Path(file.filename).suffix if file.filename else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        # Read the uploaded file content
        content = await file.read()
        # Write to the temporary file
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Initialize the ingestion service
        service = IngestionService()

        # Ingest the temporary file
        result = service.ingest_file(temp_path)

        # Log successful ingestion
        logger.info("Upload ingestion complete: %s -> %d chunks", file.filename, result["chunks_created"])

        return IngestResponse(**result)

    finally:
        # Clean up the temporary file
        temp_file_path = Path(temp_path)
        if temp_file_path.exists():
            temp_file_path.unlink()
            logger.debug("Temporary file cleaned up: %s", temp_path)
