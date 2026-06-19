# Lead service module - manages construction lead storage, retrieval, and lifecycle.
# Provides CRUD operations for leads extracted by the agent.

import json
from pathlib import Path

from src.config.settings import get_settings
from src.models.lead_schemas import ScoredLead
from src.utils.logger import get_logger

# Module logger for tracking lead service operations
logger = get_logger(__name__)

# File path for persisting leads as JSON
LEADS_STORAGE_FILE = "leads_data.json"


class LeadService:
    """Manages the lifecycle of extracted construction leads.

    Provides methods to store, retrieve, filter, and delete leads.
    Persists leads to a JSON file for simple storage without a database.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize the lead service with a storage file path.

        Args:
            storage_path: Path to the JSON file for lead storage.
                         Defaults to leads_data.json in the project root.
        """
        # Set the storage file path
        if storage_path:
            self._storage_path = Path(storage_path)
        else:
            self._storage_path = Path(LEADS_STORAGE_FILE)

        # Load existing leads from storage on initialization
        self._leads: list[ScoredLead] = self._load_leads_from_storage()

        # Log initialization with lead count
        logger.info("LeadService initialized with %d existing leads", len(self._leads))

    def add_leads(self, leads: list[ScoredLead]) -> int:
        """Add new scored leads to the storage.

        Args:
            leads: List of ScoredLead objects to store.

        Returns:
            int: Number of leads successfully added.
        """
        # Add the new leads to the in-memory list
        self._leads.extend(leads)

        # Persist to storage file
        self._save_leads_to_storage()

        # Log the addition
        logger.info("Added %d leads. Total leads: %d", len(leads), len(self._leads))

        return len(leads)

    def get_all_leads(
        self,
        limit: int = 50,
        offset: int = 0,
        min_score: float = 0.0,
        project_type: str | None = None,
    ) -> tuple[list[ScoredLead], int]:
        """Get all leads with optional filtering and pagination.

        Args:
            limit: Maximum number of leads to return.
            offset: Number of leads to skip from the beginning.
            min_score: Minimum score threshold for filtering.
            project_type: Optional filter by project type.

        Returns:
            tuple: (list of filtered leads, total count before pagination).
        """
        # Start with all leads
        filtered = self._leads

        # Apply minimum score filter
        if min_score > 0:
            filtered = [lead for lead in filtered if lead.score >= min_score]

        # Apply project type filter if specified
        if project_type:
            filtered = [lead for lead in filtered if lead.lead.project.project_type.lower() == project_type.lower()]

        # Get total count before pagination
        total = len(filtered)

        # Apply pagination
        paginated = filtered[offset : offset + limit]

        # Log the query
        logger.debug("Get leads: total=%d, returned=%d (offset=%d, limit=%d)", total, len(paginated), offset, limit)

        return paginated, total

    def get_lead_by_id(self, lead_id: str) -> ScoredLead | None:
        """Get a single lead by its unique identifier.

        Args:
            lead_id: The unique identifier of the lead to retrieve.

        Returns:
            ScoredLead | None: The matching lead, or None if not found.
        """
        # Search for the lead by ID
        for scored_lead in self._leads:
            if scored_lead.lead.id == lead_id:
                return scored_lead

        # Log that the lead was not found
        logger.debug("Lead not found: %s", lead_id)
        return None

    def get_lead_count(self) -> int:
        """Get the total number of stored leads.

        Returns:
            int: Total lead count.
        """
        return len(self._leads)

    def clear_all_leads(self) -> None:
        """Delete all stored leads.

        Clears both in-memory storage and the persistent file.
        """
        # Clear the in-memory list
        self._leads = []

        # Remove the storage file if it exists
        if self._storage_path.exists():
            self._storage_path.unlink()

        # Log the deletion
        logger.info("All leads cleared")

    def _save_leads_to_storage(self) -> None:
        """Persist current leads to the JSON storage file.

        Serializes all leads to JSON and writes to the configured file path.
        """
        try:
            # Serialize leads to JSON-compatible dictionaries
            leads_data = [lead.model_dump(mode="json") for lead in self._leads]

            # Write to the storage file
            with open(self._storage_path, "w", encoding="utf-8") as storage_file:
                json.dump(leads_data, storage_file, indent=2, default=str)

            # Log successful save
            logger.debug("Saved %d leads to %s", len(self._leads), self._storage_path)

        except Exception as error:
            # Log the error but don't raise (storage is best-effort)
            logger.error("Failed to save leads to storage: %s", str(error))

    def _load_leads_from_storage(self) -> list[ScoredLead]:
        """Load leads from the JSON storage file.

        Returns:
            list[ScoredLead]: List of leads loaded from storage, or empty list if file doesn't exist.
        """
        # Return empty list if storage file doesn't exist
        if not self._storage_path.exists():
            return []

        try:
            # Read and parse the JSON storage file
            with open(self._storage_path, "r", encoding="utf-8") as storage_file:
                leads_data = json.load(storage_file)

            # Convert dictionaries back to ScoredLead models
            leads = [ScoredLead.model_validate(lead_dict) for lead_dict in leads_data]

            # Log successful load
            logger.debug("Loaded %d leads from %s", len(leads), self._storage_path)

            return leads

        except Exception as error:
            # Log the error and return empty list
            logger.error("Failed to load leads from storage: %s", str(error))
            return []
