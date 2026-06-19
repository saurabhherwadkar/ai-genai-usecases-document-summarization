# Lead extractor tool module - extracts structured construction leads from context using Claude.
# Sends retrieved document context to the LLM for structured data extraction.

import json

from src.agent.llm_client import LLMClient
from src.agent.prompt_templates import LEAD_EXTRACTION_SYSTEM_PROMPT, LEAD_EXTRACTION_USER_TEMPLATE
from src.models.lead_schemas import ConstructionLead, ContactInfo, ProjectDetails
from src.utils.exceptions import LeadExtractionError
from src.utils.logger import get_logger

# Module logger for tracking lead extraction operations
logger = get_logger(__name__)


def extract_leads_from_context(context: str, query: str, llm_client: LLMClient) -> list[ConstructionLead]:
    """Extract structured construction leads from document context using Claude.

    Sends the retrieved context to Claude with a structured extraction prompt
    and parses the JSON response into ConstructionLead model instances.

    Args:
        context: The combined document context text from RAG retrieval.
        query: The original user query for context in extraction.
        llm_client: The LLM client instance for Claude API calls.

    Returns:
        list[ConstructionLead]: List of extracted and validated construction leads.

    Raises:
        LeadExtractionError: If the LLM response cannot be parsed into valid leads.
    """
    # Return empty list if context is empty
    if not context or not context.strip():
        logger.warning("Empty context provided to lead extractor, returning no leads")
        return []

    # Log the extraction attempt
    logger.info("Extracting leads from context (%d chars) for query: '%s'", len(context), query[:100])

    try:
        # Format the user message with the context and query
        user_message = LEAD_EXTRACTION_USER_TEMPLATE.format(context=context, query=query)

        # Call Claude for structured extraction
        response_text = llm_client.generate_response(
            system_prompt=LEAD_EXTRACTION_SYSTEM_PROMPT,
            user_message=user_message,
        )

        # Parse the JSON response from Claude
        leads = _parse_extraction_response(response_text)

        # Log the extraction result
        logger.info("Extracted %d construction leads from context", len(leads))

        return leads

    except LeadExtractionError:
        # Re-raise extraction errors without wrapping
        raise
    except Exception as error:
        # Wrap unexpected errors
        raise LeadExtractionError(
            f"Lead extraction failed: {str(error)}",
            details={"context_length": len(context), "error": str(error)},
        ) from error


def _parse_extraction_response(response_text: str) -> list[ConstructionLead]:
    """Parse the Claude JSON response into ConstructionLead model instances.

    Handles JSON extraction from the response text, including cases where
    the JSON is wrapped in markdown code blocks.

    Args:
        response_text: The raw text response from Claude containing JSON.

    Returns:
        list[ConstructionLead]: List of parsed ConstructionLead instances.

    Raises:
        LeadExtractionError: If the response cannot be parsed as valid JSON.
    """
    # Strip any markdown code block formatting from the response
    cleaned_response = _strip_code_blocks(response_text)

    try:
        # Parse the JSON response
        parsed_data = json.loads(cleaned_response)
    except json.JSONDecodeError as error:
        # Log the parsing failure with the raw response for debugging
        logger.error("Failed to parse LLM response as JSON: %s", str(error))
        raise LeadExtractionError(
            "Failed to parse lead extraction response as JSON",
            details={"response_preview": response_text[:500], "error": str(error)},
        ) from error

    # Extract the leads array from the parsed data
    leads_data = parsed_data.get("leads", [])

    # Convert each lead dictionary to a ConstructionLead model instance
    leads = []
    for lead_dict in leads_data:
        lead = _convert_dict_to_lead(lead_dict)
        if lead:
            leads.append(lead)

    return leads


def _convert_dict_to_lead(lead_dict: dict) -> ConstructionLead | None:
    """Convert a lead dictionary from the LLM response to a ConstructionLead model.

    Handles missing or malformed data gracefully by using defaults.

    Args:
        lead_dict: Dictionary containing lead data from the JSON response.

    Returns:
        ConstructionLead | None: The constructed lead model, or None if project_name is missing.
    """
    try:
        # Extract project details from the dictionary
        project_data = lead_dict.get("project", {})

        # Skip leads without a project name (minimum required field)
        if not project_data.get("project_name"):
            logger.debug("Skipping lead without project_name")
            return None

        # Create the ProjectDetails model from the extracted data
        project = ProjectDetails(**project_data)

        # Extract and create ContactInfo models
        contacts_data = lead_dict.get("contacts", [])
        contacts = [ContactInfo(**contact) for contact in contacts_data if isinstance(contact, dict)]

        # Create and return the complete ConstructionLead
        return ConstructionLead(
            project=project,
            contacts=contacts,
            source_documents=lead_dict.get("source_documents", []),
        )

    except Exception as error:
        # Log the conversion error but don't raise (skip this lead)
        logger.warning("Failed to convert lead dict to model: %s", str(error))
        return None


def _strip_code_blocks(text: str) -> str:
    """Remove markdown code block formatting from text.

    Handles responses where the JSON is wrapped in ```json ... ``` blocks.

    Args:
        text: The text that may contain code block markers.

    Returns:
        str: The text with code block markers removed.
    """
    # Strip leading/trailing whitespace
    stripped = text.strip()

    # Remove opening code block marker (```json or ```)
    if stripped.startswith("```"):
        # Find the end of the first line (after ```json or just ```)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]

    # Remove closing code block marker
    if stripped.endswith("```"):
        stripped = stripped[:-3]

    return stripped.strip()
