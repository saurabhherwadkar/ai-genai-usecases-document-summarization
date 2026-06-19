# Lead extraction node module - extracts structured construction leads from retrieved context.
# Uses Claude to parse document context into structured ConstructionLead models.

from src.agent.llm_client import LLMClient
from src.agent.tools.lead_extractor_tool import extract_leads_from_context
from src.models.agent_state import LeadsAgentState
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

# Module logger for tracking lead extraction
logger = get_logger(__name__)


def lead_extraction_node(state: LeadsAgentState, llm_client: LLMClient, retriever: Retriever) -> dict:
    """Extract structured construction leads from the retrieved context.

    Builds a context string from the retrieved document chunks and sends
    it to Claude for structured lead extraction.

    Args:
        state: The current LangGraph agent state with retrieved_context.
        llm_client: The LLM client for Claude API calls.
        retriever: The Retriever instance for building context strings.

    Returns:
        dict: State update dictionary with extracted_leads list.
    """
    # Extract the user query and retrieved context from state
    query = state.get("user_query", "")
    retrieved_context = state.get("retrieved_context", [])

    # Log the extraction start
    logger.info("Lead extraction node executing with %d context chunks", len(retrieved_context))

    # Check if we have context to extract from
    if not retrieved_context:
        logger.warning("No retrieved context available for lead extraction")
        return {"extracted_leads": [], "errors": ["No context available for lead extraction."]}

    # Build a combined context string from the retrieved chunks
    context_string = retriever.build_context_string(retrieved_context)

    # Check that the context string is not empty
    if not context_string.strip():
        logger.warning("Built context string is empty, skipping extraction")
        return {"extracted_leads": [], "errors": ["Retrieved context produced empty text."]}

    try:
        # Use the lead extractor tool to extract structured leads
        leads = extract_leads_from_context(
            context=context_string,
            query=query,
            llm_client=llm_client,
        )

        # Add source document info to each extracted lead
        source_docs = _extract_source_documents(retrieved_context)
        for lead in leads:
            lead.source_documents = source_docs

        # Log the extraction result
        logger.info("Extracted %d construction leads from context", len(leads))

        return {"extracted_leads": leads}

    except Exception as error:
        # Log the error and return empty leads with error message
        logger.error("Lead extraction failed: %s", str(error))
        return {"extracted_leads": [], "errors": [f"Lead extraction failed: {str(error)}"]}


def _extract_source_documents(retrieved_context: list[dict]) -> list[str]:
    """Extract unique source document names from the retrieved context.

    Args:
        retrieved_context: List of retrieval result dictionaries with metadata.

    Returns:
        list[str]: Deduplicated list of source document names.
    """
    # Collect unique source document names from metadata
    sources = set()
    for chunk in retrieved_context:
        # Get the source name from the chunk metadata
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "")
        if source:
            sources.add(source)

    # Return as a sorted list for consistent ordering
    return sorted(sources)
