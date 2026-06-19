# Summarizer tool module - generates natural language summaries using Claude.
# Provides response generation for different query intents (leads, context, general).

from src.agent.llm_client import LLMClient
from src.agent.prompt_templates import (
    SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARIZATION_WITH_LEADS_TEMPLATE,
    SUMMARIZATION_GENERAL_TEMPLATE,
    SUMMARIZATION_NO_RESULTS_TEMPLATE,
    SUMMARIZATION_GENERAL_QUERY_TEMPLATE,
)
from src.models.lead_schemas import ScoredLead
from src.utils.exceptions import LLMError
from src.utils.logger import get_logger

# Module logger for tracking summarization operations
logger = get_logger(__name__)


def generate_summary(
    query: str,
    llm_client: LLMClient,
    scored_leads: list[ScoredLead] | None = None,
    context: str | None = None,
    intent: str = "general",
) -> str:
    """Generate a natural language summary response for the user.

    Selects the appropriate prompt template based on the query intent
    and available data (leads, context, or neither).

    Args:
        query: The original user query.
        llm_client: The LLM client for Claude API calls.
        scored_leads: Optional list of scored leads to summarize.
        context: Optional document context for summarization.
        intent: The classified query intent ("find_leads", "summarize", "general").

    Returns:
        str: The generated summary response text.
    """
    # Log the summarization request
    logger.info("Generating summary for intent='%s', query='%s'", intent, query[:100])

    try:
        # Select the appropriate user message template based on available data
        user_message = _build_user_message(query, scored_leads, context, intent)

        # Generate the response using Claude
        response = llm_client.generate_response(
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            user_message=user_message,
        )

        # Log successful generation
        logger.info("Summary generated: %d characters", len(response))

        return response

    except LLMError:
        # Re-raise LLM errors for upstream handling
        raise
    except Exception as error:
        # Log the error and return a fallback message
        logger.error("Summary generation failed: %s", str(error))
        return "I encountered an error while generating a response. Please try again."


def _build_user_message(
    query: str,
    scored_leads: list[ScoredLead] | None,
    context: str | None,
    intent: str,
) -> str:
    """Build the appropriate user message template based on available data.

    Args:
        query: The original user query.
        scored_leads: Optional list of scored leads.
        context: Optional document context string.
        intent: The query intent classification.

    Returns:
        str: The formatted user message for Claude.
    """
    # If we have scored leads, use the leads summary template
    if scored_leads and len(scored_leads) > 0:
        leads_summary = _format_leads_for_prompt(scored_leads)
        return SUMMARIZATION_WITH_LEADS_TEMPLATE.format(query=query, leads_summary=leads_summary)

    # If we have context but no leads, use the general context template
    if context and context.strip():
        return SUMMARIZATION_GENERAL_TEMPLATE.format(query=query, context=context)

    # If intent is general (no retrieval needed), use the general query template
    if intent == "general":
        return SUMMARIZATION_GENERAL_QUERY_TEMPLATE.format(query=query)

    # No leads and no context found for a lead-finding or summarize query
    return SUMMARIZATION_NO_RESULTS_TEMPLATE.format(query=query)


def _format_leads_for_prompt(scored_leads: list[ScoredLead]) -> str:
    """Format scored leads into a readable summary string for the LLM prompt.

    Args:
        scored_leads: List of ScoredLead objects to format.

    Returns:
        str: Formatted string representation of the leads.
    """
    # Build a formatted summary for each lead
    lead_summaries = []
    for index, scored_lead in enumerate(scored_leads, start=1):
        # Extract the project details for readability
        project = scored_lead.lead.project
        # Build the lead summary with available fields
        summary_parts = [f"Lead {index} (Score: {scored_lead.score:.2f}):"]
        summary_parts.append(f"  Project: {project.project_name}")

        # Add non-empty fields to the summary
        if project.project_type:
            summary_parts.append(f"  Type: {project.project_type}")
        if project.location:
            summary_parts.append(f"  Location: {project.location}")
        if project.owner:
            summary_parts.append(f"  Owner: {project.owner}")
        if project.budget:
            summary_parts.append(f"  Budget: {project.budget}")
        if project.timeline:
            summary_parts.append(f"  Timeline: {project.timeline}")
        if project.project_phase:
            summary_parts.append(f"  Phase: {project.project_phase}")
        if project.scope_of_work:
            summary_parts.append(f"  Scope: {project.scope_of_work}")

        # Add contact information if available
        if scored_lead.lead.contacts:
            for contact in scored_lead.lead.contacts:
                contact_str = f"  Contact: {contact.name}"
                if contact.company:
                    contact_str += f" ({contact.company})"
                if contact.role:
                    contact_str += f" - {contact.role}"
                summary_parts.append(contact_str)

        # Add source documents
        if scored_lead.lead.source_documents:
            summary_parts.append(f"  Sources: {', '.join(scored_lead.lead.source_documents)}")

        # Join all parts for this lead
        lead_summaries.append("\n".join(summary_parts))

    # Join all lead summaries with blank line separators
    return "\n\n".join(lead_summaries)
