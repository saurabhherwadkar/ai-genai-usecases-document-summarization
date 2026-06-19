# Lead scoring node module - scores and ranks extracted construction leads.
# Assigns quality scores based on data completeness, budget, timeline, and contacts.

from src.config.settings import get_settings
from src.models.agent_state import LeadsAgentState
from src.models.lead_schemas import ConstructionLead, ScoredLead
from src.utils.logger import get_logger

# Module logger for tracking lead scoring
logger = get_logger(__name__)


def lead_scoring_node(state: LeadsAgentState) -> dict:
    """Score and rank extracted construction leads by quality.

    Evaluates each lead based on configurable scoring weights for
    completeness, budget presence, timeline, contacts, and recency.

    Args:
        state: The current LangGraph agent state with extracted_leads.

    Returns:
        dict: State update dictionary with scored_leads list sorted by score descending.
    """
    # Extract the leads from state
    extracted_leads = state.get("extracted_leads", [])

    # Log the scoring start
    logger.info("Scoring %d extracted leads", len(extracted_leads))

    # Return empty if no leads to score
    if not extracted_leads:
        logger.info("No leads to score")
        return {"scored_leads": []}

    # Load scoring weights from settings
    settings = get_settings()
    weights = settings.lead_extraction.scoring_weights

    # Score each lead individually
    scored_leads = []
    for lead in extracted_leads:
        scored_lead = _score_single_lead(lead, weights)
        scored_leads.append(scored_lead)

    # Sort by overall score descending (highest quality leads first)
    scored_leads.sort(key=lambda sl: sl.score, reverse=True)

    # Log the scoring results
    logger.info(
        "Scored %d leads. Top score: %.2f, Bottom score: %.2f",
        len(scored_leads),
        scored_leads[0].score if scored_leads else 0,
        scored_leads[-1].score if scored_leads else 0,
    )

    return {"scored_leads": scored_leads}


def _score_single_lead(lead: ConstructionLead, weights: dict) -> ScoredLead:
    """Calculate the quality score for a single construction lead.

    Evaluates multiple scoring dimensions and combines them using
    the configured weights.

    Args:
        lead: The construction lead to score.
        weights: Dictionary of scoring weights for each dimension.

    Returns:
        ScoredLead: The lead wrapped with its calculated scores.
    """
    # Calculate individual scoring dimensions
    completeness = _calculate_completeness_score(lead)
    budget_score = _calculate_budget_score(lead)
    timeline_score = _calculate_timeline_score(lead)
    contact_score = _calculate_contact_score(lead)
    recency_score = _calculate_recency_score(lead)

    # Combine scores using configured weights
    overall_score = (
        completeness * weights.get("completeness", 0.3)
        + budget_score * weights.get("budget_presence", 0.2)
        + timeline_score * weights.get("timeline_presence", 0.2)
        + contact_score * weights.get("contact_info", 0.15)
        + recency_score * weights.get("recency", 0.15)
    )

    # Build the score breakdown dictionary
    breakdown = {
        "completeness": completeness,
        "budget_presence": budget_score,
        "timeline_presence": timeline_score,
        "contact_info": contact_score,
        "recency": recency_score,
    }

    # Create and return the ScoredLead instance
    return ScoredLead(
        lead=lead,
        score=round(overall_score, 3),
        completeness_score=round(completeness, 3),
        relevance_score=round(overall_score, 3),
        score_breakdown=breakdown,
    )


def _calculate_completeness_score(lead: ConstructionLead) -> float:
    """Calculate how complete the lead data is based on populated fields.

    Args:
        lead: The construction lead to evaluate.

    Returns:
        float: Completeness score from 0.0 to 1.0.
    """
    # Define the project fields to check for completeness
    project_fields = [
        lead.project.project_name,
        lead.project.project_type,
        lead.project.description,
        lead.project.location,
        lead.project.owner,
        lead.project.budget,
        lead.project.timeline,
        lead.project.project_phase,
        lead.project.scope_of_work,
        lead.project.square_footage,
    ]

    # Count non-empty fields
    filled_count = sum(1 for field in project_fields if field and field.strip())

    # Calculate the ratio of filled fields to total fields
    return filled_count / len(project_fields)


def _calculate_budget_score(lead: ConstructionLead) -> float:
    """Calculate a score based on budget information presence and specificity.

    Args:
        lead: The construction lead to evaluate.

    Returns:
        float: Budget score from 0.0 to 1.0.
    """
    # Check if budget field is populated
    budget = lead.project.budget
    if not budget or not budget.strip():
        return 0.0

    # Higher score if budget contains a number (specific amount)
    if any(char.isdigit() for char in budget):
        return 1.0

    # Lower score if budget is mentioned but not specific (e.g., "TBD", "large")
    return 0.5


def _calculate_timeline_score(lead: ConstructionLead) -> float:
    """Calculate a score based on timeline information presence.

    Args:
        lead: The construction lead to evaluate.

    Returns:
        float: Timeline score from 0.0 to 1.0.
    """
    # Check if timeline field is populated
    timeline = lead.project.timeline
    if not timeline or not timeline.strip():
        return 0.0

    # Higher score if timeline contains dates or year references
    if any(char.isdigit() for char in timeline):
        return 1.0

    # Lower score for vague timeline descriptions
    return 0.5


def _calculate_contact_score(lead: ConstructionLead) -> float:
    """Calculate a score based on contact information availability.

    Args:
        lead: The construction lead to evaluate.

    Returns:
        float: Contact score from 0.0 to 1.0.
    """
    # No contacts means zero score
    if not lead.contacts:
        return 0.0

    # Score based on contact detail completeness
    total_score = 0.0
    for contact in lead.contacts:
        # Each populated contact field adds to the score
        contact_fields = [contact.name, contact.company, contact.email, contact.phone, contact.role]
        filled = sum(1 for field in contact_fields if field and field.strip())
        total_score += filled / len(contact_fields)

    # Average across all contacts, capped at 1.0
    return min(1.0, total_score / len(lead.contacts))


def _calculate_recency_score(lead: ConstructionLead) -> float:
    """Calculate a score based on lead recency.

    Currently returns a default score since we don't track document dates.
    Can be enhanced later with document metadata dates.

    Args:
        lead: The construction lead to evaluate.

    Returns:
        float: Recency score from 0.0 to 1.0.
    """
    # Default recency score (all leads treated equally without date metadata)
    # Future enhancement: compare extraction date with document dates
    return 0.5
