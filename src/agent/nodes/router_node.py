# Router node module - classifies the user's query intent for graph routing.
# Determines whether to find leads, summarize documents, or answer generally.

from src.agent.llm_client import LLMClient
from src.agent.prompt_templates import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from src.models.agent_state import LeadsAgentState
from src.utils.logger import get_logger

# Module logger for tracking routing decisions
logger = get_logger(__name__)

# Valid intent classifications that the router can return
VALID_INTENTS = {"find_leads", "summarize", "general"}

# Keywords that strongly indicate lead-finding intent
LEAD_KEYWORDS = [
    "lead",
    "leads",
    "find",
    "search",
    "discover",
    "identify",
    "project",
    "projects",
    "bid",
    "bids",
    "rfp",
    "permit",
    "opportunity",
    "opportunities",
    "construction",
    "building",
    "development",
    "contractor",
    "upcoming",
]

# Keywords that strongly indicate summarization intent
SUMMARIZE_KEYWORDS = [
    "summarize",
    "summary",
    "explain",
    "describe",
    "tell me about",
    "what does",
    "overview",
    "detail",
    "details",
]


def router_node(state: LeadsAgentState, llm_client: LLMClient) -> dict:
    """Classify the user query intent and update the agent state.

    Uses keyword matching first for efficiency, falling back to Claude
    for ambiguous queries that keywords cannot confidently classify.

    Args:
        state: The current LangGraph agent state containing the user query.
        llm_client: The LLM client instance for Claude API calls.

    Returns:
        dict: State update dictionary with the classified query_intent.
    """
    # Extract the user query from state
    query = state.get("user_query", "")

    # Log the routing attempt
    logger.info("Router classifying query: '%s'", query[:100])

    # Attempt fast keyword-based classification first
    intent = _classify_by_keywords(query)

    # If keywords are inconclusive, use Claude for classification
    if intent is None:
        intent = _classify_by_llm(query, llm_client)

    # Validate that the intent is one of the expected values
    if intent not in VALID_INTENTS:
        logger.warning("Unexpected intent '%s', defaulting to 'general'", intent)
        intent = "general"

    # Log the routing decision
    logger.info("Query routed to intent: '%s'", intent)

    # Return the state update with the classified intent
    return {"query_intent": intent}


def _classify_by_keywords(query: str) -> str | None:
    """Attempt to classify query intent using keyword matching.

    Counts matches against lead-finding and summarization keyword lists.
    Returns a classification only if one category has a clear majority.

    Args:
        query: The user query string to classify.

    Returns:
        str | None: The classified intent, or None if keywords are inconclusive.
    """
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()

    # Count matches against lead-finding keywords
    lead_score = sum(1 for keyword in LEAD_KEYWORDS if keyword in query_lower)

    # Count matches against summarization keywords
    summarize_score = sum(1 for keyword in SUMMARIZE_KEYWORDS if keyword in query_lower)

    # Require a minimum score and clear winner for keyword classification
    if lead_score >= 2 and lead_score > summarize_score:
        logger.debug("Keyword classification: find_leads (score=%d)", lead_score)
        return "find_leads"

    if summarize_score >= 2 and summarize_score > lead_score:
        logger.debug("Keyword classification: summarize (score=%d)", summarize_score)
        return "summarize"

    # Keywords are inconclusive, return None to trigger LLM classification
    logger.debug("Keyword classification inconclusive: lead=%d, summarize=%d", lead_score, summarize_score)
    return None


def _classify_by_llm(query: str, llm_client: LLMClient) -> str:
    """Classify query intent using Claude for ambiguous queries.

    Falls back to "general" if the LLM call fails or returns unexpected output.

    Args:
        query: The user query string to classify.
        llm_client: The LLM client for Claude API calls.

    Returns:
        str: The classified intent string.
    """
    try:
        # Format the classification prompt
        user_message = ROUTER_USER_TEMPLATE.format(query=query)

        # Call Claude for classification
        response = llm_client.generate_response(
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_message=user_message,
        )

        # Clean the response and extract the intent
        intent = response.strip().lower()

        # Validate the response is a known intent
        if intent in VALID_INTENTS:
            logger.debug("LLM classification: '%s'", intent)
            return intent

        # If response doesn't match expected values, default to general
        logger.warning("LLM returned unexpected intent: '%s', defaulting to general", intent)
        return "general"

    except Exception as error:
        # Log the error and default to general intent
        logger.error("LLM classification failed: %s, defaulting to general", str(error))
        return "general"
