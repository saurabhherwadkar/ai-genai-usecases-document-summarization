# Summarization node module - generates the final user-facing response.
# Produces natural language summaries of leads, context, or general answers.

from src.agent.llm_client import LLMClient
from src.agent.tools.summarizer_tool import generate_summary
from src.models.agent_state import LeadsAgentState
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

# Module logger for tracking summarization
logger = get_logger(__name__)


def summarization_node(state: LeadsAgentState, llm_client: LLMClient, retriever: Retriever) -> dict:
    """Generate the final natural language response for the user.

    Selects the appropriate summarization approach based on the query intent
    and available data in the state (leads, context, or neither).

    Args:
        state: The current LangGraph agent state with query, intent, leads, and context.
        llm_client: The LLM client for Claude API calls.
        retriever: The Retriever instance for building context strings.

    Returns:
        dict: State update dictionary with the response text.
    """
    # Extract relevant state values
    query = state.get("user_query", "")
    intent = state.get("query_intent", "general")
    scored_leads = state.get("scored_leads", [])
    retrieved_context = state.get("retrieved_context", [])

    # Log the summarization start
    logger.info(
        "Summarization node: intent='%s', leads=%d, context_chunks=%d",
        intent,
        len(scored_leads),
        len(retrieved_context),
    )

    # Build context string if we have retrieved chunks
    context_string = ""
    if retrieved_context:
        context_string = retriever.build_context_string(retrieved_context)

    # Generate the summary using the summarizer tool
    response = generate_summary(
        query=query,
        llm_client=llm_client,
        scored_leads=scored_leads if scored_leads else None,
        context=context_string if context_string else None,
        intent=intent,
    )

    # Log the response generation
    logger.info("Summarization complete: %d character response", len(response))

    # Return the state update with the generated response
    return {"response": response}
