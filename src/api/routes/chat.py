# Chat route module - provides the chat endpoint for agent interactions.
# Handles user queries by invoking the LangGraph agent graph.

from fastapi import APIRouter

from src.agent.graph import build_leads_graph
from src.agent.llm_client import LLMClient
from src.api.models.schemas import ChatRequest, ChatResponse
from src.models.agent_state import LeadsAgentState
from src.rag.retriever import Retriever
from src.services.lead_service import LeadService
from src.utils.input_sanitizer import InputSanitizer
from src.utils.logger import get_logger

# Module logger for tracking chat interactions
logger = get_logger(__name__)

# Create the router for chat endpoints
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user chat query through the LangGraph agent.

    Sanitizes the query, invokes the agent graph, and returns the
    response with any extracted leads and source references.

    Args:
        request: The chat request containing the user's query.

    Returns:
        ChatResponse: The agent's response with leads and sources.
    """
    # Sanitize the user query for security
    sanitized_query = InputSanitizer.sanitize_query(request.query)

    # Log the incoming chat request
    logger.info("Chat request received: '%s'", sanitized_query[:100])

    # Initialize the LLM client and retriever
    llm_client = LLMClient()
    retriever = Retriever()

    # Build the LangGraph agent
    graph = build_leads_graph(llm_client=llm_client, retriever=retriever)

    # Prepare the initial state for the graph
    initial_state: LeadsAgentState = {
        "user_query": sanitized_query,
        "query_intent": "",
        "retrieved_context": [],
        "extracted_leads": [],
        "scored_leads": [],
        "response": "",
        "messages": [],
        "errors": [],
    }

    # Invoke the graph and get the final state
    final_state = graph.invoke(initial_state)

    # Extract results from the final state
    response_text = final_state.get("response", "I could not generate a response. Please try again.")
    scored_leads = final_state.get("scored_leads", [])
    query_intent = final_state.get("query_intent", "")
    retrieved_context = final_state.get("retrieved_context", [])

    # Extract source documents from retrieved context
    sources = _extract_sources(retrieved_context)

    # Store extracted leads if any were found
    if scored_leads:
        lead_service = LeadService()
        lead_service.add_leads(scored_leads)
        logger.info("Stored %d leads from chat interaction", len(scored_leads))

    # Log the chat response
    logger.info("Chat response generated: intent='%s', leads=%d", query_intent, len(scored_leads))

    # Return the chat response
    return ChatResponse(
        response=response_text,
        leads=scored_leads,
        sources=sources,
        query_intent=query_intent,
    )


def _extract_sources(retrieved_context: list[dict]) -> list[str]:
    """Extract unique source document names from retrieved context.

    Args:
        retrieved_context: List of retrieval result dictionaries.

    Returns:
        list[str]: Deduplicated sorted list of source document names.
    """
    # Collect unique source names from chunk metadata
    sources = set()
    for chunk in retrieved_context:
        source = chunk.get("metadata", {}).get("source", "")
        if source:
            sources.add(source)

    # Return as sorted list
    return sorted(sources)
