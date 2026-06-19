# Graph module - defines and builds the LangGraph StateGraph for the construction leads agent.
# Assembles nodes, edges, and conditional routing into a compiled graph.

from functools import partial

from langgraph.graph import END, StateGraph

from src.agent.llm_client import LLMClient
from src.agent.nodes.lead_extraction_node import lead_extraction_node
from src.agent.nodes.lead_scoring_node import lead_scoring_node
from src.agent.nodes.retrieval_node import retrieval_node
from src.agent.nodes.router_node import router_node
from src.agent.nodes.summarization_node import summarization_node
from src.config.settings import get_settings
from src.models.agent_state import LeadsAgentState
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

# Module logger for tracking graph construction and execution
logger = get_logger(__name__)


def build_leads_graph(llm_client: LLMClient, retriever: Retriever):
    """Build and compile the LangGraph StateGraph for construction lead extraction.

    Creates a graph with conditional routing based on query intent:
    - find_leads: router -> retrieval -> extraction -> scoring -> summarization
    - summarize: router -> retrieval -> summarization
    - general: router -> summarization

    Args:
        llm_client: The LLM client instance for Claude API calls.
        retriever: The Retriever instance for vector store queries.

    Returns:
        CompiledGraph: The compiled LangGraph ready for invocation.
    """
    # Load agent configuration
    settings = get_settings()

    # Log the graph construction
    logger.info("Building leads agent graph")

    # Create the StateGraph with the LeadsAgentState schema
    graph = StateGraph(LeadsAgentState)

    # Create partial functions that bind dependencies to each node
    bound_router = partial(router_node, llm_client=llm_client)
    bound_retrieval = partial(retrieval_node, retriever=retriever)
    bound_extraction = partial(lead_extraction_node, llm_client=llm_client, retriever=retriever)
    bound_summarization = partial(summarization_node, llm_client=llm_client, retriever=retriever)

    # Add nodes to the graph
    graph.add_node("router", bound_router)
    graph.add_node("retrieval", bound_retrieval)
    graph.add_node("lead_extraction", bound_extraction)
    graph.add_node("lead_scoring", lead_scoring_node)
    graph.add_node("summarization", bound_summarization)

    # Set the entry point to the router node
    graph.set_entry_point("router")

    # Add conditional routing from the router based on intent
    graph.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "retrieval": "retrieval",
            "summarization": "summarization",
        },
    )

    # Add conditional routing from retrieval based on intent
    graph.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {
            "lead_extraction": "lead_extraction",
            "summarization": "summarization",
        },
    )

    # Add edge from lead extraction to scoring
    graph.add_edge("lead_extraction", "lead_scoring")

    # Add edge from scoring to summarization
    graph.add_edge("lead_scoring", "summarization")

    # Add edge from summarization to END
    graph.add_edge("summarization", END)

    # Compile the graph with the configured recursion limit
    compiled_graph = graph.compile()

    # Log successful compilation
    logger.info("Leads agent graph compiled successfully")

    return compiled_graph


def _route_by_intent(state: LeadsAgentState) -> str:
    """Route from the router node based on the classified query intent.

    Args:
        state: The current agent state with the query_intent field.

    Returns:
        str: The next node name to execute ("retrieval" or "summarization").
    """
    # Get the classified intent from state
    intent = state.get("query_intent", "general")

    # Route find_leads and summarize intents to retrieval first
    if intent in ("find_leads", "summarize"):
        logger.debug("Routing to retrieval for intent: %s", intent)
        return "retrieval"

    # Route general queries directly to summarization
    logger.debug("Routing to summarization for intent: %s", intent)
    return "summarization"


def _route_after_retrieval(state: LeadsAgentState) -> str:
    """Route from the retrieval node based on intent (lead extraction or summarization).

    Args:
        state: The current agent state with the query_intent field.

    Returns:
        str: The next node name ("lead_extraction" or "summarization").
    """
    # Get the intent to determine the next step after retrieval
    intent = state.get("query_intent", "general")

    # Only route to lead extraction for find_leads intent
    if intent == "find_leads":
        logger.debug("Routing to lead_extraction after retrieval")
        return "lead_extraction"

    # For summarize intent, go directly to summarization
    logger.debug("Routing to summarization after retrieval")
    return "summarization"
