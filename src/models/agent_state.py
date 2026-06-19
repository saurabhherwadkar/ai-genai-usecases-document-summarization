# Agent state module - defines the LangGraph TypedDict state schema for the leads agent.
# The state flows through all nodes in the graph and accumulates results.

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from src.models.lead_schemas import ConstructionLead, ScoredLead


def _merge_list(existing: list, new: list) -> list:
    """Reducer function that merges two lists by concatenation.

    Used as an Annotated reducer for LangGraph state fields that
    accumulate results across multiple node executions.

    Args:
        existing: The current list in the state.
        new: The new items to append to the existing list.

    Returns:
        list: The combined list with new items appended.
    """
    return existing + new


class LeadsAgentState(TypedDict):
    """Central state schema for the construction leads LangGraph agent.

    This TypedDict defines all fields that flow through the agent graph.
    Fields with Annotated reducers accumulate values across node executions.
    Fields without reducers are overwritten by the latest node output.
    """

    # The original user query text
    user_query: str

    # Classified intent of the query: "find_leads", "summarize", or "general"
    query_intent: str

    # Retrieved document chunks from the vector store (accumulates across retrievals)
    retrieved_context: Annotated[list[dict], _merge_list]

    # Extracted construction leads from the retrieved context (accumulates)
    extracted_leads: Annotated[list[ConstructionLead], _merge_list]

    # Scored and ranked leads after quality assessment (overwritten by scoring node)
    scored_leads: list[ScoredLead]

    # Final natural language response text for the user
    response: str

    # Conversation message history managed by LangGraph's message reducer
    messages: Annotated[list, add_messages]

    # Error messages accumulated during graph execution
    errors: Annotated[list[str], _merge_list]
