# Retrieval node module - performs RAG retrieval from the vector store.
# Fetches relevant document chunks based on the user query.

from src.agent.tools.rag_tool import retrieve_documents
from src.models.agent_state import LeadsAgentState
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

# Module logger for tracking retrieval operations
logger = get_logger(__name__)


def retrieval_node(state: LeadsAgentState, retriever: Retriever) -> dict:
    """Retrieve relevant document chunks from the vector store.

    Uses the RAG tool to find document chunks semantically similar
    to the user's query and adds them to the agent state.

    Args:
        state: The current LangGraph agent state containing the user query.
        retriever: The Retriever instance for vector store search.

    Returns:
        dict: State update dictionary with retrieved_context list.
    """
    # Extract the user query from state
    query = state.get("user_query", "")

    # Log the retrieval start
    logger.info("Retrieval node executing for query: '%s'", query[:100])

    # Use the RAG tool to retrieve relevant documents
    results = retrieve_documents(query=query, retriever=retriever)

    # Check if any results were returned
    if not results:
        logger.warning("No relevant documents found for query: '%s'", query[:100])
        return {"retrieved_context": [], "errors": ["No relevant documents found in the vector store."]}

    # Log the successful retrieval
    logger.info("Retrieved %d relevant document chunks", len(results))

    # Return the state update with the retrieved context
    return {"retrieved_context": results}
