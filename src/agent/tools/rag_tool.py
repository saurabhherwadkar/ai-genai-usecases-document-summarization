# RAG tool module - provides document retrieval functionality for the agent.
# Wraps the Retriever to perform vector search and return relevant context.

from src.rag.retriever import Retriever
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

# Module logger for tracking RAG tool usage
logger = get_logger(__name__)


def retrieve_documents(query: str, retriever: Retriever, top_k: int | None = None) -> list[dict]:
    """Retrieve relevant document chunks from the vector store for a query.

    This tool function wraps the Retriever to provide a simple interface
    for the agent nodes to fetch relevant context.

    Args:
        query: The natural language search query.
        retriever: The Retriever instance to use for searching.
        top_k: Optional override for the number of results to return.

    Returns:
        list[dict]: List of retrieved document chunks with content, metadata, and distance.
    """
    # Log the tool invocation
    logger.info("RAG tool invoked for query: '%s'", query[:100])

    try:
        # Perform the retrieval using the provided retriever instance
        results = retriever.retrieve(query=query, top_k=top_k)

        # Log the number of results
        logger.info("RAG tool returned %d relevant chunks", len(results))

        return results

    except RetrievalError as error:
        # Log the retrieval error and return empty results
        logger.error("RAG tool retrieval failed: %s", error.message)
        return []
