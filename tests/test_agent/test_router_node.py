# Test router node module - unit tests for query intent classification.
# Tests keyword-based and LLM-based routing logic.

import pytest
from unittest.mock import MagicMock

from src.agent.nodes.router_node import router_node, _classify_by_keywords


class TestRouterNode:
    """Tests for the router_node function and classification logic."""

    def test_classify_find_leads_by_keywords(self):
        """Test that lead-finding queries are classified by keywords."""
        # Queries with strong lead-finding signals
        assert _classify_by_keywords("Find construction projects in Chicago") == "find_leads"
        assert _classify_by_keywords("Search for building leads and bids") == "find_leads"
        assert _classify_by_keywords("Show me upcoming construction opportunities") == "find_leads"

    def test_classify_summarize_by_keywords(self):
        """Test that summarization queries are classified by keywords."""
        assert _classify_by_keywords("Summarize the project details") == "summarize"
        assert _classify_by_keywords("Give me a summary overview of the documents") == "summarize"

    def test_classify_ambiguous_returns_none(self):
        """Test that ambiguous queries return None for LLM fallback."""
        # Queries without strong keyword signals
        assert _classify_by_keywords("Hello") is None
        assert _classify_by_keywords("What can you do?") is None
        assert _classify_by_keywords("Thanks") is None

    def test_router_node_returns_valid_intent(self, mock_llm_client):
        """Test that router_node returns a valid intent in the state update."""
        # Set up state with a clear lead-finding query
        state = {
            "user_query": "Find all construction leads in the downtown area",
            "query_intent": "",
            "retrieved_context": [],
            "extracted_leads": [],
            "scored_leads": [],
            "response": "",
            "messages": [],
            "errors": [],
        }

        # Execute the router node
        result = router_node(state, mock_llm_client)

        # Verify a valid intent is returned
        assert "query_intent" in result
        assert result["query_intent"] in {"find_leads", "summarize", "general"}

    def test_router_node_classifies_lead_query(self, mock_llm_client):
        """Test that router correctly classifies a lead-finding query."""
        state = {
            "user_query": "Search for construction projects and bids near downtown",
            "query_intent": "",
            "retrieved_context": [],
            "extracted_leads": [],
            "scored_leads": [],
            "response": "",
            "messages": [],
            "errors": [],
        }

        result = router_node(state, mock_llm_client)

        # Strong keyword match should classify as find_leads
        assert result["query_intent"] == "find_leads"

    def test_router_node_falls_back_to_general_on_llm_error(self):
        """Test that router defaults to 'general' when LLM fails."""
        # Create a mock that raises an exception
        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = Exception("API error")

        state = {
            "user_query": "Something ambiguous",
            "query_intent": "",
            "retrieved_context": [],
            "extracted_leads": [],
            "scored_leads": [],
            "response": "",
            "messages": [],
            "errors": [],
        }

        result = router_node(state, mock_llm)

        # Should fall back to general
        assert result["query_intent"] == "general"
