# Test lead scoring node module - unit tests for the scoring algorithm.
# Tests individual scoring dimensions and overall score calculation.

import pytest

from src.agent.nodes.lead_scoring_node import (
    lead_scoring_node,
    _calculate_budget_score,
    _calculate_completeness_score,
    _calculate_contact_score,
    _calculate_timeline_score,
)
from src.models.lead_schemas import ConstructionLead, ContactInfo, ProjectDetails


class TestLeadScoringNode:
    """Tests for the lead_scoring_node function."""

    def test_scoring_node_returns_scored_leads(self, sample_construction_lead):
        """Test that the scoring node produces ScoredLead objects."""
        state = {
            "user_query": "find leads",
            "query_intent": "find_leads",
            "retrieved_context": [],
            "extracted_leads": [sample_construction_lead],
            "scored_leads": [],
            "response": "",
            "messages": [],
            "errors": [],
        }

        result = lead_scoring_node(state)

        # Should produce scored leads
        assert "scored_leads" in result
        assert len(result["scored_leads"]) == 1
        assert result["scored_leads"][0].score > 0

    def test_scoring_node_empty_leads_returns_empty(self):
        """Test that empty input produces empty scored leads."""
        state = {
            "user_query": "find leads",
            "query_intent": "find_leads",
            "retrieved_context": [],
            "extracted_leads": [],
            "scored_leads": [],
            "response": "",
            "messages": [],
            "errors": [],
        }

        result = lead_scoring_node(state)

        assert result["scored_leads"] == []

    def test_scoring_node_sorts_by_score_descending(self):
        """Test that scored leads are sorted highest to lowest score."""
        # Create two leads with different completeness
        complete_lead = ConstructionLead(
            project=ProjectDetails(
                project_name="Big Project",
                project_type="commercial",
                location="Chicago, IL",
                budget="$50M",
                timeline="2025-2027",
                owner="DevCorp",
                project_phase="bidding",
                scope_of_work="Full build",
                description="Large office",
                square_footage="100,000 sq ft",
            ),
            contacts=[ContactInfo(name="John", company="DevCorp", email="j@dev.com", phone="555-0001", role="PM")],
        )

        sparse_lead = ConstructionLead(
            project=ProjectDetails(project_name="Small Project"),
        )

        state = {
            "user_query": "find leads",
            "query_intent": "find_leads",
            "retrieved_context": [],
            "extracted_leads": [sparse_lead, complete_lead],
            "scored_leads": [],
            "response": "",
            "messages": [],
            "errors": [],
        }

        result = lead_scoring_node(state)

        # Complete lead should be first (higher score)
        assert result["scored_leads"][0].lead.project.project_name == "Big Project"
        assert result["scored_leads"][0].score > result["scored_leads"][1].score


class TestScoringFunctions:
    """Tests for individual scoring dimension functions."""

    def test_budget_score_with_numeric_budget(self):
        """Test that a numeric budget gets full score."""
        lead = ConstructionLead(project=ProjectDetails(project_name="Test", budget="$45,000,000"))
        assert _calculate_budget_score(lead) == 1.0

    def test_budget_score_with_vague_budget(self):
        """Test that a vague budget gets partial score."""
        lead = ConstructionLead(project=ProjectDetails(project_name="Test", budget="TBD"))
        assert _calculate_budget_score(lead) == 0.5

    def test_budget_score_with_no_budget(self):
        """Test that missing budget gets zero score."""
        lead = ConstructionLead(project=ProjectDetails(project_name="Test", budget=""))
        assert _calculate_budget_score(lead) == 0.0

    def test_timeline_score_with_dates(self):
        """Test that a timeline with dates gets full score."""
        lead = ConstructionLead(project=ProjectDetails(project_name="Test", timeline="Q2 2025 - Q4 2027"))
        assert _calculate_timeline_score(lead) == 1.0

    def test_timeline_score_empty(self):
        """Test that missing timeline gets zero score."""
        lead = ConstructionLead(project=ProjectDetails(project_name="Test", timeline=""))
        assert _calculate_timeline_score(lead) == 0.0

    def test_contact_score_with_complete_contacts(self):
        """Test that complete contacts get high score."""
        lead = ConstructionLead(
            project=ProjectDetails(project_name="Test"),
            contacts=[ContactInfo(name="John", company="Corp", email="j@c.com", phone="555", role="PM")],
        )
        assert _calculate_contact_score(lead) == 1.0

    def test_contact_score_with_no_contacts(self):
        """Test that no contacts get zero score."""
        lead = ConstructionLead(project=ProjectDetails(project_name="Test"), contacts=[])
        assert _calculate_contact_score(lead) == 0.0

    def test_completeness_score_fully_populated(self):
        """Test that a fully populated lead gets high completeness score."""
        lead = ConstructionLead(
            project=ProjectDetails(
                project_name="Test",
                project_type="commercial",
                description="Desc",
                location="Chicago",
                owner="Corp",
                budget="$5M",
                timeline="2025",
                project_phase="bidding",
                scope_of_work="Full",
                square_footage="10000",
            )
        )
        assert _calculate_completeness_score(lead) == 1.0

    def test_completeness_score_minimal_data(self):
        """Test that a sparse lead gets low completeness score."""
        lead = ConstructionLead(project=ProjectDetails(project_name="Test"))
        # Only 1 out of 10 fields populated
        assert _calculate_completeness_score(lead) == 0.1
