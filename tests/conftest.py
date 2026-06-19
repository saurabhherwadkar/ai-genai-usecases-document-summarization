# Conftest module - shared test fixtures for the entire test suite.
# Provides reusable test data, mock objects, and configuration overrides.

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.lead_schemas import (
    ConstructionLead,
    ContactInfo,
    DocumentChunk,
    ProjectDetails,
    ScoredLead,
)


@pytest.fixture
def sample_project_details():
    """Create a sample ProjectDetails instance for testing.

    Returns:
        ProjectDetails: A fully populated project details model.
    """
    return ProjectDetails(
        project_name="Downtown Office Tower",
        project_type="commercial",
        description="A 20-story mixed-use office tower with ground floor retail",
        location="123 Main Street, Chicago, IL 60601",
        owner="Skyline Development Corp",
        budget="$45,000,000",
        timeline="Q2 2025 - Q4 2027",
        project_phase="bidding",
        scope_of_work="Full construction including foundation, structure, MEP, and finish",
        square_footage="250,000 sq ft",
    )


@pytest.fixture
def sample_contact_info():
    """Create a sample ContactInfo instance for testing.

    Returns:
        ContactInfo: A fully populated contact info model.
    """
    return ContactInfo(
        name="John Smith",
        company="Skyline Development Corp",
        email="jsmith@skylinedev.com",
        phone="(312) 555-0100",
        role="Project Manager",
    )


@pytest.fixture
def sample_construction_lead(sample_project_details, sample_contact_info):
    """Create a sample ConstructionLead instance for testing.

    Args:
        sample_project_details: The project details fixture.
        sample_contact_info: The contact info fixture.

    Returns:
        ConstructionLead: A complete construction lead model.
    """
    return ConstructionLead(
        id="test-lead-001",
        project=sample_project_details,
        contacts=[sample_contact_info],
        source_documents=["downtown_permits.pdf", "bid_notice_2025.docx"],
    )


@pytest.fixture
def sample_scored_lead(sample_construction_lead):
    """Create a sample ScoredLead instance for testing.

    Args:
        sample_construction_lead: The construction lead fixture.

    Returns:
        ScoredLead: A scored lead with quality metrics.
    """
    return ScoredLead(
        lead=sample_construction_lead,
        score=0.85,
        completeness_score=0.9,
        relevance_score=0.8,
        score_breakdown={
            "completeness": 0.9,
            "budget_presence": 1.0,
            "timeline_presence": 1.0,
            "contact_info": 0.8,
            "recency": 0.5,
        },
    )


@pytest.fixture
def sample_document_chunks():
    """Create a list of sample DocumentChunk instances for testing.

    Returns:
        list[DocumentChunk]: List of document chunks with construction content.
    """
    return [
        DocumentChunk(
            chunk_id="chunk-001",
            document_name="permits_2025.pdf",
            content="Building permit #BP-2025-001 issued for Downtown Office Tower at 123 Main Street. "
            "Owner: Skyline Development Corp. Estimated cost: $45,000,000. "
            "Project includes 20-story mixed-use building with retail.",
            chunk_index=0,
            metadata={"source": "permits_2025.pdf", "chunk_index": 0, "total_chunks": 3},
        ),
        DocumentChunk(
            chunk_id="chunk-002",
            document_name="permits_2025.pdf",
            content="Contact: John Smith, Project Manager at Skyline Development Corp. "
            "Phone: (312) 555-0100. Email: jsmith@skylinedev.com. "
            "Construction timeline: Q2 2025 through Q4 2027.",
            chunk_index=1,
            metadata={"source": "permits_2025.pdf", "chunk_index": 1, "total_chunks": 3},
        ),
        DocumentChunk(
            chunk_id="chunk-003",
            document_name="rfp_notice.txt",
            content="Request for Proposals: Highway Bridge Replacement Project. "
            "Location: Interstate 90 at mile marker 42, Springfield, IL. "
            "Budget: $12,500,000. Deadline for submissions: March 15, 2025.",
            chunk_index=0,
            metadata={"source": "rfp_notice.txt", "chunk_index": 0, "total_chunks": 1},
        ),
    ]


@pytest.fixture
def sample_retrieval_results():
    """Create sample retrieval results as returned by the vector store.

    Returns:
        list[dict]: List of retrieval result dictionaries.
    """
    return [
        {
            "id": "chunk-001",
            "content": "Building permit #BP-2025-001 issued for Downtown Office Tower.",
            "metadata": {"source": "permits_2025.pdf", "chunk_index": 0},
            "distance": 0.25,
        },
        {
            "id": "chunk-003",
            "content": "Request for Proposals: Highway Bridge Replacement Project.",
            "metadata": {"source": "rfp_notice.txt", "chunk_index": 0},
            "distance": 0.42,
        },
    ]


@pytest.fixture
def mock_llm_extraction_response():
    """Create a mock Claude response for lead extraction.

    Returns:
        str: JSON string mimicking Claude's lead extraction response.
    """
    return """{
        "leads": [
            {
                "project": {
                    "project_name": "Downtown Office Tower",
                    "project_type": "commercial",
                    "description": "20-story mixed-use office tower",
                    "location": "123 Main Street, Chicago, IL",
                    "owner": "Skyline Development Corp",
                    "budget": "$45,000,000",
                    "timeline": "Q2 2025 - Q4 2027",
                    "project_phase": "bidding",
                    "scope_of_work": "Full construction",
                    "square_footage": "250,000 sq ft"
                },
                "contacts": [
                    {
                        "name": "John Smith",
                        "company": "Skyline Development Corp",
                        "email": "jsmith@skylinedev.com",
                        "phone": "(312) 555-0100",
                        "role": "Project Manager"
                    }
                ]
            }
        ]
    }"""


@pytest.fixture
def tmp_documents_dir(tmp_path):
    """Create a temporary directory with sample document files for testing.

    Args:
        tmp_path: pytest's built-in temporary path fixture.

    Returns:
        Path: Path to the temporary directory containing test documents.
    """
    # Create a sample text file
    txt_file = tmp_path / "sample_project.txt"
    txt_file.write_text(
        "Construction Project: New Community Center\n"
        "Location: 456 Oak Avenue, Springfield, IL\n"
        "Budget: $8,500,000\n"
        "Owner: Springfield City Council\n"
        "Timeline: January 2025 - December 2026\n"
        "Contact: Jane Doe, City Planner\n"
        "Phone: (217) 555-0200\n"
        "Email: jdoe@springfield.gov\n",
        encoding="utf-8",
    )

    # Create another sample text file
    txt_file_2 = tmp_path / "rfp_bridge.txt"
    txt_file_2.write_text(
        "REQUEST FOR PROPOSALS\n"
        "Project: Highway Bridge Replacement\n"
        "Location: Interstate 90, Mile Marker 42\n"
        "Estimated Budget: $12,500,000\n"
        "Submission Deadline: March 15, 2025\n"
        "Contact: Bob Johnson, IDOT\n"
        "Email: bjohnson@idot.gov\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing without API calls.

    Returns:
        MagicMock: A mocked LLMClient instance.
    """
    mock = MagicMock()
    mock.generate_response.return_value = "find_leads"
    return mock
