# Streamlit application module - provides the chat UI and leads dashboard.
# Communicates with the FastAPI backend for agent interactions and document ingestion.

import os

import requests
import streamlit as st

# API base URL from environment or default
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def main():
    """Main entry point for the Streamlit chat application.

    Sets up the page configuration, sidebar, and chat interface.
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Construction Leads Finder",
        page_icon="🏗️",
        layout="wide",
    )

    # Display the application header
    st.title("Construction Leads Finder")
    st.markdown("*Find construction leads from ingested documents using RAG + AI*")

    # Set up the sidebar with document management and leads view
    _render_sidebar()

    # Set up the main chat interface
    _render_chat_interface()


def _render_sidebar():
    """Render the sidebar with document upload, ingestion, and leads dashboard."""
    with st.sidebar:
        # Document ingestion section
        st.header("Document Management")

        # File upload widget
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "docx", "txt", "xlsx", "xls"],
            help="Upload construction documents for analysis",
        )

        # Handle file upload
        if uploaded_file is not None:
            if st.button("Ingest Document"):
                _handle_file_upload(uploaded_file)

        # Directory ingestion
        st.divider()
        directory_path = st.text_input(
            "Directory Path",
            placeholder="./data/sample_documents",
            help="Path to a directory containing documents to ingest",
        )

        if st.button("Ingest Directory") and directory_path:
            _handle_directory_ingestion(directory_path)

        # Leads dashboard section
        st.divider()
        st.header("Leads Dashboard")

        # Show leads count and list
        if st.button("Refresh Leads"):
            _load_leads_dashboard()

        # Display leads if they exist in session state
        if "leads_data" in st.session_state and st.session_state.leads_data:
            _render_leads_list(st.session_state.leads_data)

        # Health check display
        st.divider()
        if st.button("Check Health"):
            _check_health()


def _render_chat_interface():
    """Render the main chat interface with message history and input."""
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display leads if present in assistant messages
            if message.get("leads"):
                _render_chat_leads(message["leads"])

    # Chat input field
    if prompt := st.chat_input("Ask about construction leads..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send query to the API and display response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                response = _send_chat_query(prompt)

            if response:
                # Display the response text
                st.markdown(response.get("response", "No response generated."))

                # Display leads if found
                leads = response.get("leads", [])
                if leads:
                    _render_chat_leads(leads)

                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.get("response", ""),
                    "leads": leads,
                })
            else:
                # Display error message
                error_msg = "Failed to get a response. Please check the API connection."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


def _send_chat_query(query: str) -> dict | None:
    """Send a chat query to the FastAPI backend.

    Args:
        query: The user's natural language query.

    Returns:
        dict | None: The API response data, or None if the request fails.
    """
    try:
        # Send POST request to the chat endpoint
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"query": query},
            timeout=60,
        )

        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Is the backend running?")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The query may be too complex.")
        return None
    except Exception as error:
        st.error(f"Unexpected error: {str(error)}")
        return None


def _handle_file_upload(uploaded_file):
    """Handle file upload and ingestion via the API.

    Args:
        uploaded_file: The Streamlit UploadedFile object.
    """
    try:
        # Send the file to the upload endpoint
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/api/ingest/upload", files=files, timeout=120)

        # Check for success
        if response.status_code == 200:
            result = response.json()
            st.success(
                f"Ingested '{uploaded_file.name}': "
                f"{result['documents_processed']} doc(s), "
                f"{result['chunks_created']} chunks created"
            )
        else:
            st.error(f"Ingestion failed: {response.text}")

    except Exception as error:
        st.error(f"Upload error: {str(error)}")


def _handle_directory_ingestion(directory_path: str):
    """Handle directory-based document ingestion.

    Args:
        directory_path: Path to the directory to ingest.
    """
    try:
        # Send ingestion request to the API
        response = requests.post(
            f"{API_BASE_URL}/api/ingest",
            json={"directory_path": directory_path},
            timeout=300,
        )

        # Check for success
        if response.status_code == 200:
            result = response.json()
            st.success(
                f"Directory ingested: "
                f"{result['documents_processed']} doc(s), "
                f"{result['chunks_created']} chunks created"
            )
        else:
            st.error(f"Ingestion failed: {response.text}")

    except Exception as error:
        st.error(f"Ingestion error: {str(error)}")


def _load_leads_dashboard():
    """Load and display leads from the API in the sidebar."""
    try:
        # Fetch leads from the API
        response = requests.get(f"{API_BASE_URL}/api/leads", params={"limit": 20}, timeout=10)

        if response.status_code == 200:
            data = response.json()
            st.session_state.leads_data = data.get("leads", [])
            st.info(f"Total leads: {data.get('total', 0)}")
        else:
            st.error("Failed to load leads")

    except Exception as error:
        st.error(f"Error loading leads: {str(error)}")


def _render_leads_list(leads: list):
    """Render a compact list of leads in the sidebar.

    Args:
        leads: List of scored lead dictionaries.
    """
    for lead_data in leads[:10]:
        # Extract project info from the lead
        lead = lead_data.get("lead", {})
        project = lead.get("project", {})
        score = lead_data.get("score", 0)

        # Display as an expander with project name
        project_name = project.get("project_name", "Unknown Project")
        with st.expander(f"{'🟢' if score > 0.7 else '🟡' if score > 0.4 else '🔴'} {project_name} ({score:.0%})"):
            if project.get("location"):
                st.write(f"**Location:** {project['location']}")
            if project.get("budget"):
                st.write(f"**Budget:** {project['budget']}")
            if project.get("project_phase"):
                st.write(f"**Phase:** {project['project_phase']}")
            if project.get("owner"):
                st.write(f"**Owner:** {project['owner']}")


def _render_chat_leads(leads: list):
    """Render leads within the chat message area.

    Args:
        leads: List of scored lead dictionaries from the chat response.
    """
    if not leads:
        return

    # Display leads in an expander to avoid cluttering the chat
    with st.expander(f"Found {len(leads)} lead(s) - click to expand"):
        for lead_data in leads:
            lead = lead_data.get("lead", {}) if isinstance(lead_data, dict) else {}
            project = lead.get("project", {})
            score = lead_data.get("score", 0) if isinstance(lead_data, dict) else 0

            # Display lead as a card-like block
            st.markdown(f"**{project.get('project_name', 'Unknown')}** (Score: {score:.0%})")
            cols = st.columns(2)
            with cols[0]:
                if project.get("location"):
                    st.write(f"Location: {project['location']}")
                if project.get("budget"):
                    st.write(f"Budget: {project['budget']}")
            with cols[1]:
                if project.get("project_phase"):
                    st.write(f"Phase: {project['project_phase']}")
                if project.get("timeline"):
                    st.write(f"Timeline: {project['timeline']}")
            st.divider()


def _check_health():
    """Check and display the API health status."""
    try:
        # Send health check request
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            st.success(
                f"API Healthy | "
                f"v{data.get('version', '?')} | "
                f"Docs: {data.get('vector_store_count', 0)} | "
                f"Leads: {data.get('leads_count', 0)}"
            )
        else:
            st.error("API unhealthy")

    except Exception:
        st.error("Cannot reach API")


# Run the application
if __name__ == "__main__":
    main()
