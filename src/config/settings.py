# Settings module - loads application configuration from YAML file with environment variable overrides.
# Uses Pydantic Settings for validation and type safety.

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml_config() -> dict:
    """Load the YAML configuration file from the config directory.

    Returns:
        dict: Parsed YAML configuration dictionary.
    """
    # Determine the config file path relative to the project root
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    # Read and parse the YAML configuration file
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


class AppSettings(BaseSettings):
    """Application-level settings such as name, version, host, and port."""

    # Application display name
    name: str = Field(default="Construction Leads Finder", description="Application name")
    # Semantic version of the application
    version: str = Field(default="0.1.0", description="Application version")
    # Deployment environment (development, staging, production)
    env: str = Field(default="development", description="Application environment")
    # Server bind host address
    host: str = Field(default="0.0.0.0", description="Server host")
    # Server bind port number
    port: int = Field(default=8000, description="Server port")
    # Minimum log level for the application
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = SettingsConfigDict(env_prefix="APP_")


class LLMSettings(BaseSettings):
    """Anthropic Claude LLM configuration settings."""

    # Claude model identifier to use for generation
    model: str = Field(default="claude-sonnet-4-20250514", description="Claude model name")
    # Maximum tokens in the generated response
    max_tokens: int = Field(default=4096, description="Maximum response tokens")
    # Sampling temperature for response diversity
    temperature: float = Field(default=0.2, description="Sampling temperature")

    model_config = SettingsConfigDict(env_prefix="LLM_")


class EmbeddingsSettings(BaseSettings):
    """Embedding model configuration settings."""

    # Sentence-transformers model name for generating embeddings
    model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    # Dimensionality of the embedding vectors
    dimension: int = Field(default=384, description="Embedding vector dimension")

    model_config = SettingsConfigDict(env_prefix="EMBEDDINGS_")


class VectorStoreSettings(BaseSettings):
    """ChromaDB vector store configuration settings."""

    # Directory path for ChromaDB persistent storage
    persist_directory: str = Field(default="./chroma_data", description="ChromaDB storage path")
    # Collection name within ChromaDB for this application
    collection_name: str = Field(default="construction_documents", description="ChromaDB collection name")

    model_config = SettingsConfigDict(env_prefix="CHROMA_")


class RAGSettings(BaseSettings):
    """RAG pipeline configuration settings."""

    # Number of characters per text chunk
    chunk_size: int = Field(default=1000, description="Text chunk size in characters")
    # Number of overlapping characters between consecutive chunks
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    # Number of top results to retrieve from vector store
    top_k: int = Field(default=8, description="Number of retrieval results")
    # Maximum distance threshold for retrieval relevance filtering
    max_distance: float = Field(default=1.5, description="Maximum retrieval distance threshold")

    model_config = SettingsConfigDict(env_prefix="RAG_")


class IngestionSettings(BaseSettings):
    """Document ingestion pipeline configuration settings."""

    # List of supported file extensions for ingestion
    supported_formats: list[str] = Field(
        default=[".pdf", ".docx", ".txt", ".xlsx", ".xls"],
        description="Supported document formats",
    )
    # Default directory containing documents to ingest
    documents_directory: str = Field(default="./data/sample_documents", description="Documents directory path")
    # Maximum allowed file size in megabytes
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")

    model_config = SettingsConfigDict(env_prefix="INGESTION_")


class LeadExtractionSettings(BaseSettings):
    """Lead extraction and scoring configuration settings."""

    # Minimum confidence score to accept an extracted lead
    min_confidence_score: float = Field(default=0.5, description="Minimum lead confidence score")
    # Maximum number of leads to return per query
    max_leads_per_query: int = Field(default=20, description="Maximum leads per query")
    # Weights for individual scoring factors
    scoring_weights: dict = Field(
        default={
            "completeness": 0.3,
            "budget_presence": 0.2,
            "timeline_presence": 0.2,
            "contact_info": 0.15,
            "recency": 0.15,
        },
        description="Lead scoring weight configuration",
    )

    model_config = SettingsConfigDict(env_prefix="LEAD_")


class AgentSettings(BaseSettings):
    """LangGraph agent configuration settings."""

    # Maximum number of iterations the agent graph can execute
    max_iterations: int = Field(default=5, description="Maximum agent iterations")
    # Recursion depth limit for graph execution
    recursion_limit: int = Field(default=25, description="Graph recursion limit")

    model_config = SettingsConfigDict(env_prefix="AGENT_")


class UISettings(BaseSettings):
    """Streamlit UI configuration settings."""

    # Port number for the Streamlit application
    port: int = Field(default=8501, description="Streamlit port")
    # Base URL of the FastAPI backend for API calls
    api_base_url: str = Field(default="http://localhost:8000", description="API base URL")

    model_config = SettingsConfigDict(env_prefix="UI_")


class Settings:
    """Root settings container that aggregates all configuration sections.

    Loads values from YAML configuration file and allows environment variable overrides.
    """

    def __init__(self) -> None:
        """Initialize settings by loading YAML config and creating typed sub-settings."""
        # Load the base YAML configuration
        yaml_config = _load_yaml_config()

        # Initialize each settings section with YAML values as defaults
        self.app = AppSettings(**yaml_config.get("app", {}))
        self.llm = LLMSettings(**yaml_config.get("llm", {}))
        self.embeddings = EmbeddingsSettings(**yaml_config.get("embeddings", {}))
        self.vector_store = VectorStoreSettings(**yaml_config.get("vector_store", {}))
        self.rag = RAGSettings(**yaml_config.get("rag", {}))
        self.ingestion = IngestionSettings(**yaml_config.get("ingestion", {}))
        self.lead_extraction = LeadExtractionSettings(**yaml_config.get("lead_extraction", {}))
        self.agent = AgentSettings(**yaml_config.get("agent", {}))
        self.ui = UISettings(**yaml_config.get("ui", {}))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the singleton application settings instance.

    Returns:
        Settings: The cached application settings object.
    """
    return Settings()
