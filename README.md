# Construction Leads Finder

A RAG-powered chatbot agent that ingests construction documents and uses AI to identify, extract, and rank construction leads — project opportunities with details like owner, budget, timeline, location, and contacts.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Setup and Installation](#setup-and-installation)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [Docker Deployment](#docker-deployment)
9. [API Endpoints](#api-endpoints)
10. [Testing](#testing)

---

## Overview

This application combines **RAG (Retrieval-Augmented Generation)** with a **LangGraph agent** and **Anthropic Claude** to:

- Ingest construction documents (PDF, DOCX, TXT, Excel)
- Chunk and embed documents into a ChromaDB vector store
- Accept natural language queries via a chat interface
- Route queries through an intelligent agent graph
- Extract structured construction leads from relevant documents
- Score and rank leads by completeness and quality
- Present results via a Streamlit chat UI

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Chat UI (:8501)                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend (:8000)                         │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ /api/chat  │  │ /api/ingest│  │/api/leads│  │/api/health │  │
│  └─────┬──────┘  └─────┬──────┘  └──────────┘  └────────────┘  │
└────────┼────────────────┼───────────────────────────────────────┘
         │                │
┌────────▼────────┐  ┌────▼─────────────────┐
│  LangGraph Agent│  │  Ingestion Pipeline  │
│  ┌────────────┐ │  │  Load → Split → Embed│
│  │Router Node │ │  └────────────┬─────────┘
│  └─────┬──────┘ │               │
│  ┌─────▼──────┐ │  ┌────────────▼─────────┐
│  │Retrieval   │ │  │   ChromaDB Vector    │
│  │Node        │◄├──┤   Store (Persistent) │
│  └─────┬──────┘ │  └──────────────────────┘
│  ┌─────▼──────┐ │
│  │Lead Extract│ │  ┌──────────────────────┐
│  │Node (Claude├─┼──►  Anthropic Claude    │
│  └─────┬──────┘ │  │  (claude-sonnet-4)   │
│  ┌─────▼──────┐ │  └──────────────────────┘
│  │Lead Scoring│ │
│  └─────┬──────┘ │
│  ┌─────▼──────┐ │
│  │Summarize   │ │
│  └────────────┘ │
└─────────────────┘
```

---

## Project Structure

```
ai-genai-usecases-document-summarization/
├── pyproject.toml              # Poetry dependencies and tool config
├── .env.example                # Environment variable template
├── .gitignore                  # Python gitignore patterns
├── Dockerfile                  # FastAPI backend container
├── Dockerfile.streamlit        # Streamlit UI container
├── docker-compose.yml          # Multi-service orchestration
├── config/
│   ├── settings.yaml           # Application configuration
│   └── logging.yaml            # Logging configuration
├── data/
│   └── sample_documents/       # Place documents here for ingestion
├── scripts/
│   └── ingest_documents.py     # CLI ingestion script
├── src/
│   ├── main.py                 # FastAPI app entry point
│   ├── config/                 # Pydantic settings loader
│   ├── models/                 # Data models (leads, agent state)
│   ├── ingestion/              # Document loading and chunking
│   ├── rag/                    # Embeddings, vector store, retriever
│   ├── agent/                  # LangGraph agent (graph, nodes, tools)
│   ├── api/                    # FastAPI routes and middleware
│   ├── services/               # Business logic (ingestion, leads)
│   └── utils/                  # Logger, sanitizer, exceptions
├── ui/
│   └── streamlit_app.py        # Chat interface
├── tests/                      # pytest test suite
└── logs/                       # Application log files
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| fastapi | REST API framework |
| uvicorn | ASGI server |
| anthropic | Claude LLM API client |
| langgraph | Agent graph orchestration |
| langchain-core | LangGraph message types |
| chromadb | Vector database |
| sentence-transformers | Embedding generation |
| pypdf2 | PDF document parsing |
| python-docx | DOCX document parsing |
| openpyxl | Excel file parsing |
| pydantic / pydantic-settings | Data validation and settings |
| pyyaml | YAML configuration parsing |
| python-dotenv | Environment variable loading |
| streamlit | Chat UI frontend |
| tenacity | Retry logic for API calls |

---

## Setup and Installation

### Prerequisites

- Python 3.11+
- Poetry (install: `pip install poetry`)
- An Anthropic API key

### Installation Steps

```bash
# 1. Clone the repository
cd ai-genai-usecases-document-summarization

# 2. Install dependencies with Poetry
poetry install

# 3. Create your environment file
cp .env.example .env

# 4. Add your Anthropic API key to .env
# Edit .env and set: ANTHROPIC_API_KEY=your_key_here

# 5. Create the logs directory (if not exists)
mkdir -p logs
```

---

## Configuration

All configuration is managed via `config/settings.yaml` with environment variable overrides.

### Key Configuration Sections

| Section | Description | Env Prefix |
|---------|-------------|------------|
| app | Server host, port, log level | `APP_` |
| llm | Claude model, max tokens, temperature | `LLM_` |
| embeddings | Model name and dimension | `EMBEDDINGS_` |
| vector_store | ChromaDB path and collection | `CHROMA_` |
| rag | Chunk size, overlap, top_k, max distance | `RAG_` |
| ingestion | Supported formats, max file size | `INGESTION_` |
| lead_extraction | Scoring weights, confidence threshold | `LEAD_` |
| agent | Max iterations, recursion limit | `AGENT_` |
| ui | Streamlit port, API base URL | `UI_` |

Environment variables take precedence over YAML values.

---

## Running the Application

### Start the FastAPI Backend

```bash
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Streamlit UI

```bash
poetry run streamlit run ui/streamlit_app.py --server.port 8501
```

### Ingest Documents via CLI

```bash
poetry run python scripts/ingest_documents.py ./data/sample_documents
```

### Ingest a Single File

```bash
poetry run python scripts/ingest_documents.py ./data/sample_documents/permit.pdf
```

---

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Set your API key in .env first
docker-compose up --build
```

This starts:
- **API** at http://localhost:8000
- **Streamlit UI** at http://localhost:8501

### Stop Services

```bash
docker-compose down
```

### Persistent Data

ChromaDB data is persisted via a Docker volume (`chroma_data`). To reset:

```bash
docker-compose down -v
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check with metrics |
| POST | `/api/chat` | Chat with the agent |
| POST | `/api/ingest` | Ingest from directory/file paths |
| POST | `/api/ingest/upload` | Upload and ingest a file |
| GET | `/api/leads` | List extracted leads (paginated) |
| GET | `/api/leads/{id}` | Get single lead detail |
| DELETE | `/api/leads` | Clear all leads |

### Example Chat Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find construction leads in downtown Chicago"}'
```

### Example Ingest Request

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "./data/sample_documents"}'
```

---

## Testing

### Run All Tests

```bash
poetry run pytest
```

### Run with Coverage

```bash
poetry run pytest --cov=src --cov-report=term-missing
```

### Run Specific Test Module

```bash
poetry run pytest tests/test_agent/test_lead_scoring_node.py -v
```

### Linting and Formatting

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/
```
