# Dockerfile - Multi-stage build for the FastAPI backend application.
# Stage 1: Install dependencies with Poetry.
# Stage 2: Production image with only runtime dependencies.

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

# Set working directory for the build stage
WORKDIR /app

# Install Poetry for dependency management
RUN pip install --no-cache-dir poetry==1.8.4

# Copy dependency files first for layer caching
COPY pyproject.toml ./

# Export dependencies to requirements.txt format (without dev dependencies)
RUN poetry export -f requirements.txt --without-hashes --without dev -o requirements.txt

# Stage 2: Production image
FROM python:3.11-slim AS production

# Set working directory for the production image
WORKDIR /app

# Install system dependencies for document processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements from builder stage
COPY --from=builder /app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create logs directory for the application
RUN mkdir -p logs

# Expose the FastAPI server port
EXPOSE 8000

# Health check to verify the application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the FastAPI application with uvicorn
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
