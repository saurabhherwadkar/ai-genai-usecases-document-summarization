# Ingest documents script - CLI tool for batch document ingestion.
# Ingests all supported documents from a specified directory into the vector store.

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.services.ingestion_service import IngestionService
from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Module logger for tracking script execution
logger = get_logger(__name__)


def main():
    """Main entry point for the document ingestion CLI script.

    Parses command-line arguments and runs the ingestion pipeline
    against the specified directory or file.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Ingest construction documents into the vector store for RAG retrieval."
    )

    # Add directory argument
    parser.add_argument(
        "path",
        type=str,
        help="Path to a directory or file to ingest",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Validate the provided path exists
    path = Path(args.path)
    if not path.exists():
        logger.error("Path does not exist: %s", args.path)
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)

    # Initialize the ingestion service
    service = IngestionService()

    # Log the ingestion start
    logger.info("Starting document ingestion from: %s", args.path)
    print(f"Ingesting documents from: {args.path}")

    try:
        # Determine if path is a file or directory and ingest accordingly
        if path.is_file():
            result = service.ingest_file(str(path))
        else:
            result = service.ingest_directory(str(path))

        # Display the results
        print(f"\nIngestion complete!")
        print(f"  Status: {result['status']}")
        print(f"  Documents processed: {result['documents_processed']}")
        print(f"  Chunks created: {result['chunks_created']}")

        # Log the result
        logger.info("Ingestion complete: %s", result)

    except Exception as error:
        logger.error("Ingestion failed: %s", str(error))
        print(f"\nError: {str(error)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
