# LLM client module - provides async and sync wrappers for the Anthropic Claude API.
# Includes retry logic with exponential backoff for rate limits and transient errors.

import os

from anthropic import Anthropic, AsyncAnthropic, RateLimitError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config.settings import get_settings
from src.utils.exceptions import LLMError
from src.utils.logger import get_logger

# Module logger for tracking LLM API calls
logger = get_logger(__name__)


class LLMClient:
    """Wrapper for the Anthropic Claude API with retry logic.

    Provides both synchronous and asynchronous methods for generating
    responses from Claude, with automatic retries on transient failures.
    """

    def __init__(self) -> None:
        """Initialize the LLM client with API credentials and model configuration.

        Reads the API key from environment variables and loads model settings.

        Raises:
            LLMError: If the ANTHROPIC_API_KEY environment variable is not set.
        """
        # Load model configuration from settings
        settings = get_settings()

        # Read the API key from environment variables
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY environment variable is not set")

        # Store model configuration
        self._model = settings.llm.model
        self._max_tokens = settings.llm.max_tokens
        self._temperature = settings.llm.temperature

        # Initialize the synchronous Anthropic client
        self._client = Anthropic(api_key=api_key)
        # Initialize the asynchronous Anthropic client
        self._async_client = AsyncAnthropic(api_key=api_key)

        # Log successful initialization
        logger.info("LLM client initialized: model=%s, max_tokens=%d", self._model, self._max_tokens)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, InternalServerError)),
        reraise=True,
    )
    def generate_response(self, system_prompt: str, user_message: str) -> str:
        """Generate a response from Claude using the synchronous client.

        Sends a message to Claude with the system prompt and user message,
        retrying on rate limits and internal server errors.

        Args:
            system_prompt: The system-level instruction for Claude's behavior.
            user_message: The user's message or query to respond to.

        Returns:
            str: The generated text response from Claude.

        Raises:
            LLMError: If the API call fails after all retry attempts.
        """
        try:
            # Log the API call at debug level
            logger.debug("Sending request to Claude: model=%s, user_msg_len=%d", self._model, len(user_message))

            # Make the synchronous API call to Claude
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Extract the text content from the response
            response_text = response.content[0].text

            # Log successful response with token usage
            logger.debug(
                "Claude response received: %d chars, input_tokens=%d, output_tokens=%d",
                len(response_text),
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            return response_text

        except (RateLimitError, InternalServerError):
            # Let tenacity handle retries for these errors
            logger.warning("Transient API error, retrying...")
            raise
        except Exception as error:
            # Wrap other errors in LLMError
            raise LLMError(
                f"LLM API call failed: {str(error)}",
                details={"model": self._model, "error": str(error)},
            ) from error

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, InternalServerError)),
        reraise=True,
    )
    async def async_generate_response(self, system_prompt: str, user_message: str) -> str:
        """Generate a response from Claude using the asynchronous client.

        Sends a message to Claude with the system prompt and user message,
        retrying on rate limits and internal server errors.

        Args:
            system_prompt: The system-level instruction for Claude's behavior.
            user_message: The user's message or query to respond to.

        Returns:
            str: The generated text response from Claude.

        Raises:
            LLMError: If the API call fails after all retry attempts.
        """
        try:
            # Log the async API call
            logger.debug("Sending async request to Claude: model=%s", self._model)

            # Make the asynchronous API call to Claude
            response = await self._async_client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Extract the text content from the response
            response_text = response.content[0].text

            # Log successful response
            logger.debug(
                "Async Claude response: %d chars, tokens=%d/%d",
                len(response_text),
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            return response_text

        except (RateLimitError, InternalServerError):
            # Let tenacity handle retries for these errors
            logger.warning("Transient async API error, retrying...")
            raise
        except Exception as error:
            # Wrap other errors in LLMError
            raise LLMError(
                f"Async LLM API call failed: {str(error)}",
                details={"model": self._model, "error": str(error)},
            ) from error
