# graphiti_ingestion/core/compatible_openai_client.py

import logging
from typing import Any

from graphiti_core.llm_client.config import ModelSize
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

# Import our new, self-healing repair service
from .repair_service import repair_and_validate

# Use Python's standard logging module
logger = logging.getLogger(__name__)


class CompatibleOpenAIClient(OpenAIClient):
    """
    A robust, production-grade OpenAI client that uses a two-tiered repair strategy
    (programmatic and LLM-powered self-healing) to ensure compatibility with
    instruction-tuned models.

    This client's primary role is to act as an orchestrator. It retrieves the raw
    response from the LLM and then delegates the complex task of fixing and
    validating the JSON payload to a dedicated repair service.
    """

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Generates a response from the LLM, then uses a dedicated, self-healing
        service to repair and validate the output before returning it.
        """
        # Step 1: Get the raw dictionary from the underlying LLM.
        # We pass `response_model=None` to the parent class's method. This is a
        # critical step that prevents the original OpenAIClient from attempting
        # its own Pydantic validation, which would fail on the malformed JSON.
        # We are taking over the validation responsibility.
        logger.debug("Calling parent OpenAIClient to get raw LLM response...")
        raw_response_dict = await super().generate_response(
            messages, response_model=None, max_tokens=max_tokens, model_size=model_size
        )

        # Step 2: If the original call did not expect a structured response,
        # there is nothing to repair or validate. Return immediately.
        if not response_model:
            logger.debug("No response_model expected. Returning raw response.")
            return raw_response_dict

        # Step 3: Delegate the entire repair and validation process to our robust service.
        # We pass in all the necessary context:
        # - `raw_dict`: The potentially messy JSON from the LLM.
        # - `response_model`: The Pydantic class we need the output to conform to.
        # - `llm_client`: A reference to this client instance (`self`), which the
        #   service needs to make a self-healing API call if the first repair fails.
        # - `original_messages`: The original prompt, needed for context in the
        #   self-healing prompt.
        logger.debug(f"Delegating repair and validation for model '{response_model.__name__}' to the repair service.")
        return await repair_and_validate(
            raw_dict=raw_response_dict,
            response_model=response_model,
            llm_client=self,
            original_messages=messages,
        )