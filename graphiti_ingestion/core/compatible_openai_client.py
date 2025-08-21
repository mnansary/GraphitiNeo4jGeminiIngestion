# graphiti_ingestion/core/compatible_openai_client.py

import json
import logging
from typing import Any

from graphiti_core.llm_client.client import MULTILINGUAL_EXTRACTION_RESPONSES
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

# We import this here to avoid circular dependency issues
from .repair_service import repair_and_validate

logger = logging.getLogger(__name__)


class CompatibleOpenAIClient(OpenAIClient):
    """
    A robust, production-grade OpenAI client that uses a two-tiered repair strategy
    and includes a corrected retry loop to ensure compatibility with strict models like Gemma.
    """

    async def _direct_llm_call(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Makes a direct, base-level call to the LLM without any repair or
        complex retry logic. This is used by the self-healing mechanism to prevent
        recursive loops.
        """
        # We call the parent method from graphiti-core's OpenAIClient directly.
        return await super().generate_response(
            messages, response_model=None, max_tokens=max_tokens, model_size=model_size
        )

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
        # Step 1: Prepare the initial request
        if max_tokens is None:
            max_tokens = self.max_tokens

        if response_model:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[-1].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )
        
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES
        
        # Step 2: Make the initial, raw call to the LLM.
        raw_response_dict = await self._direct_llm_call(
            messages, max_tokens=max_tokens, model_size=model_size
        )

        # Step 3: If no structured response is expected, return immediately.
        if not response_model:
            return raw_response_dict

        # Step 4: Delegate the entire repair and validation process to our robust service.
        return await repair_and_validate(
            raw_dict=raw_response_dict,
            response_model=response_model,
            llm_client=self,
            original_messages=messages,
        )