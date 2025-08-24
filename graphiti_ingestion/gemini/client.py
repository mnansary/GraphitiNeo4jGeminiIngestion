# graphiti_ingestion/gemini/client.py

from __future__ import annotations

import asyncio
import json
import logging
import queue
from asyncio import Future
from typing import Any, Dict, List, Tuple

from google.genai import types
from pydantic import BaseModel

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.gemini_client import (
    MULTILINGUAL_EXTRACTION_RESPONSES,
)
from graphiti_core.prompts.models import Message

from .manager import ComprehensiveManager
from .worker import GeminiAPIWorker

logger = logging.getLogger(__name__)


class ManagedGeminiClient(LLMClient):
    """
    An async-safe Gemini client compatible with `graphiti-core`.

    This client offloads all API calls to a dedicated synchronous worker thread.
    This ensures strict, sequential execution to manage rate limits and API key
    rotation effectively. It uses an internal state variable (`_is_retry_request`)
    as a "side channel" to communicate retry attempts to the worker, a workaround
    for `graphiti-core`'s inability to pass custom arguments.
    """

    def __init__(
        self,
        manager: ComprehensiveManager,
        config: LLMConfig | None = None,
        cache: bool = False,
        global_cooldown_seconds: float = 1.0,
    ):
        super().__init__(config or LLMConfig(), cache)
        self.manager = manager
        self.config = config or LLMConfig()
        self._work_queue: queue.Queue = queue.Queue()
        self._worker = GeminiAPIWorker(
            manager=self.manager,
            work_queue=self._work_queue,
            delay_between_calls=global_cooldown_seconds,
        )
        self._worker.start()

        # Internal state to signal a retry attempt to the worker
        self._is_retry_request: bool = False

        logger.info("ManagedGeminiClient initialized and worker thread started.")

    def set_retry_state(self, is_retry: bool) -> None:
        """
        Sets the retry state for the NEXT `generate_response` call.

        This is a workaround for `graphiti-core`'s lack of custom arg passing.
        It should be called by the service layer immediately before a call
        to `graphiti.add_episode` that is known to be a retry.

        Args:
            is_retry: True if the next request should use the retry logic.
        """
        self._is_retry_request = is_retry

    def close(self) -> None:
        """Shuts down the worker thread cleanly."""
        if self._worker.is_alive():
            self._work_queue.put(None)
            self._worker.join(timeout=5.0)  # Prevent hanging on shutdown
            logger.info("ManagedGeminiClient worker has been closed.")

    async def _execute_job(
        self,
        messages: List[Message],
        gen_config: Dict[str, Any],
        retry_count: int,
    ) -> Tuple[types.GenerateContentResponse, str]:
        """
        Submits a job to the worker's queue and awaits the result.

        Args:
            messages: The list of messages for the prompt.
            gen_config: The generation configuration dictionary.
            retry_count: The number of times this job has been attempted.

        Returns:
            A tuple containing the API response and the name of the model used.
        """
        loop = asyncio.get_running_loop()
        future: Future = loop.create_future()
        # The job tuple now includes the retry_count for the worker
        self._work_queue.put((messages, gen_config, future, loop, retry_count))
        return await future

    async def _generate_response(
        self,
        messages: List[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> Dict[str, Any]:
        """
        Prepares the request, passes the current retry state to the worker,
        and parses the final response.
        """
        system_prompt = ""
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content or ""
            messages = messages[1:]

        # Base generation config. The worker will override max_output_tokens.
        generation_config = {
            "temperature": self.config.temperature,
            "max_output_tokens": max_tokens or 8192,
            "response_mime_type": "application/json" if response_model else "text/plain",
            "response_schema": (
                response_model.model_json_schema() if response_model else None
            ),
        }
        # The google-genai library expects system_instruction to be a top-level
        # argument to generate_content, not part of the config dict.
        # However, our worker's signature is generic. We will handle this there.
        # For now, let's keep it simple. The worker will handle passing it correctly.
        # Our `_to_contents` function handles system prompts now.

        try:
            retry_count_for_worker = 1 if self._is_retry_request else 0
            response, used_model = await self._execute_job(
                messages, generation_config, retry_count_for_worker
            )
        except Exception as e:
            logger.error(f"ManagedGeminiClient: job failed in worker: {e}")
            raise

        raw_output = getattr(response, "text", None)

        if not raw_output:
            raise ValueError(
                f"Model {used_model} returned no text. This could be due to "
                f"safety filters or other limits. Full response: {response}"
            )

        if response_model:
            try:
                validated = response_model.model_validate_json(raw_output)
                return validated.model_dump()
            except Exception as e:
                logger.error(f"Failed to parse JSON from model {used_model}. Raw output: {raw_output}")
                raise ValueError(f"Failed to parse structured JSON from {used_model}: {e}") from e

        return {"content": raw_output}

    async def generate_response(
        self,
        messages: List[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        **kwargs, # Accept and ignore any other keyword arguments
    ) -> Dict[str, Any]:
        """
        Public entrypoint for `graphiti-core`.

        This method relies on the `_is_retry_request` internal state, which
        must be set via `set_retry_state()` before this method is called.
        """
        if messages and messages[0].content:
            messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        return await self._generate_response(
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )