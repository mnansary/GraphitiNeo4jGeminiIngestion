# managed_gemini_client.py

from __future__ import annotations

import asyncio
import json
import queue
import typing
from asyncio import Future
from google.genai import types
from pydantic import BaseModel

# --- Local Project Imports ---
from graphiti_ingestion.gemini.manager import ComprehensiveManager
from graphiti_ingestion.gemini.worker import GeminiAPIWorker

# --- Graphiti Core Imports ---
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message
from graphiti_core.llm_client.gemini_client import MULTILINGUAL_EXTRACTION_RESPONSES
import logging

logger = logging.getLogger(__name__)

class ManagedGeminiClient(LLMClient):
    """
    Async-safe Gemini client that offloads API calls to a dedicated
    synchronous worker thread for strict sequential execution.
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
            delay_between_calls=global_cooldown_seconds
        )
        self._worker.start()
        logger.info("ManagedGeminiClient initialized and worker thread started.")

    def close(self) -> None:
        """Shut down the worker thread cleanly."""
        if self._worker.is_alive():
            self._work_queue.put(None)
            self._worker.join()
            logger.info("ManagedGeminiClient worker has been closed.")

    async def _execute_job(
        self,
        messages: list[Message],
        gen_config: types.GenerateContentConfig
    ) -> tuple[types.GenerateContentResponse, str]:
        """
        Put a job in the worker queue and wait for the result.
        """
        loop = asyncio.get_running_loop()
        future: Future = loop.create_future()
        self._work_queue.put((messages, gen_config, future, loop))
        return await future

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Prepare the request config, execute via the worker, and parse the response.
        """
        system_prompt = ""
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content or ""
            messages = messages[1:]

        generation_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=max_tokens or 8192,
            response_mime_type="application/json" if response_model else None,
            # --- THIS IS THE CORRECTED LINE ---
            response_schema=(
                response_model.model_json_schema()
                if response_model else None
            ),
            # --- END OF CORRECTION ---
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            response, used_model = await self._execute_job(messages, generation_config)
        except Exception as e:
            logger.error(f"ManagedGeminiClient: job failed in worker: {e}")
            raise

        raw_output = getattr(response, "text", None)
        if response_model:
            if not raw_output:
                raise ValueError(f"Model {used_model} returned no text for structured output.")
            try:
                validated = response_model.model_validate(json.loads(raw_output))
                return validated.model_dump()
            except Exception as e:
                raise ValueError(
                    f"Failed to parse structured JSON from model {used_model}: {e}"
                ) from e

        return {"content": raw_output}

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Public entrypoint for Graphiti-core to request a Gemini generation.
        """
        if messages and messages[0].content:
            messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        return await self._generate_response(
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )