# managed_gemini_reranker.py

from __future__ import annotations

import asyncio
import json
import queue
import typing
from asyncio import Future

from google.genai import types
from pydantic import BaseModel, Field

# --- Local Project Imports ---
from graphiti_ingestion.gemini.manager import ComprehensiveManager
from graphiti_ingestion.gemini.worker import GeminiAPIWorker

# --- Graphiti Core Imports ---
from graphiti_core.cross_encoder.client import CrossEncoderClient # CORRECT IMPORT
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message

import logging

logger = logging.getLogger(__name__)


# Define the expected JSON structure for the LLM response.
class RerankedDocument(BaseModel):
    """Represents a single document with its relevance score."""
    document: str
    relevance_score: float = Field(
        ...,
        description="A score from 0.0 to 1.0 indicating relevance.",
        ge=0.0,
        le=1.0,
    )

class RerankResponse(BaseModel):
    """The root model for the reranking JSON response."""
    reranked_documents: list[RerankedDocument]


# CORRECTED CLASS DEFINITION: Inherits from CrossEncoderClient
class ManagedGeminiReranker(CrossEncoderClient):
    """
    A Graphiti-compatible cross-encoder that uses a managed, sequential Gemini
    worker for robust, rate-limit-aware reranking.
    """

    def __init__(
        self,
        manager: ComprehensiveManager,
        config: LLMConfig | None = None,
        global_cooldown_seconds: float = 1.0,
    ):
        self.manager = manager
        self.config = config or LLMConfig()

        self._work_queue: queue.Queue = queue.Queue()
        self._worker = GeminiAPIWorker(
            manager=self.manager,
            work_queue=self._work_queue,
            delay_between_calls=global_cooldown_seconds
        )
        self._worker.start()
        logger.info("ManagedGeminiReranker worker thread started.")

    def close(self) -> None:
        """Gracefully stop the reranker worker."""
        if self._worker.is_alive():
            self._work_queue.put(None)
            self._worker.join()
            logger.info("ManagedGeminiReranker worker has been closed.")

    async def _execute_job(
        self,
        messages: list[Message],
        gen_config: types.GenerateContentConfig
    ) -> tuple[types.GenerateContentResponse, str]:
        loop = asyncio.get_running_loop()
        future: Future = loop.create_future()
        self._work_queue.put((messages, gen_config, future, loop))
        return await future

    # CORRECTED METHOD: Renamed to `rank` with the required signature.
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Reranks passages for a query using a single, structured call to the Gemini API.

        This method implements the abstract method from CrossEncoderClient.
        """
        if not passages:
            return []
        if len(passages) == 1:
            return [(passages[0], 1.0)]

        # Updated instruction to request a specific JSON format.
        instruction = (
            "You are an expert reranker. Given a query and a list of documents, "
            "you must reorder the documents from most to least relevant to the query. "
            "Provide a relevance score between 0.0 and 1.0 for each document. "
            f"Your output MUST be a valid JSON object matching this schema: {RerankResponse.model_json_schema()}"
        )

        system_msg = Message(role="system", content=instruction)
        user_msg = Message(
            role="user",
            content=json.dumps({"query": query, "documents": passages}, ensure_ascii=False)
        )
        messages = [system_msg, user_msg]

        generation_config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
            response_mime_type="application/json",
            # --- THIS IS THE CORRECTED LINE ---
            response_schema=RerankResponse.model_json_schema(),
            # --- END OF CORRECTION ---
        )

        try:
            # The _execute_job will return a dictionary pre-validated against RerankResponse
            response_dict = await self._execute_job_with_model(messages, generation_config)

            # Transform the dictionary into the required list[tuple[str, float]] format
            results = [
                (item['document'], item['relevance_score'])
                for item in response_dict.get('reranked_documents', [])
            ]
            return results

        except Exception as e:
            logger.error(f"ManagedGeminiReranker failed during rank: {e}")
            # As a fallback, return the original passages with a neutral score
            return [(p, 0.5) for p in passages]

    async def _execute_job_with_model(
        self, messages: list[Message], gen_config: types.GenerateContentConfig
    ) -> dict[str, typing.Any]:
        """Helper to execute job and parse with the internal Pydantic model."""
        try:
            response, used_model = await self._execute_job(messages, gen_config)
        except Exception as e:
            logger.error(f"ManagedGeminiReranker: job failed in worker: {e}")
            raise

        raw_output = getattr(response, "text", None)
        if not raw_output:
            raise ValueError(f"Model {used_model} returned no text for reranking.")
        try:
            # The API should already return validated JSON, but we re-validate for safety.
            validated = RerankResponse.model_validate(json.loads(raw_output))
            return validated.model_dump()
        except Exception as e:
            raise ValueError(
                f"Failed to parse reranker JSON from model {used_model}: {e}"
            ) from e