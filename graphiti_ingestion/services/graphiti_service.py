# graphiti_ingestion/services/graphiti_service.py

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.nodes import EpisodeType

from graphiti_ingestion.config import Settings, get_settings
from graphiti_ingestion.embeder.jina_triton_embedder import (
    JinaV3TritonEmbedder,
    JinaV3TritonEmbedderConfig,
)
from graphiti_ingestion.gemini.client import ManagedGeminiClient
from graphiti_ingestion.gemini.manager import ComprehensiveManager
from graphiti_ingestion.gemini.reranker import ManagedGeminiReranker

logger = logging.getLogger(__name__)


class GraphitiService:
    """
    A service class to manage the Graphiti instance and its dependencies.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.graphiti: Graphiti
        self.jina_embedder: JinaV3TritonEmbedder
        self.managed_llm_client: ManagedGeminiClient
        self.managed_reranker: ManagedGeminiReranker

        logger.info("Initializing ComprehensiveManager for Gemini API...")
        gemini_manager = ComprehensiveManager(
            api_key_csv_path=self.settings.GEMINI_API_CSV_PATH,
            model_config_path=self.settings.GEMINI_MODEL_CONFIG,
            api_key_cooldown_seconds=self.settings.GEMINI_API_KEY_COOLDOWN_SECONDS,
        )

        logger.info("Initializing ManagedGeminiClient...")
        self.managed_llm_client = ManagedGeminiClient(
            manager=gemini_manager,
            config=LLMConfig(temperature=self.settings.GEMINI_MODEL_TEMPERATURE),
            global_cooldown_seconds=self.settings.GEMINI_GLOBAL_COOLDOWN_SECONDS,
        )

        logger.info("Initializing ManagedGeminiReranker...")
        self.managed_reranker = ManagedGeminiReranker(
            manager=gemini_manager,
            config=LLMConfig(model=self.settings.GEMINI_DEFAULT_RERANKER),
            global_cooldown_seconds=self.settings.GEMINI_GLOBAL_COOLDOWN_SECONDS,
        )

        logger.info("Initializing Jina Triton Embedder...")
        jina_config = JinaV3TritonEmbedderConfig(
            triton_url=self.settings.TRITON_URL
        )
        self.jina_embedder = JinaV3TritonEmbedder(config=jina_config)

        logger.info("Initializing Graphiti Core...")
        self.graphiti = Graphiti(
            uri=self.settings.NEO4J_URI,
            user=self.settings.NEO4J_USER,
            password=self.settings.NEO4J_PASSWORD,
            llm_client=self.managed_llm_client,
            embedder=self.jina_embedder,
            cross_encoder=self.managed_reranker
        )
        logger.info("GraphitiService initialized successfully.")


    async def startup(self) -> None:
        """Initializes connections and builds database constraints/indices."""
        logger.info("Building Graphiti indices and constraints in Neo4j...")
        await self.graphiti.build_indices_and_constraints()
        logger.info("Graphiti indices and constraints are set up.")

    async def shutdown(self) -> None:
        """Closes all connections gracefully."""
        logger.info("Closing all service connections...")
        if self.graphiti:
            await self.graphiti.close()
        if self.jina_embedder:
            await self.jina_embedder.close()
        if self.managed_llm_client:
            self.managed_llm_client.close()
            logger.info("ManagedGeminiClient worker has been closed.")
        if self.managed_reranker:
            self.managed_reranker.close()
            logger.info("ManagedGeminiReranker worker has been closed.")
        logger.info("All services and connections are now closed.")

    async def process_and_add_episode(
        self, episode_data: Dict[str, Any], retry_count: int = 0
    ) -> None:
        """
        Processes a single episode payload and adds it to the graph.
        """
        content = episode_data["content"]
        episode_type_str = episode_data["type"]
        description = episode_data["description"]
        
        # ---> THIS IS THE CORRECTED LINE <---
        # The EpisodeType enum in graphiti-core uses lowercase members ('text', 'json').
        # We remove the `.upper()` to prevent a KeyError.
        episode_type_enum = EpisodeType[episode_type_str]
        # ---> END OF CORRECTION <---

        episode_body = json.dumps(content, ensure_ascii=False) if episode_type_enum == EpisodeType.json else content
        
        episode_name = f"Ingested Episode - {datetime.now(timezone.utc).isoformat()}"

        logger.info(f"Adding episode '{episode_name}' (Attempt #{retry_count + 1}) to the graph.")
        
        try:
            self.managed_llm_client.set_retry_state(is_retry=retry_count > 0)
            await self.graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source=episode_type_enum,
                source_description=description,
                reference_time=datetime.now(timezone.utc)
            )
            logger.info(f"Successfully added episode '{episode_name}'.")
        finally:
            self.managed_llm_client.set_retry_state(is_retry=False)


# --- Dependency Injection Singleton ---
_graphiti_service_instance: Optional[GraphitiService] = None

def get_graphiti_service() -> GraphitiService:
    """FastAPI dependency to get the singleton GraphitiService instance."""
    if _graphiti_service_instance is None:
        raise RuntimeError("GraphitiService has not been initialized.")
    return _graphiti_service_instance

def initialize_graphiti_service() -> GraphitiService:
    """Creates and stores the singleton instance of the GraphitiService."""
    global _graphiti_service_instance
    if _graphiti_service_instance is None:
        logger.info("Creating singleton instance of GraphitiService.")
        settings = get_settings()
        _graphiti_service_instance = GraphitiService(settings)
    return _graphiti_service_instance