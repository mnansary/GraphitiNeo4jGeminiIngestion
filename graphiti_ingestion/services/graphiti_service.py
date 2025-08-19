import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType

# Note: These imports will resolve once we create the corresponding files.
from graphiti_ingestion.config import Settings, get_settings
from graphiti_ingestion.core.jina_triton_embedder import (
    JinaV3TritonEmbedder,
    JinaV3TritonEmbedderConfig,
)

logger = logging.getLogger(__name__)


class GraphitiService:
    """
    A service class to manage the Graphiti instance and its dependencies.

    This class handles the lifecycle of the Graphiti client, including the
    configuration and instantiation of the LLM and embedder clients.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.graphiti: Graphiti
        self.jina_embedder: JinaV3TritonEmbedder

        # 1. Configure the LLM Client (vLLM OpenAI-compatible server)
        llm_client_vllm = AsyncOpenAI(
            api_key=self.settings.VLLM_API_KEY,
            base_url=self.settings.VLLM_BASE_URL,
        )

        # The model name here MUST match the model ID served by vLLM
        vllm_llm_config = LLMConfig(
            small_model=self.settings.VLLM_MODEL_NAME,
            model=self.settings.VLLM_MODEL_NAME,
        )

        # 2. Configure the Embedder Client (Jina Triton Server)
        jina_config = JinaV3TritonEmbedderConfig(
            triton_url=self.settings.TRITON_URL
        )
        self.jina_embedder = JinaV3TritonEmbedder(config=jina_config)

        # 3. Initialize Graphiti with all components
        self.graphiti = Graphiti(
            uri=self.settings.NEO4J_URI,
            user=self.settings.NEO4J_USER,
            password=self.settings.NEO4J_PASSWORD,
            llm_client=OpenAIClient(
                config=vllm_llm_config,
                client=llm_client_vllm
            ),
            embedder=self.jina_embedder,
            # cross_encoder can be added here if needed
        )
        logger.info("GraphitiService initialized.")

    async def startup(self):
        """
        Initializes connections and builds necessary database constraints/indices.
        This should be called once when the application starts.
        """
        logger.info("Building Graphiti indices and constraints...")
        await self.graphiti.build_indices_and_constraints()
        logger.info("Graphiti indices and constraints are set up.")

    async def shutdown(self):
        """
        Closes all connections gracefully.
        This should be called once when the application shuts down.
        """
        logger.info("Closing Graphiti connections...")
        await self.graphiti.close()
        await self.jina_embedder.close()
        logger.info("Graphiti connections closed.")

    async def process_and_add_episode(self, episode_data: dict):
        """
        Processes a single episode payload and adds it to the graph.

        Args:
            episode_data: A dictionary containing the episode's content,
                          type, and description.
        """
        content = episode_data["content"]
        episode_type_str = episode_data["type"]
        description = episode_data["description"]
        
        # Map the string type from the API to the Graphiti Enum
        episode_type_enum = EpisodeType[episode_type_str.upper()]

        # Ensure JSON content is serialized to a string for Graphiti
        if episode_type_enum == EpisodeType.JSON:
            episode_body = json.dumps(content)
        else:
            episode_body = content
        
        episode_name = f"Ingested Episode - {datetime.now(timezone.utc).isoformat()}"

        logger.info(f"Adding episode '{episode_name}' of type '{episode_type_str}' to the graph.")
        
        await self.graphiti.add_episode(
            name=episode_name,
            episode_body=episode_body,
            source=episode_type_enum,
            source_description=description,
            reference_time=datetime.now(timezone.utc),
        )
        logger.info(f"Successfully added episode '{episode_name}'.")


# --- Dependency Injection Singleton ---
_graphiti_service_instance: Optional[GraphitiService] = None


def get_graphiti_service() -> GraphitiService:
    """
    FastAPI dependency to get the singleton instance of the GraphitiService.
    The instance is created by the application's lifespan event manager.
    """
    if _graphiti_service_instance is None:
        # This state should not be reached in a running FastAPI app
        # because the lifespan event handler initializes it on startup.
        raise RuntimeError("GraphitiService has not been initialized.")
    return _graphiti_service_instance


def initialize_graphiti_service():
    """
    Creates and stores the singleton instance of the GraphitiService.
    This is called from the main application's startup event.
    """
    global _graphiti_service_instance
    if _graphiti_service_instance is None:
        logger.info("Creating singleton instance of GraphitiService.")
        settings = get_settings()
        _graphiti_service_instance = GraphitiService(settings)
    return _graphiti_service_instance