import json
import logging
from datetime import datetime, timezone
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.nodes import EpisodeType

from graphiti_ingestion.gemini.manager import ComprehensiveManager
from graphiti_ingestion.gemini.client import ManagedGeminiClient
from graphiti_ingestion.gemini.reranker import ManagedGeminiReranker

from graphiti_ingestion.config import Settings, get_settings

from graphiti_ingestion.embeder.jina_triton_embedder import (
    JinaV3TritonEmbedder,
    JinaV3TritonEmbedderConfig,
)

logger = logging.getLogger(__name__)


class GraphitiService:
    """
    A service class to manage the Graphiti instance and its dependencies,
    now powered by Google Gemini and configured via a Pydantic Settings object.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.graphiti: Graphiti
        self.jina_embedder: JinaV3TritonEmbedder
        self.managed_llm_client: ManagedGeminiClient
        self.managed_reranker: ManagedGeminiReranker

        # 1. Initialize the Gemini API Manager using settings from the .env file
        logger.info("Initializing ComprehensiveManager for Gemini API...")
        gemini_manager = ComprehensiveManager(
            api_key_csv_path=self.settings.GEMINI_API_CSV_PATH,
            model_config_path=self.settings.GEMINI_MODEL_CONFIG,
            api_key_cooldown_seconds=self.settings.GEMINI_API_KEY_COOLDOWN_SECONDS,
        )

        # 2. Initialize the Managed Gemini Client (for LLM tasks)
        logger.info("Initializing ManagedGeminiClient...")
        self.managed_llm_client = ManagedGeminiClient(
            manager=gemini_manager,
            config=LLMConfig(temperature=self.settings.GEMINI_MODEL_TEMPERATURE),
            global_cooldown_seconds=self.settings.GEMINI_GLOBAL_COOLDOWN_SECONDS,
        )

        # 3. Initialize the Managed Gemini Reranker (for Cross-Encoder tasks)
        logger.info("Initializing ManagedGeminiReranker...")
        self.managed_reranker = ManagedGeminiReranker(
            manager=gemini_manager,
            config=LLMConfig(model=self.settings.GEMINI_DEFAULT_RERANKER),
            global_cooldown_seconds=self.settings.GEMINI_GLOBAL_COOLDOWN_SECONDS,
        )


        # 4. Configure the Embedder Client (Jina Triton Server)
        jina_config = JinaV3TritonEmbedderConfig(
            triton_url=self.settings.TRITON_URL
        )
        self.jina_embedder = JinaV3TritonEmbedder(config=jina_config)

        # 5. Initialize Graphiti with all the new components
        self.graphiti = Graphiti(
            uri=self.settings.NEO4J_URI,
            user=self.settings.NEO4J_USER,
            password=self.settings.NEO4J_PASSWORD,
            llm_client=self.managed_llm_client,
            embedder=self.jina_embedder,
            cross_encoder=self.managed_reranker
        )
        logger.info("GraphitiService initialized with Gemini clients.")


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
        Closes all connections gracefully, including the Gemini worker threads.
        """
        logger.info("Closing Graphiti connections and shutting down services...")
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
        episode_type_enum = EpisodeType[episode_type_str]

        # Ensure JSON content is serialized to a string for Graphiti
        if episode_type_enum == EpisodeType.json:
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