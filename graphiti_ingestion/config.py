import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Defines the application's configuration settings, loaded from a .env file.
    """
    # --- Application Settings ---
    LOG_LEVEL: str = "INFO"

    # --- Neo4j Connection ---
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # --- vLLM (Gemma LLM) Connection ---
    VLLM_BASE_URL: str
    VLLM_API_KEY: str
    # The model name must match the one vLLM is serving
    VLLM_MODEL_NAME: str

    # --- Triton (Jina Embedder) Connection ---
    TRITON_URL: str

    # Pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields in the environment
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns the singleton instance of the Settings object.

    The @lru_cache decorator ensures that the Settings are loaded from the
    .env file only once, making it an efficient way to access configuration.
    """
    logger.info("Loading application settings from .env file...")
    try:
        settings = Settings()
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise


# You can optionally log the loaded settings on startup for verification
# settings = get_settings()
# logger.debug(f"Loaded settings: {settings.model_dump(exclude={'NEO4J_PASSWORD', 'VLLM_API_KEY'})}")