# graphiti_ingestion/config.py

import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Defines the application's configuration settings, loaded from a .env file.
    Pydantic automatically validates the types and presence of these settings.
    """
    # --- Application Settings ---
    LOG_LEVEL: str = "INFO"
    JOB_QUEUE_PATH: DirectoryPath # Ensures the path exists

    # --- Neo4j Connection ---
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # --- Triton (Jina Embedder) Connection ---d
    TRITON_URL: str
    
    # --- Gemini API Manager Settings ---
    GEMINI_API_CSV_PATH: str
    GEMINI_MODEL_CONFIG: str
    GEMINI_MODEL_TEMPERATURE: float = 0.3
    GEMINI_MODEL_SIZE: str = "medium"
    GEMINI_DEFAULT_RERANKER: str = "gemini-2.5-flash-lite"
    GEMINI_GLOBAL_COOLDOWN_SECONDS: float = 5.0
    GEMINI_API_KEY_COOLDOWN_SECONDS: float = 60.0
    POST_SUCCESS_DELAY_SECONDS: float = 60.0

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
        logger.critical(f"FATAL: Failed to load settings from .env file: {e}", exc_info=True)
        raise

# You can optionally log the loaded settings on startup for verification
# settings = get_settings()
# logger.debug(f"Loaded settings: {settings.model_dump(exclude={'NEO4J_PASSWORD'})}")