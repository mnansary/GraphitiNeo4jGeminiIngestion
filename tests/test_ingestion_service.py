import asyncio
import logging

# Ensure all services (Neo4j, vLLM, Triton) are running before executing this script.
# This script must be run from the root of your project where the .env file is located.

# --- Step 1: Import necessary components ---
from graphiti_ingestion.services.graphiti_service import (
    initialize_graphiti_service,
    get_graphiti_service,
)
from graphiti_ingestion.config import get_settings


async def main():
    """
    A standalone script to test the core functionality of the GraphitiService.
    """
    # --- Step 2: Configure logging and load settings ---
    settings = get_settings()
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("IntegrationTest")
    
    graphiti_service = None
    try:
        # --- Step 3: Initialize the service (mimics application startup) ---
        logger.info("Initializing Graphiti Service for the test...")
        # This function creates the singleton instance and stores it
        initialize_graphiti_service()
        # We retrieve the instance to use it
        graphiti_service = get_graphiti_service()
        
        # This builds the necessary indices and constraints in Neo4j
        await graphiti_service.startup()
        logger.info("Service startup complete. Indices should be ready.")

        # --- Step 4: Define dummy episode data ---
        text_episode = {
            "content": "The Eiffel Tower, located in Paris, was completed in 1889.",
            "type": "text",
            "description": "Historical fact from a test script.",
        }

        json_episode = {
            "content": {
                "landmark": "Eiffel Tower",
                "city": "Paris",
                "country": "France",
                "year_completed": 1889,
            },
            "type": "json",
            "description": "Structured data from a test script.",
        }

        # --- Step 5: Run the ingestion logic ---
        logger.info("--- Testing TEXT episode ingestion ---")
        await graphiti_service.process_and_add_episode(text_episode)
        logger.info("✅ Text episode ingestion test PASSED.")
        
        # Add a small delay if needed, though usually not necessary
        await asyncio.sleep(1)

        logger.info("--- Testing JSON episode ingestion ---")
        await graphiti_service.process_and_add_episode(json_episode)
        logger.info("✅ JSON episode ingestion test PASSED.")
        
        logger.info("\nIntegration test completed successfully!")
        logger.info("Check your Neo4j database to verify that the nodes and relationships for the 'Eiffel Tower' have been created.")

    except Exception as e:
        logger.critical(f"An error occurred during the integration test: {e}", exc_info=True)
    finally:
        # --- Step 6: Cleanly shut down the service ---
        if graphiti_service:
            logger.info("Shutting down the Graphiti Service...")
            await graphiti_service.shutdown()
            logger.info("Service shutdown complete.")


if __name__ == "__main__":
    # Ensure you have a .env file in the same directory where you run this script
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")