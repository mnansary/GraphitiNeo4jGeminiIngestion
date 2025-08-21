# tests/test_api_client.py
import asyncio
import httpx
import logging

# Configure basic logging
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ApiClientTest")

API_BASE_URL = "http://localhost:6000"

text_episode = {
    "content": "The Eiffel Tower, located in Paris, was completed in 1889.",
    "type": "text",
    "description": "Historical fact from a test client script.",
}

json_episode = {
    "content": {
        "landmark": "Eiffel Tower",
        "city": "Paris",
        "country": "France",
        "year_completed": 1889,
    },
    "type": "json",
    "description": "Structured data from a test client script.",
}

async def test_episode_ingestion(client: httpx.AsyncClient, episode_data: dict, episode_name: str):
    """Helper function to submit an episode and poll for its completion status."""
    logger.info(f"--- Submitting {episode_name} episode ---")
    
    # 1. Submit the job
    response = await client.post(f"{API_BASE_URL}/episodes/", json=episode_data)
    
    if response.status_code != 202:
        logger.error(f"Failed to submit job. Status: {response.status_code}, Body: {response.text}")
        return

    response_data = response.json()
    job_id = response_data.get("job_id")
    logger.info(f"Successfully submitted job with ID: {job_id}")

    # 2. Poll for status until completed or failed
    while True:
        await asyncio.sleep(5)  # Wait 5 seconds between checks
        logger.info(f"Polling status for job {job_id}...")
        status_response = await client.get(f"{API_BASE_URL}/episodes/status/{job_id}")
        
        if status_response.status_code != 200:
            logger.error(f"Failed to get job status. Status: {status_response.status_code}")
            break
        
        status_data = status_response.json()
        current_status = status_data.get("status")
        message = status_data.get("message")
        logger.info(f"Current status for {job_id}: {current_status} - {message}")

        if current_status in ["completed", "failed"]:
            if current_status == "completed":
                logger.info(f"✅ {episode_name} episode ingestion test PASSED.")
            else:
                logger.error(f"❌ {episode_name} episode ingestion test FAILED.")
            break

async def main():
    """Main function to run the API tests."""
    async with httpx.AsyncClient() as client:
        await test_episode_ingestion(client, text_episode, "TEXT")
        await test_episode_ingestion(client, json_episode, "JSON")
    
    logger.info("\nAPI client test completed successfully!")
    logger.info("Check your Neo4j database and the API server logs for details.")


if __name__ == "__main__":
    asyncio.run(main())