# tests/test_api_ingestion.py

import asyncio
import httpx
import logging
from typing import Dict

# --- Configuration ---
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ApiJobSubmitter")

# The API URL must point to your public Nginx server and the correct location block.
API_BASE_URL = "http://localhost:6000"

# --- Test Data Payloads ---
text_episode = {
    "content": "The Statue of Liberty was a gift to the United States from the people of France in 1886. It was designed by French sculptor Frédéric Auguste Bartholdi.",
    "type": "text",
    "description": "API Test: Historical fact about the Statue of Liberty.",
}

json_episode = {
    "content": {
        "event": "Completion of the Statue of Liberty",
        "location": "New York Harbor, USA",
        "origin_country": "France",
        "year": 1886,
        "key_figures": ["Frédéric Auguste Bartholdi", "Gustave Eiffel"],
    },
    "type": "json",
    "description": "API Test: Structured data about the Statue of Liberty.",
}


async def submit_ingestion_job(
    client: httpx.AsyncClient, episode_data: Dict, test_name: str
):
    """
    Handles submitting a single ingestion job to the API endpoint.
    This function does NOT wait for the job to complete.
    """
    logger.info(f"--- [SUBMITTING] Job: '{test_name}' ---")

    try:
        # Submit the job to the ingestion service
        submit_response = await client.post(
            f"{API_BASE_URL}/episodes/", json=episode_data, timeout=10.0
        )

        # Check if the submission was accepted by the server
        if submit_response.status_code == 202:
            response_data = submit_response.json()
            job_id = response_data.get("job_id")
            logger.info(f"✅ [{test_name}] SUBMITTED SUCCESSFULLY. Job ID: {job_id}")
            logger.info(f"   ---> Monitor the dashboard for its progress from 'pending' to 'completed'.")
        else:
            logger.error(
                f"❌ [{test_name}] FAILED to submit job. "
                f"Status: {submit_response.status_code}, Body: {submit_response.text}"
            )

    except httpx.RequestError as e:
        logger.error(f"❌ [{test_name}] FAILED during submission. Connection error: {e}")


async def main():
    """
    Main function to orchestrate and run all API job submissions concurrently.
    """
    print("\n" + "="*80)
    print("🚀 STARTING INGESTION JOB SUBMITTER 🚀")
    print("="*80)
    print("Instructions:")
    print("1. This script will fire multiple jobs at the API endpoint.")
    print("2. It does NOT wait for them to complete.")
    print(f"3. Your job is to watch the dashboard at https://114.130.116.79/ingestion/dashboard/ in real-time!")
    print("   You should see jobs instantly appear in the 'Pending' column.")
    print("-" * 80)

    # The verify=False is needed for self-signed SSL certificates.
    async with httpx.AsyncClient(verify=False) as client:
        # Create a list of submission tasks to run concurrently
        tasks = [
            submit_ingestion_job(client, text_episode, "Text Episode Ingestion"),
            submit_ingestion_job(client, json_episode, "JSON Episode Ingestion"),
            # You can add more jobs here to test higher loads
            # submit_ingestion_job(client, text_episode, "Text Episode Ingestion #2"),
        ]
        await asyncio.gather(*tasks)

    print("\n" + "="*80)
    print("✅ SUBMISSION COMPLETE ✅")
    print("="*80)
    print("All jobs have been sent to the server. Please check your dashboard to monitor their progress.")
    print("")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nJob submission interrupted by user.")