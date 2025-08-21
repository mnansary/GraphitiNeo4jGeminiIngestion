# bulk_ingest.py

import asyncio
import json
import logging
import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Tuple

import httpx

# --- Configuration ---
API_BASE_URL = "http://localhost:6000"  # Ensure this port matches your service
# How many seconds to wait between status checks for a single job.
POLL_INTERVAL_SECONDS = 5
# The name of the file where errors will be logged.
ERROR_LOG_FILE = "errors.csv"

# --- Logging Setup ---
# We'll use print statements for progress and logging for script-level events.
logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BulkIngestClient")


def prepare_payload_for_graphiti(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms the raw, nested JSON into a cleaner JSON object with an explicit
    hierarchy, optimized for LLM extraction in Graphiti.
    """
    title = data.get("service_title_bn", "")
    details = data.get("details", {}) or {}
    rules = details.get("application_rules_md")
    categories = data.get("categories", {}) or {}
    dept_hierarchy = categories.get("department_hierarchy") or []

    # Transform the hierarchy list into explicit, named keys
    ministry = dept_hierarchy[0] if len(dept_hierarchy) > 0 else ""
    department = dept_hierarchy[1] if len(dept_hierarchy) > 1 else ""
    directorate = dept_hierarchy[2] if len(dept_hierarchy) > 2 else ""

    clean_payload = {
        "service_id": data.get("service_id"),
        "service_title_bn": title,
        "service_url": data.get("service_url"),
        "ministry_bn": ministry,
        "department_bn": department,
        "directorate_bn": directorate,
        "service_sector_bn": categories.get("service_sector"),
        "recipient_type_bn": categories.get("recipient_type"),
        "service_class_bn": categories.get("service_class"),
        "application_rules_bn": rules,
    }
    return {k: v for k, v in clean_payload.items() if v is not None and v != ""}


async def process_single_file(
    client: httpx.AsyncClient, file_path: Path
) -> Tuple[str, str]:
    """
    Handles the entire lifecycle for one file: submit, poll until completion, and return status.

    Returns:
        A tuple of (status, message). status is 'completed' or 'failed'.
    """
    # 1. Read and prepare the file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        prepared_json_content = prepare_payload_for_graphiti(raw_data)
        service_title = prepared_json_content.get("service_title_bn", file_path.name)
        api_payload = {
            "content": prepared_json_content,
            "type": "json",
            "description": service_title,
        }
    except Exception as e:
        error_msg = f"Failed to read or parse file: {e}"
        logger.error(error_msg)
        return "failed", error_msg

    # 2. Submit the job to the API
    try:
        response = await client.post(f"{API_BASE_URL}/episodes/", json=api_payload, timeout=60.0)
        if response.status_code != 202:
            return "failed", f"API submission error. Status: {response.status_code}, Body: {response.text}"
        job_id = response.json().get("job_id")
    except httpx.RequestError as e:
        return "failed", f"API connection error during submission: {e}"

    # 3. Poll for the job's status until it's finished
    while True:
        try:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            status_response = await client.get(f"{API_BASE_URL}/episodes/status/{job_id}", timeout=30.0)
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                current_status = status_data.get("status")
                message = status_data.get("message", "")

                if current_status == "completed":
                    return "completed", message
                if current_status == "failed":
                    return "failed", message
                
                # If still pending or processing, the loop continues
                print(f"  ... Job {job_id} is {current_status}", end='\r')

            else:
                return "failed", f"Failed to get job status. HTTP {status_response.status_code}"

        except httpx.RequestError as e:
            return "failed", f"API connection error during polling: {e}"


def log_error_to_csv(filename: str, error_message: str):
    """Appends a single error record to the CSV log file."""
    # Sanitize the error message to remove newlines for clean CSV formatting
    clean_message = str(error_message).replace('\n', ' ').replace('\r', ' ')
    with open(ERROR_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([filename, clean_message])


async def main(folder_path: str):
    """
    Main function to find and process all JSON files sequentially.
    """
    logger.info(f"Starting synchronous bulk ingestion from folder: {folder_path}")
    source_path = Path(folder_path)

    if not source_path.is_dir():
        logger.critical(f"Error: Provided path '{folder_path}' is not a valid directory.")
        return

    json_files = sorted(list(source_path.glob("*.json")))
    if not json_files:
        logger.warning(f"No .json files found in '{folder_path}'. Exiting.")
        return
        
    total_files = len(json_files)
    logger.info(f"Found {total_files} JSON files to process.")

    # Prepare the error log file
    with open(ERROR_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "error_message"])
    logger.info(f"Errors will be logged to '{ERROR_LOG_FILE}'")

    success_count = 0
    failure_count = 0

    async with httpx.AsyncClient() as client:
        for i, file_path in enumerate(json_files):
            print("-" * 70)
            print(f"[{i+1}/{total_files}] Processing file: {file_path.name}")
            
            status, message = await process_single_file(client, file_path)

            if status == "completed":
                success_count += 1
                print(f"✅ SUCCESS: {file_path.name} ingested successfully.")
            else:
                failure_count += 1
                print(f"❌ FAILED: {file_path.name}. Reason: {message}")
                log_error_to_csv(file_path.name, message)

    print("\n" + "=" * 70)
    print("--- BULK INGESTION COMPLETE ---")
    print(f"Total files processed: {total_files}")
    print(f"✅ Successful ingestions: {success_count}")
    print(f"❌ Failed ingestions: {failure_count}")
    if failure_count > 0:
        print(f"Details for all failures have been logged in '{ERROR_LOG_FILE}'.")
    print("=" * 70)


if __name__ == "__main__":
    parser = ArgumentParser(description="Synchronously ingest a folder of JSON files and log errors.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing the JSON files.")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.folder_path))
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")