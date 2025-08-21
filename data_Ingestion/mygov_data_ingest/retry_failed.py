# retry_failed.py
import asyncio
import csv
from argparse import ArgumentParser
from pathlib import Path

# We can reuse the main logic from the bulk_ingest script
from graphiti_ingest import (
    ERROR_LOG_FILE,
    main as run_ingestion,
    process_single_file,
    log_error_to_csv,
)
import httpx

async def retry_main(folder_path: str, error_file: str):
    """
    Reads the error CSV, identifies unique failed files, and re-runs ingestion for them.
    """
    source_path = Path(folder_path)
    error_log_path = Path(error_file)

    if not error_log_path.exists():
        print(f"Error: The error log file '{error_file}' was not found.")
        return

    failed_files = set()
    with open(error_log_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                failed_files.add(row[0])

    if not failed_files:
        print("No failed files found in the log. Nothing to retry.")
        return

    print(f"Found {len(failed_files)} unique files to retry.")

    # Prepare a new, temporary error log for this retry attempt
    retry_error_log = f"retry_{ERROR_LOG_FILE}"
    with open(retry_error_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "error_message"])
    print(f"Errors from this retry attempt will be logged to '{retry_error_log}'")

    success_count = 0
    failure_count = 0
    total_files = len(failed_files)

    async with httpx.AsyncClient() as client:
        for i, filename in enumerate(sorted(list(failed_files))):
            file_path = source_path / filename
            if not file_path.exists():
                print(f"⚠️ SKIPPING: {filename} not found in the source directory.")
                continue

            print("-" * 70)
            print(f"[{i+1}/{total_files}] Retrying file: {filename}")
            
            status, message = await process_single_file(client, file_path)

            if status == "completed":
                success_count += 1
                print(f"✅ SUCCESS: {filename} ingested successfully.")
            else:
                failure_count += 1
                print(f"❌ FAILED AGAIN: {filename}. Reason: {message}")
                # Use the original log_error_to_csv with the new file name
                with open(retry_error_log, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    clean_message = str(message).replace('\n', ' ').replace('\r', ' ')
                    writer.writerow([filename, clean_message])

    print("\n" + "=" * 70)
    print("--- RETRY COMPLETE ---")
    print(f"Total files retried: {total_files}")
    print(f"✅ Successful ingestions: {success_count}")
    print(f"❌ Failed ingestions: {failure_count}")
    if failure_count > 0:
        print(f"Details for remaining failures logged in '{retry_error_log}'.")
    print("=" * 70)


if __name__ == "__main__":
    parser = ArgumentParser(description="Retry failed ingestions from an error log.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing the original JSON files.")
    parser.add_argument("--error-file", type=str, default=ERROR_LOG_FILE, help="The CSV file containing the list of failed files.")
    args = parser.parse_args()
    
    asyncio.run(retry_main(args.folder_path, args.error_file))