# validate_json.py

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

from rich.console import Console
from tqdm import tqdm

# --- Configuration ---
# Use rich for nice, colored terminal output
console = Console()

# --- Logging Setup ---
# We'll use print for immediate feedback and logging for script-level events.
logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JsonValidator")


def validate_single_file(file_path: Path) -> Tuple[bool, str]:
    """
    Performs validation checks on a single JSON file.

    Checks:
    1. If the file is empty (0 bytes).
    2. If the file content is valid, parsable JSON.

    Returns:
        A tuple of (is_valid: bool, reason: str).
    """
    # Check 1: Is the file empty?
    if file_path.stat().st_size == 0:
        return False, "File is empty (0 bytes)"

    # Check 2: Can the file be parsed as JSON?
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        # If we reach here, the file is valid
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {e}"
    except Exception as e:
        return False, f"An unexpected error occurred while reading: {e}"


def main(folder_path: str, log_invalid_path: str = None):
    """
    Main function to find and validate all JSON files in a directory.
    """
    console.print(f"[bold blue]Starting JSON validation for folder:[/] {folder_path}\n")
    source_path = Path(folder_path)

    if not source_path.is_dir():
        console.print(f"[bold red]Error:[/] Provided path '{folder_path}' is not a valid directory.")
        return

    json_files = sorted(list(source_path.glob("*.json")))
    if not json_files:
        console.print(f"[yellow]Warning:[/] No .json files found in '{folder_path}'. Exiting.")
        return

    total_files = len(json_files)
    invalid_files: List[Tuple[Path, str]] = []
    valid_count = 0

    # Use tqdm for a progress bar
    for file_path in tqdm(json_files, desc="Validating files"):
        is_valid, reason = validate_single_file(file_path)
        if not is_valid:
            invalid_files.append((file_path, reason))
    
    valid_count = total_files - len(invalid_files)

    # --- Print Final Report ---
    console.print("\n" + "=" * 70)
    console.print("[bold green]Validation Complete![/]")
    console.print("-" * 70)
    console.print(f"Total files scanned: {total_files}")
    console.print(f"[green]✅ Valid files: {valid_count}[/]")
    console.print(f"[red]❌ Invalid files: {len(invalid_files)}[/]")
    console.print("=" * 70)

    if invalid_files:
        console.print("\n[bold yellow]Details of invalid files:[/]")
        for file, reason in invalid_files:
            console.print(f"- [cyan]{file.name}[/]: [red]{reason}[/]")

        # Log invalid files to a file if requested
        if log_invalid_path:
            console.print(f"\n[bold blue]Logging invalid file paths to '{log_invalid_path}'...[/]")
            try:
                with open(log_invalid_path, "w", encoding="utf-8") as f:
                    for file, reason in invalid_files:
                        f.write(f"{file.resolve()}\n")
                console.print("[green]Successfully wrote to log file.[/]")
            except IOError as e:
                console.print(f"[bold red]Error writing to log file: {e}[/]")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Validate all JSON files in a directory to check for emptiness or format errors."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder containing the JSON files.",
    )
    parser.add_argument(
        "--log-invalid",
        type=str,
        nargs='?', # Makes the argument optional
        const="invalid_files.txt", # Default value if flag is present but no value given
        default=None, # Value if the flag is not present at all
        help="If provided, saves the full paths of all invalid files to the specified file (default: invalid_files.txt).",
    )
    args = parser.parse_args()

    main(args.folder_path, args.log_invalid)