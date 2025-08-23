# graphiti_ingestion/services/job_manager.py

import asyncio
import json
import logging
import os
# --- CORRECTED IMPORT ---
# We only import the main classes/modules we need.
from datetime import datetime, timezone
# --- END CORRECTION ---
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import get_settings

logger = logging.getLogger(__name__)


class JobStatus:
    """Provides consistent string constants for job statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    """
    Manages ingestion jobs using a file-based queue system for persistence.
    This provides robustness and introspection capabilities for a dashboard.
    """
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.paths = {
            status: self.base_path / status for status in
            [JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.COMPLETED, JobStatus.FAILED]
        }
        self._create_directories()

    def _create_directories(self):
        """Ensures all necessary status subdirectories exist on startup."""
        logger.info(f"Initializing job directories in: {self.base_path}")
        for path in self.paths.values():
            path.mkdir(exist_ok=True, parents=True)

    def _get_job_paths(self, job_id: str, status: str) -> Tuple[Path, Path]:
        """Returns the data and status file paths for a given job and status."""
        dir_path = self.paths[status]
        return dir_path / f"{job_id}.json", dir_path / f"{job_id}.status.json"

    async def submit_job(self, job_id: str, data: Dict[str, Any]):
        """Saves a new job to the 'pending' directory."""
        job_path, status_path = self._get_job_paths(job_id, JobStatus.PENDING)

        status_info = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Job is waiting in the queue.",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, job_path.write_text, json.dumps(data, indent=2, ensure_ascii=False))
        await loop.run_in_executor(None, status_path.write_text, json.dumps(status_info, indent=2, ensure_ascii=False))
        logger.info(f"Submitted job {job_id} to the pending queue.")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Finds and reads the status file for a single job by its ID across all directories.
        """
        for status in self.paths.keys():
            _, status_path = self._get_job_paths(job_id, status)
            if status_path.exists():
                loop = asyncio.get_running_loop()
                content = await loop.run_in_executor(None, status_path.read_text)
                return json.loads(content)
        return None

    async def get_all_job_statuses(self) -> List[Dict[str, Any]]:
        """
        Scans all status directories, reads all status files, and returns a
        comprehensive list of all jobs and their current states.
        """
        all_jobs = []
        loop = asyncio.get_running_loop()

        for status, path in self.paths.items():
            status_files = path.glob("*.status.json")
            for status_file in status_files:
                try:
                    content = await loop.run_in_executor(None, status_file.read_text)
                    job_data = json.loads(content)

                    if job_data.get("status") == JobStatus.COMPLETED:
                        # --- CORRECTED METHOD CALL ---
                        # We call fromisoformat() on the datetime class itself.
                        submitted = datetime.fromisoformat(job_data["submitted_at"])
                        completed = datetime.fromisoformat(job_data["last_updated"])
                        # --- END CORRECTION ---
                        duration = (completed - submitted).total_seconds()
                        job_data["processing_time_seconds"] = round(duration, 2)

                    all_jobs.append(job_data)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Could not parse status file {status_file}: {e}")

        all_jobs.sort(key=lambda j: j.get("submitted_at", ""), reverse=True)
        return all_jobs

    async def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Finds the oldest pending job based on file creation time, moves it to
        'processing' to claim it, and returns its data for the worker.
        """
        pending_path = self.paths[JobStatus.PENDING]
        pending_job_files = [f for f in pending_path.glob("*.json") if ".status" not in f.name]
        if not pending_job_files:
            return None

        oldest_job_path = min(pending_job_files, key=os.path.getctime)
        job_id = oldest_job_path.stem

        loop = asyncio.get_running_loop()
        processing_job_path, processing_status_path = self._get_job_paths(job_id, JobStatus.PROCESSING)
        pending_status_path = pending_path / f"{job_id}.status.json"

        try:
            await loop.run_in_executor(None, lambda: oldest_job_path.rename(processing_job_path))
            if pending_status_path.exists():
                await loop.run_in_executor(None, lambda: pending_status_path.rename(processing_status_path))
        except FileNotFoundError:
            logger.warning(f"Job {job_id} was moved by another process before this worker could claim it.")
            return None

        await self.update_job_status(job_id, JobStatus.PROCESSING, "Worker started processing job.")
        content = await loop.run_in_executor(None, processing_job_path.read_text)
        logger.info(f"Moved job {job_id} to processing.")
        return job_id, json.loads(content)

    async def update_job_status(self, job_id: str, new_status: str, message: Optional[str] = None):
        """
        Updates a job's status by moving its files and updating its status file content.
        """
        current_status_info = await self.get_job_status(job_id)
        if not current_status_info:
            logger.error(f"Cannot update status for non-existent job {job_id}.")
            return

        current_status = current_status_info["status"]
        if current_status != new_status:
            current_job_path, current_status_path = self._get_job_paths(job_id, current_status)
            new_job_path, new_status_path = self._get_job_paths(job_id, new_status)
            
            loop = asyncio.get_running_loop()
            if current_job_path.exists():
                await loop.run_in_executor(None, lambda: current_job_path.rename(new_job_path))
            if current_status_path.exists():
                await loop.run_in_executor(None, lambda: current_status_path.rename(new_status_path))

        _, final_status_path = self._get_job_paths(job_id, new_status)
        current_status_info["status"] = new_status
        current_status_info["last_updated"] = datetime.now(timezone.utc).isoformat()
        if message:
            current_status_info["message"] = message

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, final_status_path.write_text, json.dumps(current_status_info, indent=2, ensure_ascii=False))
        logger.info(f"Updated job {job_id} to status '{new_status}'.")


# --- Dependency Injection Singleton ---
_job_manager_instance: Optional[JobManager] = None

def get_job_manager() -> JobManager:
    """FastAPI dependency to get the singleton instance of the JobManager."""
    global _job_manager_instance
    if _job_manager_instance is None:
        logger.info("Creating singleton instance of JobManager.")
        settings = get_settings()
        _job_manager_instance = JobManager(base_path=settings.JOB_QUEUE_PATH)
    return _job_manager_instance