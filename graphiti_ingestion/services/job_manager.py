# graphiti_ingestion/services/job_manager.py

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config import get_settings

logger = logging.getLogger(__name__)

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobManager:
    """
    Manages ingestion jobs using a file-based queue system for persistence.
    """
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.paths = {
            status: self.base_path / status for status in
            [JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.COMPLETED, JobStatus.FAILED]
        }
        self._create_directories()

    def _create_directories(self):
        """Ensures all necessary subdirectories exist."""
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
        
        # Use async file I/O for better performance
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, job_path.write_text, json.dumps(data, indent=2))
        await loop.run_in_executor(None, status_path.write_text, json.dumps(status_info, indent=2))
        logger.info(f"Submitted job {job_id} to the pending queue.")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Finds and reads the status file for a given job_id."""
        for status in self.paths.keys():
            _, status_path = self._get_job_paths(job_id, status)
            if status_path.exists():
                loop = asyncio.get_running_loop()
                content = await loop.run_in_executor(None, status_path.read_text)
                return json.loads(content)
        return None

    async def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Finds the oldest pending job, moves it to 'processing', and returns its data.
        Returns None if no pending jobs are found.
        """
        pending_path = self.paths[JobStatus.PENDING]
        
        # Find the oldest file to process first
        pending_jobs = sorted(pending_path.glob("*.json"), key=os.path.getctime)
        if not pending_jobs:
            return None

        job_path = pending_jobs[0]
        job_id = job_path.stem
        
        # Atomically move files to claim the job
        loop = asyncio.get_running_loop()
        processing_job_path, processing_status_path = self._get_job_paths(job_id, JobStatus.PROCESSING)
        pending_status_path = pending_path / f"{job_id}.status.json"

        await loop.run_in_executor(None, lambda: job_path.rename(processing_job_path))
        await loop.run_in_executor(None, lambda: pending_status_path.rename(processing_status_path))

        # Update status to 'processing'
        await self.update_job_status(job_id, JobStatus.PROCESSING, "Worker started processing job.")
        
        # Read and return job data
        content = await loop.run_in_executor(None, processing_job_path.read_text)
        logger.info(f"Moved job {job_id} to processing.")
        return job_id, json.loads(content)

    async def update_job_status(self, job_id: str, new_status: str, message: Optional[str] = None):
        """Moves job files to the new status directory and updates the status file."""
        current_status_info = await self.get_job_status(job_id)
        if not current_status_info:
            logger.error(f"Cannot update status for non-existent job {job_id}.")
            return

        current_status = current_status_info["status"]
        if current_status == new_status and new_status != JobStatus.PROCESSING:
             return # No move needed if status is the same (unless it's the initial processing update)

        # Move files if the status represents a directory change
        if current_status != new_status:
            current_job_path, current_status_path = self._get_job_paths(job_id, current_status)
            new_job_path, new_status_path = self._get_job_paths(job_id, new_status)
            
            loop = asyncio.get_running_loop()
            if current_job_path.exists():
                await loop.run_in_executor(None, lambda: current_job_path.rename(new_job_path))
            if current_status_path.exists():
                await loop.run_in_executor(None, lambda: current_status_path.rename(new_status_path))
        
        # Update the content of the status file
        _, final_status_path = self._get_job_paths(job_id, new_status)
        current_status_info["status"] = new_status
        current_status_info["last_updated"] = datetime.now(timezone.utc).isoformat()
        if message:
            current_status_info["message"] = message

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, final_status_path.write_text, json.dumps(current_status_info, indent=2))
        logger.info(f"Updated job {job_id} to status '{new_status}'.")


# --- Dependency Injection Singleton ---
_job_manager_instance: Optional[JobManager] = None

def get_job_manager() -> JobManager:
    """FastAPI dependency to get the singleton instance of the JobManager."""
    global _job_manager_instance
    if _job_manager_instance is None:
        settings = get_settings()
        _job_manager_instance = JobManager(base_path=settings.JOB_QUEUE_PATH)
    return _job_manager_instance