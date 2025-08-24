# graphiti_ingestion/services/job_manager.py

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..api.dashboard_websockets import websocket_manager
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
    Manages ingestion jobs using a file-based queue for persistence.

    This class handles the job lifecycle by moving files between status
    directories. It now includes logic to track and manage job retries
    by creating and reading a `.retry.json` metadata file for each job.
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.paths = {
            status: self.base_path / status for status in
            [JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.COMPLETED, JobStatus.FAILED]
        }
        self._create_directories()

    def _create_directories(self) -> None:
        """Ensures all necessary status subdirectories exist on startup."""
        logger.info(f"Initializing job directories in: {self.base_path}")
        for path in self.paths.values():
            path.mkdir(exist_ok=True, parents=True)

    def _get_job_paths(self, job_id: str, status: str) -> Tuple[Path, Path, Path]:
        """
        Returns the data, status, and retry file paths for a job in a given state.
        """
        dir_path = self.paths[status]
        return (
            dir_path / f"{job_id}.json",
            dir_path / f"{job_id}.status.json",
            dir_path / f"{job_id}.retry.json",
        )

    def _broadcast_job_update(self, job_data: Dict[str, Any]) -> None:
        """Helper to send a real-time job update to all dashboard clients."""
        message = {"type": "job_update", "payload": job_data}
        websocket_manager.broadcast_threadsafe(json.dumps(message))
        logger.debug(f"Broadcasted job update for {job_data.get('job_id')}")

    async def submit_job(self, job_id: str, data: Dict[str, Any]) -> None:
        """Saves a new job to the 'pending' directory and notifies the dashboard."""
        job_path, status_path, _ = self._get_job_paths(job_id, JobStatus.PENDING)

        status_info = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Job is waiting in the queue.",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "retry_count": 0,
        }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, job_path.write_text, json.dumps(data, indent=2, ensure_ascii=False))
        await loop.run_in_executor(None, status_path.write_text, json.dumps(status_info, indent=2, ensure_ascii=False))
        
        logger.info(f"Submitted job {job_id} to the pending queue.")
        self._broadcast_job_update(status_info)

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Finds and reads the status file for a single job by its ID."""
        for status in self.paths.keys():
            _, status_path, _ = self._get_job_paths(job_id, status)
            if status_path.exists():
                loop = asyncio.get_running_loop()
                content = await loop.run_in_executor(None, status_path.read_text)
                return json.loads(content)
        return None

    async def get_all_job_statuses(self) -> List[Dict[str, Any]]:
        """Scans all directories for the dashboard's initial data load."""
        all_jobs = []
        loop = asyncio.get_running_loop()

        for path in self.paths.values():
            for status_file in path.glob("*.status.json"):
                try:
                    content = await loop.run_in_executor(None, status_file.read_text)
                    job_data = json.loads(content)
                    if job_data.get("status") == JobStatus.COMPLETED:
                        submitted = datetime.fromisoformat(job_data["submitted_at"])
                        completed = datetime.fromisoformat(job_data["last_updated"])
                        job_data["processing_time_seconds"] = round((completed - submitted).total_seconds(), 2)
                    all_jobs.append(job_data)
                except Exception as e:
                    logger.error(f"Could not parse status file {status_file}: {e}")

        all_jobs.sort(key=lambda j: j.get("submitted_at", ""), reverse=True)
        return all_jobs

    async def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any], int]]:
        """
        Finds the oldest pending job, moves it to 'processing', and returns its data.
        Prioritizes non-retried jobs over retried ones.
        """
        pending_path = self.paths[JobStatus.PENDING]
        job_files = [f for f in pending_path.glob("*.json") if ".status" not in f.name]
        if not job_files:
            return None

        non_retried = [p for p in job_files if not (p.parent / f"{p.stem}.retry.json").exists()]
        target_path = min(non_retried, key=os.path.getctime) if non_retried else min(job_files, key=os.path.getctime)
        job_id = target_path.stem

        loop = asyncio.get_running_loop()
        _, _, pending_retry_path = self._get_job_paths(job_id, JobStatus.PENDING)
        retry_count = 0
        if pending_retry_path.exists():
            try:
                content = await loop.run_in_executor(None, pending_retry_path.read_text)
                retry_count = json.loads(content).get("retry_count", 0)
            except Exception:
                logger.error(f"Could not read retry file for {job_id}, assuming 0.", exc_info=True)

        processing_job_path, proc_status_path, proc_retry_path = self._get_job_paths(job_id, JobStatus.PROCESSING)
        try:
            await loop.run_in_executor(None, lambda: target_path.rename(processing_job_path))
            # Move associated metadata files
            for src_suffix, dest_path in [
                (".status.json", proc_status_path),
                (".retry.json", proc_retry_path)
            ]:
                src_path = pending_path / (job_id + src_suffix)
                if src_path.exists():
                    await loop.run_in_executor(None, lambda: src_path.rename(dest_path))
        except FileNotFoundError:
            logger.warning(f"Job {job_id} was moved by another process. Skipping.")
            return None

        await self.update_job_status(job_id, JobStatus.PROCESSING, f"Worker processing (Attempt #{retry_count + 1}).", retry_count)
        content = await loop.run_in_executor(None, processing_job_path.read_text)
        return job_id, json.loads(content), retry_count

    async def requeue_job_for_retry(self, job_id: str, new_retry_count: int, message: str) -> None:
        """Moves a job back to the pending queue and updates its retry count."""
        status_info = await self.get_job_status(job_id)
        if not status_info:
            return
        
        current_status = status_info["status"]
        paths = {s: self._get_job_paths(job_id, s) for s in [current_status, JobStatus.PENDING]}
        loop = asyncio.get_running_loop()

        for i in range(3): # Move data, status, and retry files
            if paths[current_status][i].exists():
                await loop.run_in_executor(None, lambda: paths[current_status][i].rename(paths[JobStatus.PENDING][i]))
        
        retry_data = {"retry_count": new_retry_count, "last_failure_reason": message}
        await loop.run_in_executor(None, paths[JobStatus.PENDING][2].write_text, json.dumps(retry_data))
        
        await self.update_job_status(job_id, JobStatus.PENDING, message, new_retry_count)

    async def update_job_status(
        self, job_id: str, new_status: str, message: Optional[str] = None, retry_count: Optional[int] = None
    ) -> None:
        """Updates a job's status file, moves its files, and notifies the dashboard."""
        status_info = await self.get_job_status(job_id)
        if not status_info:
            return

        current_status = status_info["status"]
        if current_status != new_status:
            paths = {s: self._get_job_paths(job_id, s) for s in [current_status, new_status]}
            loop = asyncio.get_running_loop()
            for i in range(3):
                 if paths[current_status][i].exists():
                    await loop.run_in_executor(None, lambda: paths[current_status][i].rename(paths[new_status][i]))

        _, final_status_path, _ = self._get_job_paths(job_id, new_status)
        status_info.update({
            "status": new_status,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "message": message or status_info["message"],
            "retry_count": retry_count if retry_count is not None else status_info.get("retry_count", 0)
        })

        if new_status == JobStatus.COMPLETED:
            submitted = datetime.fromisoformat(status_info["submitted_at"])
            completed = datetime.fromisoformat(status_info["last_updated"])
            status_info["processing_time_seconds"] = round((completed - submitted).total_seconds(), 2)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, final_status_path.write_text, json.dumps(status_info, indent=2, ensure_ascii=False))
        
        logger.info(f"Updated job {job_id} to status '{new_status}'.")
        self._broadcast_job_update(status_info)


# --- Dependency Injection Singleton ---
_job_manager_instance: Optional[JobManager] = None

def get_job_manager() -> JobManager:
    """FastAPI dependency to get the singleton JobManager instance."""
    global _job_manager_instance
    if _job_manager_instance is None:
        logger.info("Creating singleton instance of JobManager.")
        settings = get_settings()
        _job_manager_instance = JobManager(base_path=settings.JOB_QUEUE_PATH)
    return _job_manager_instance