import asyncio
from typing import Any, Dict, Optional


class TaskQueue:
    """
    A simple in-memory, asynchronous task queue and status tracker.
    
    This class manages a queue of jobs to be processed and stores the status
    of each job.
    """

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.job_statuses: Dict[str, Dict[str, Any]] = {}

    async def submit_job(self, job_id: str, data: Dict[str, Any]):
        """
        Adds a new job to the queue and sets its initial status.

        Args:
            job_id: A unique identifier for the job.
            data: The payload of the job to be processed.
        """
        initial_status = {
            "job_id": job_id,
            "status": "pending",
            "message": "Job is waiting in the queue.",
        }
        self.job_statuses[job_id] = initial_status
        await self.queue.put({"job_id": job_id, "data": data})

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the status of a specific job.

        Args:
            job_id: The unique identifier for the job.

        Returns:
            A dictionary with the job's status, or None if the job_id is not found.
        """
        return self.job_statuses.get(job_id)

    async def get_job(self) -> Dict[str, Any]:
        """
        Waits for and retrieves the next job from the queue.
        This is typically called by a background worker.
        """
        return await self.queue.get()

    async def update_job_status(self, job_id: str, status: str, message: Optional[str] = None):
        """
        Updates the status and message of a job.

        Args:
            job_id: The unique identifier for the job.
            status: The new status (e.g., "processing", "completed", "failed").
            message: An optional message, useful for error details.
        """
        if job_id in self.job_statuses:
            self.job_statuses[job_id]["status"] = status
            self.job_statuses[job_id]["message"] = message

    def mark_task_done(self):
        """
        Signals that a formerly enqueued task is complete.
        Called by the worker after processing a job.
        """
        self.queue.task_done()


# --- Dependency Injection Singleton ---
# This ensures that the entire application uses the same instance of TaskQueue.
_task_queue_instance = TaskQueue()


def get_task_queue() -> TaskQueue:
    """
    FastAPI dependency to get the singleton instance of the TaskQueue.
    """
    return _task_queue_instance