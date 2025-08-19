import uuid
from fastapi import APIRouter, Depends, HTTPException, status

# The imports now correctly reference the 'graphiti_ingestion' package.
# These files will be created in the upcoming steps.
from graphiti_ingestion.models.episodes import (
    EpisodeRequest,
    EpisodeResponse,
    JobStatusResponse,
)
from graphiti_ingestion.services.task_queue import TaskQueue, get_task_queue

router = APIRouter(
    prefix="/episodes",
    tags=["Episodes"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/",
    response_model=EpisodeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an episode for ingestion",
    description="Accepts an episode's content, type, and description, then adds it to a background processing queue.",
)
async def submit_episode(
    episode_request: EpisodeRequest,
    task_queue: TaskQueue = Depends(get_task_queue),
):
    """
    Submits a new episode to the ingestion queue.

    This endpoint is non-blocking. It generates a unique job ID for the request,
    places it in the queue for background processing, and immediately returns the
    job ID to the client.

    Args:
        episode_request: The episode data payload from the client.
        task_queue: The dependency-injected task queue service.

    Returns:
        An EpisodeResponse containing the job_id and initial status.
    """
    job_id = str(uuid.uuid4())
    await task_queue.submit_job(job_id, episode_request.model_dump())

    return EpisodeResponse(
        job_id=job_id,
        status="pending",
        message="Episode accepted for processing.",
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check the status of an ingestion job",
    description="Retrieves the current processing status of an episode submitted previously.",
)
async def get_job_status(
    job_id: str,
    task_queue: TaskQueue = Depends(get_task_queue),
):
    """
    Retrieves the status of a specific ingestion job by its ID.

    Args:
        job_id: The unique identifier for the job.
        task_queue: The dependency-injected task queue service.

    Returns:
        The current status of the job (pending, processing, completed, or failed).
    
    Raises:
        HTTPException: If the job_id is not found.
    """
    status_info = await task_queue.get_job_status(job_id)

    if status_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found.",
        )

    return JobStatusResponse(**status_info)