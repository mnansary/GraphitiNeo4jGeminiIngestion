# graphiti_ingestion/api/episodes.py

import uuid
from fastapi import APIRouter, Depends, HTTPException, status

from ..models.episodes import (
    EpisodeRequest,
    EpisodeResponse,
    JobStatusResponse,
)
from ..services.job_manager import JobManager, get_job_manager

router = APIRouter(
    prefix="/episodes",
    tags=["Episodes"],
    responses={
        404: {"description": "Job not found"},
        500: {"description": "Internal server error"}
    },
)


@router.post(
    "/",
    response_model=EpisodeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an episode for ingestion",
    description=(
        "Accepts episode content, type, and description. The request is "
        "persistently saved to a file-based queue for background processing. "
        "A unique job ID is returned immediately to the client for status tracking."
    ),
)
async def submit_episode(
    episode_request: EpisodeRequest,
    job_manager: JobManager = Depends(get_job_manager),
) -> EpisodeResponse:
    """
    Submits a new episode to the persistent ingestion queue.

    This endpoint is non-blocking. It generates a unique job ID, saves the
    request payload as a JSON file in the 'pending' queue directory, and
    immediately returns the job ID.

    Args:
        episode_request: The Pydantic model containing the episode data from the client.
        job_manager: The dependency-injected job manager service that handles
                     the file-based queue.

    Returns:
        An EpisodeResponse containing the generated job_id and initial status.
    """
    job_id = str(uuid.uuid4())
    await job_manager.submit_job(job_id, episode_request.model_dump())

    return EpisodeResponse(
        job_id=job_id,
        status="pending",
        message="Episode accepted for processing and saved to disk.",
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check the status of an ingestion job",
    description="Retrieves the current processing status of a previously submitted episode by querying the file-based queue.",
)
async def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobStatusResponse:
    """
    Retrieves the status of a specific ingestion job by its ID.

    The job manager locates the job's status file across all state directories
    (pending, processing, completed, failed) to provide the most current information.

    Args:
        job_id: The unique identifier for the job, provided upon submission.
        job_manager: The dependency-injected job manager service.

    Returns:
        The current status of the job, including a descriptive message.

    Raises:
        HTTPException (404 Not Found): If a job with the specified ID cannot be found.
    """

    status_info = await job_manager.get_job_status(job_id)

    if status_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found.",
        )

    return JobStatusResponse(**status_info)