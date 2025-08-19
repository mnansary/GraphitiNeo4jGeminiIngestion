from enum import Enum
from typing import Dict, Union, Optional

from pydantic import BaseModel, Field


class EpisodeContentType(str, Enum):
    """Enumeration for the type of episode content."""
    TEXT = "text"
    JSON = "json"


class EpisodeRequest(BaseModel):
    """
    Defines the structure for an incoming episode ingestion request.
    """
    content: Union[str, Dict] = Field(
        ...,  # ... means this field is required
        description="The main content of the episode. Can be a string for text episodes or a JSON object for structured data.",
        examples=[
            "Kamala Harris was the Attorney General of California.",
            {"name": "Gavin Newsom", "position": "Governor"},
        ],
    )
    type: EpisodeContentType = Field(
        ...,
        description="The type of the content, either 'text' or 'json'.",
        examples=["text"],
    )
    description: str = Field(
        ...,
        description="A brief description of the episode's source or context.",
        examples=["podcast transcript"],
    )


class EpisodeResponse(BaseModel):
    """

    Defines the response sent back to the client after successfully submitting an episode.
    """
    job_id: str = Field(description="The unique identifier for the processing job.")
    status: str = Field(description="The initial status of the job, typically 'pending'.")
    message: str = Field(description="A confirmation message.")


class JobStatusResponse(BaseModel):
    """
    Defines the structure for a job status query response.
    """
    job_id: str = Field(description="The unique identifier for the processing job.")
    status: str = Field(
        description="The current status of the job (e.g., pending, processing, completed, failed)."
    )
    message: Optional[str] = Field(
        None,
        description="An optional message, often used to provide details on failures."
    )