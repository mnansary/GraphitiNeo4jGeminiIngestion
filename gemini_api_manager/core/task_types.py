# gemini_api_manager/core/task_types.py

from enum import Enum, auto

class TaskType(Enum):
    """
    Represents high-level task categories based on input/output modalities.

    This enumeration provides a type-safe way for services to request a specific
    capability from the ComprehensiveManager. Each member corresponds directly to a
    top-level key in the 'tasks' section of the model_config.yaml file.
    
    Using this enum prevents errors from typos when requesting a client and
    makes the service-level code more explicit and readable.
    """
    
    # Represents tasks that primarily involve text input to generate text output.
    # Examples: Summarization, Q&A, translation, creative writing.
    TEXT_TO_TEXT = auto()
    
    # Represents tasks that take multiple input types (e.g., image, audio, video)
    # along with a potential text prompt to generate a text output.
    # Examples: Describing an image, answering questions about a video.
    MULTIMODAL_TO_TEXT = auto()
    
    # Represents tasks for generating speech from text (Text-to-Speech).
    TEXT_TO_AUDIO = auto()
    
    # Represents tasks dedicated to generating an image from a text prompt.
    IMAGE_GENERATION = auto()