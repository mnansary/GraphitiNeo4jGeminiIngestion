# gemini_api_manager/core/__init__.py

"""
Initializes the core components package.

This file makes the main classes from the sub-modules available directly
at the package level, allowing for cleaner imports elsewhere in the project.

For example, instead of:
from gemini_api_manager.core.manager import ComprehensiveManager

You can simply use:
from gemini_api_manager.core import ComprehensiveManager
"""

from .manager import ComprehensiveManager
from .task_types import TaskType