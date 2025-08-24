# graphiti_ingestion/gemini/manager.py

from __future__ import annotations

import csv
import logging
import threading
import time
import yaml
from collections import defaultdict
from enum import Enum, auto
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """
    Represents high-level task categories based on input/output modalities.

    This enumeration provides a type-safe way for services to request a specific
    capability from the ComprehensiveManager, preventing errors from typos.
    """
    TEXT_TO_TEXT = auto()
    MULTIMODAL_TO_TEXT = auto()
    TEXT_TO_AUDIO = auto()
    IMAGE_GENERATION = auto()


class ComprehensiveManager:
    """
    Central controller for Gemini API usage.

    This thread-safe class handles:
      - Loading and cycling through multiple API keys from a CSV.
      - Enforcing a cooldown period on keys after use.
      - Loading model configurations and capabilities from a YAML file.
      - Intelligently selecting models for tasks, with a special mode to
        force the most capable model for retry attempts.
    """

    def __init__(
        self,
        api_key_csv_path: str,
        model_config_path: str,
        api_key_cooldown_seconds: float = 60.0,
    ):
        self._lock = threading.Lock()
        self.api_keys: List[str] = self._load_api_keys(api_key_csv_path)
        self.models_config: Dict[str, Any] = self._load_model_config(model_config_path)
        self.api_key_cooldown_seconds: float = api_key_cooldown_seconds

        # Cooldown tracking for API keys
        self.cooldowns: Dict[str, float] = {}

        # Round-robin generator for cycling through (api_key, model_name) pairs
        self._client_cycle: Dict[TaskType, Iterator] = {}
        self._init_client_generators()

    def _load_api_keys(self, csv_path: str) -> List[str]:
        """Loads API keys from a CSV file with an 'api' header."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"API key CSV not found at: {csv_path}")

        keys: List[str] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "api" not in (reader.fieldnames or []):
                raise ValueError(f"CSV file '{csv_path}' must have an 'api' header.")
            for row in reader:
                key = row.get("api", "").strip()
                if key:
                    keys.append(key)
        if not keys:
            raise ValueError(f"No API keys found in '{csv_path}'.")
        logger.info(f"Loaded {len(keys)} API keys.")
        return keys

    def _load_model_config(self, yaml_path: str) -> Dict[str, Any]:
        """Loads model definitions and task mappings from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Model config YAML not found at: {yaml_path}")
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Model configuration loaded successfully.")
        return config

    def _init_client_generators(self) -> None:
        """Pre-builds the round-robin iterators for each TaskType."""
        tasks = self.models_config.get("tasks", {})
        for task_name, task_info in tasks.items():
            try:
                task_enum = TaskType[task_name]
                models = task_info.get("models", [])
                if models:
                    clients = [(k, m) for k in self.api_keys for m in models]
                    self._client_cycle[task_enum] = cycle(clients)
            except KeyError:
                logger.warning(f"Task type '{task_name}' in YAML config is not a valid TaskType enum member.")

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full configuration dictionary for a specific model.
        
        Args:
            model_name: The name of the model (e.g., 'gemini-2.5-flash').

        Returns:
            A dictionary of the model's configuration or None if not found.
        """
        return self.models_config.get("models", {}).get(model_name)

    def get_available_client_details(
        self,
        task_type: TaskType,
        force_best_model: bool = False,
    ) -> Iterator[Tuple[str, str]]:
        """
        Yields (API key, model name) pairs, skipping keys on cooldown.

        On a retry, this can be forced to use only the most capable model.

        Args:
            task_type: The category of task to perform (e.g., TEXT_TO_TEXT).
            force_best_model: If True, ignores the normal model rotation and
                              yields only the highest-tier model for the task.

        Yields:
            A tuple containing a valid, off-cooldown API key and a model name.
            
        Raises:
            RuntimeError: If all available API keys are on cooldown.
        """
        if task_type not in self._client_cycle:
            raise ValueError(f"No model mapping found for task: {task_type}")

        client_generator: Iterator[Tuple[str, str]]

        if force_best_model:
            tasks = self.models_config.get("tasks", {})
            model_list = tasks.get(task_type.name, {}).get("models", [])
            # By convention, the "best" (most capable) model is last in the list
            best_model = model_list[-1] if model_list else None
            if not best_model:
                raise ValueError(f"Could not determine the best model for task {task_type.name}")
            
            logger.info(f"Forcing best model for retry: {best_model}")
            clients = [(key, best_model) for key in self.api_keys]
            client_generator = cycle(clients)
        else:
            client_generator = self._client_cycle[task_type]
        
        max_checks = len(self.api_keys) * 5 # Prevent potential infinite loops
        for _ in range(max_checks):
            api_key, model_name = next(client_generator)
            if not self._is_on_cooldown(api_key):
                yield api_key, model_name
                return # Exit after yielding one valid client
        
        raise RuntimeError("All available API keys are currently on cooldown. Please wait or add more keys.")

    def _is_on_cooldown(self, api_key: str) -> bool:
        """Checks if a given API key is currently in its cooldown period."""
        return time.time() < self.cooldowns.get(api_key, 0)

    def mark_key_cooldown(self, api_key: str) -> None:
        """
        Marks a key as used, putting it on cooldown for the configured duration.
        This is a thread-safe operation.
        """
        with self._lock:
            self.cooldowns[api_key] = time.time() + self.api_key_cooldown_seconds
            logger.debug(f"Cooldown set for key ...{api_key[-4:]} for {self.api_key_cooldown_seconds}s")

    def update_tpm(self, api_key: str, model_name: str, tokens: int) -> None:
        """Placeholder for potential future TPM enforcement."""
        pass