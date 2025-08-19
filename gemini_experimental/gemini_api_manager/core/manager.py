# gemini_api_manager/core/manager.py

from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Iterator

import yaml
from loguru import logger

from .task_types import TaskType


class ComprehensiveManager:
    """
    Central controller for Gemini API usage.
    Handles:
      - API key rotation with cooldown
      - Enforcing per-model RPM, TPM, RPD
      - Mapping tasks -> models
    """

    def __init__(self, api_key_csv_path: str, model_config_path: str):
        self._lock = threading.Lock()
        self.api_keys = self._load_api_keys(api_key_csv_path)
        self.models_config = self._load_model_config(model_config_path)

        # Model usage counters
        self.rpm_usage = defaultdict(int)
        self.tpm_usage = defaultdict(int)
        self.rpd_usage = defaultdict(int)

        # Cooldown tracking
        self.cooldowns: dict[str, float] = {}

        # Round-robin generator for available clients
        self._client_cycle: dict[TaskType, Iterator] = {}
        self._init_client_generators()

    def _load_api_keys(self, csv_path: str) -> list[str]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"API key CSV not found: {csv_path}")

        keys = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "api" not in reader.fieldnames:
                raise ValueError("CSV must have an 'api' column")
            for row in reader:
                key = row["api"].strip()
                if key:
                    keys.append(key)
        if not keys:
            raise ValueError("No API keys found in CSV.")
        logger.info(f"Loaded {len(keys)} API keys.")
        return keys

    def _load_model_config(self, yaml_path: str) -> dict:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Model config YAML not found: {yaml_path}")
        with open(path) as f:
            config = yaml.safe_load(f)
        logger.info("Model configuration loaded.")
        return config

    def _init_client_generators(self) -> None:
        """Pre-build round-robin iterators for each TaskType."""
        tasks = self.models_config.get("tasks", {})
        for task_name, task_info in tasks.items():
            try:
                task_enum = TaskType[task_name]
            except KeyError:
                logger.warning(f"Unknown task type in config: {task_name}")
                continue
            models = task_info.get("models", [])
            # Pair each model with every API key for rotation
            clients = [(k, m, task_enum) for k in self.api_keys for m in models]
            self._client_cycle[task_enum] = cycle(clients)

    def get_available_client_details(self, task_type: TaskType) -> Iterator[tuple[str, str, TaskType]]:
        """
        Yields API key, model name, and task type in a round-robin fashion,
        skipping keys on cooldown.
        """
        if task_type not in self._client_cycle:
            raise ValueError(f"No model mapping for task: {task_type}")
        while True:
            api_key, model_name, _ = next(self._client_cycle[task_type])
            if not self._is_on_cooldown(api_key):
                yield api_key, model_name, task_type
            else:
                # Skip this key for now
                continue

    def _is_on_cooldown(self, api_key: str) -> bool:
        cooldown_until = self.cooldowns.get(api_key, 0)
        return time.time() < cooldown_until

    def _set_key_cooldown(self, api_key: str, seconds: float = 60.0) -> None:
        """Mark a key as unavailable for a cooldown period."""
        with self._lock:
            self.cooldowns[api_key] = time.time() + seconds
            logger.debug(f"Cooldown set for key ...{api_key[-4:]} until {self.cooldowns[api_key]}")

    def _increment_rpm_usage(self, api_key: str, model_name: str) -> None:
        """Increase the per-minute request counter."""
        with self._lock:
            self.rpm_usage[(api_key, model_name)] += 1

    def _update_tpm_usage(self, api_key: str, model_name: str, tokens_used: int) -> None:
        """Increase the per-minute token counter."""
        with self._lock:
            self.tpm_usage[(api_key, model_name)] += tokens_used

    def _increment_rpd_usage(self, api_key: str, model_name: str) -> None:
        """Increase the per-day request counter."""
        with self._lock:
            self.rpd_usage[(api_key, model_name)] += 1

    def reset_usage_counters(self) -> None:
        """Resets counters periodically (should be called by a scheduler)."""
        with self._lock:
            self.rpm_usage.clear()
            self.tpm_usage.clear()
            self.rpd_usage.clear()
        logger.debug("Usage counters reset.")

    # Expose cooldown and usage updates to worker
    def mark_key_cooldown(self, api_key: str) -> None:
        self._set_key_cooldown(api_key)

    def increment_rpm(self, api_key: str, model_name: str) -> None:
        self._increment_rpm_usage(api_key, model_name)

    def increment_rpd(self, api_key: str, model_name: str) -> None:
        self._increment_rpd_usage(api_key, model_name)

    def update_tpm(self, api_key: str, model_name: str, tokens: int) -> None:
        self._update_tpm_usage(api_key, model_name, tokens)
