# gemini_api_manager/worker.py

from __future__ import annotations

import queue
import threading
import time
import random
from typing import Any, Iterable, Tuple, Optional, List

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from loguru import logger

from .core.manager import ComprehensiveManager
from .core.task_types import TaskType


def _to_contents(messages: Iterable[Any]) -> List[types.Content]:
    """
    Convert app-level Message objects into google-genai types.Content.
    """
    out: List[types.Content] = []
    for m in messages:
        role = getattr(m, "role", None) or "user"
        if role == "assistant":
            role = "model"
        text = getattr(m, "content", "")
        if text is None:
            text = ""
        out.append(
            types.Content(
                role=role,
                # --- THIS IS THE CORRECTED LINE ---
                parts=[types.Part(text=text)]
                # --- END OF CORRECTION ---
            )
        )
    return out


def _is_retryable_exception(exc: BaseException) -> Tuple[bool, Optional[int]]:
    """
    Decide whether an exception is retryable and return (retryable, http_status_if_any).
    """
    # ClientError may wrap HTTP errors with response/status_code
    if isinstance(exc, genai_errors.ClientError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Retry on common transient statuses
        if status in (408, 429, 500, 502, 503, 504):
            return True, status
        return False, status

    # These are known transient conditions in many HTTP stacks
    transient_types = (
        genai_errors.ResourceExhausted,   # often 429
        genai_errors.InternalServerError, # 500
        genai_errors.BadGateway,          # 502
        genai_errors.ServiceUnavailable,  # 503
        genai_errors.GatewayTimeout,      # 504
    )
    if isinstance(exc, transient_types):
        return True, None

    # Connection resets / timeouts sometimes bubble up as generic Exceptions; be conservative.
    msg = str(exc).lower()
    network_hints = ("timeout", "temporarily", "temporary", "unavailable",
                     "connection reset", "connection aborted", "timed out")
    if any(h in msg for h in network_hints):
        return True, None

    return False, None


class GeminiAPIWorker(threading.Thread):
    """
    Dedicated, synchronous worker that processes API requests strictly one at a time.
    Ensures sequential calls to avoid rate-limit bursts and provides robust retry/rotation.
    """

    def __init__(
        self,
        manager: ComprehensiveManager,
        work_queue: queue.Queue,
        delay_between_calls: float = 10.0,
        max_attempts: int = 12,
        base_backoff: float = 0.75,
        max_backoff: float = 8.0,
    ):
        super().__init__(daemon=True)
        self.manager = manager
        self.work_queue = work_queue
        self.delay_between_calls = delay_between_calls
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff

    def _sleep_with_jitter(self, attempt_idx: int) -> None:
        # Exponential backoff with jitter
        backoff = min(self.max_backoff, self.base_backoff * (2 ** attempt_idx))
        sleep_s = backoff * (0.5 + random.random())  # jitter in [0.5x, 1.5x]
        time.sleep(sleep_s)

    def run(self) -> None:
        logger.info("Synchronous Gemini API Worker has started.")
        client_generator = self.manager.get_available_client_details(TaskType.TEXT_TO_TEXT)

        while True:
            try:
                job = self.work_queue.get()
                if job is None:
                    break  # Shutdown signal

                # Unpack job tuple provided by the async client
                original_messages, gen_config, future, loop = job

                # Prepare request contents once per job
                try:
                    contents = _to_contents(original_messages)
                except Exception as e:
                    logger.error(f"WORKER: Failed to convert messages -> contents: {e}")
                    loop.call_soon_threadsafe(future.set_exception, e)
                    continue

                # Attempt with key rotation & backoff
                last_exc: Optional[BaseException] = None
                for attempt in range(self.max_attempts):
                    api_key, model_name, _ = next(client_generator)
                    # Do NOT prefix "models/" here — your configs already use canonical names like "gemini-2.5-flash"
                    logger.info(
                        f"WORKER: Attempt {attempt + 1}/{self.max_attempts} "
                        f"using key …{api_key[-4:]} on model '{model_name}'"
                    )

                    try:
                        client = genai.Client(api_key=api_key)
                        # google-genai==1.29.0: generate_content(model=<name>, contents=[types.Content], config=GenerateContentConfig)
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=gen_config,
                        )

                        # Usage accounting
                        self.manager._increment_rpm_usage(api_key, model_name)
                        self.manager._set_key_cooldown(api_key)
                        if hasattr(response, "usage_metadata") and getattr(response.usage_metadata, "total_token_count", None) is not None:
                            tokens = response.usage_metadata.total_token_count
                            self.manager._update_tpm_usage(api_key, model_name, tokens)

                        # Success: deliver result back to async caller thread-safely
                        loop.call_soon_threadsafe(future.set_result, (response, model_name))
                        last_exc = None
                        break

                    except Exception as e:  # noqa: BLE001 – we classify below
                        retryable, status = _is_retryable_exception(e)
                        last_exc = e
                        if retryable:
                            # Log with detail; rotate to next key and backoff
                            status_txt = f" (HTTP {status})" if status else ""
                            logger.warning(
                                f"WORKER: Transient error{status_txt} with key …{api_key[-4:]}, "
                                f"will retry with backoff. Error: {e}"
                            )
                            self._sleep_with_jitter(attempt)
                            # continue to next attempt (key rotation handled by generator)
                            continue
                        else:
                            logger.error(f"WORKER: Non-retryable error: {e}")
                            loop.call_soon_threadsafe(future.set_exception, e)
                            break
                else:
                    # Exhausted attempts
                    err = genai_errors.ResourceExhausted(
                        f"Failed after {self.max_attempts} attempts; keys likely rate-limited or service unavailable."
                    )
                    logger.error(f"WORKER: Exhausted attempts. Last error: {last_exc}")
                    loop.call_soon_threadsafe(future.set_exception, err)

            finally:
                # Mark job done and pace sequential calls
                self.work_queue.task_done()
                time.sleep(self.delay_between_calls)

        logger.info("Synchronous Gemini API Worker is shutting down.")
