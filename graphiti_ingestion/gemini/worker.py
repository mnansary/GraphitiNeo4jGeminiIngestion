# graphiti_ingestion/gemini/worker.py

from __future__ import annotations

import queue
import threading
import time
import random
from typing import Any, Iterable, Tuple, Optional, List
import logging
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from graphiti_ingestion.gemini.manager import ComprehensiveManager,TaskType

logger = logging.getLogger(__name__)

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
                parts=[types.Part(text=text)]
            )
        )
    return out


def _is_retryable_exception(exc: BaseException) -> Tuple[bool, Optional[int]]:
    """
    Decide whether an exception is retryable based on the actual available
    error classes: ServerError, ClientError(429), and network errors.
    """
    # ---> THE DEFINITIVE FIX <---
    # This logic is now based on the errors.py file you provided.

    # 1. Any 5xx error is a ServerError and is always retryable.
    if isinstance(exc, genai_errors.ServerError):
        status_code = getattr(exc, 'code', 500)
        return True, status_code

    # 2. For 4xx errors (ClientError), only retry on 429 (Resource Exhausted / Rate Limited).
    if isinstance(exc, genai_errors.ClientError):
        status_code = getattr(exc, 'code', 400)
        if status_code == 429:
            return True, status_code
        else:
            # Any other client error (e.g., 400 Bad Request) is NOT retryable.
            return False, status_code

    # 3. For low-level network errors not wrapped by the genai library.
    msg = str(exc).lower()
    network_hints = (
        "timeout", "temporarily", "temporary", "unavailable",
        "connection reset by peer", "connection reset", "connection aborted", "timed out"
    )
    if any(h in msg for h in network_hints):
        return True, None

    # If none of the above, it's a non-retryable error.
    return False, None
    # ---> END OF FIX <---


class GeminiAPIWorker(threading.Thread):
    """
    Dedicated, synchronous worker that processes API requests strictly one at a time.
    """
    def __init__(
        self,
        manager: ComprehensiveManager,
        work_queue: queue.Queue,
        delay_between_calls: float = 1.0,
        max_attempts: int = 50,
        base_backoff: float = 1.0,
        max_backoff: float = 10.0,
    ):
        super().__init__(daemon=True)
        self.manager = manager
        self.work_queue = work_queue
        self.delay_between_calls = delay_between_calls
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff

    def _sleep_with_jitter(self, attempt_idx: int) -> None:
        """Linear backoff with jitter."""
        backoff = min(self.max_backoff, self.base_backoff * (attempt_idx + 1))
        sleep_s = backoff * (0.5 + random.random())
        time.sleep(sleep_s)

    def run(self) -> None:
        logger.info("Synchronous Gemini API Worker has started.")
        client_generator = self.manager.get_available_client_details(TaskType.TEXT_TO_TEXT)

        while True:
            job = None
            try:
                job = self.work_queue.get()
                if job is None:
                    break  # Shutdown signal

                original_messages, gen_config, future, loop = job

                try:
                    contents = _to_contents(original_messages)
                except Exception as e:
                    logger.error(f"WORKER: Failed to convert messages -> contents: {e}", exc_info=True)
                    loop.call_soon_threadsafe(future.set_exception, e)
                    continue

                last_exc: Optional[BaseException] = None
                for attempt in range(self.max_attempts):
                    api_key, model_name, _ = next(client_generator)
                    logger.info(
                        f"WORKER: Attempt {attempt + 1}/{self.max_attempts} "
                        f"using key …{api_key[-4:]} on model '{model_name}'"
                    )
                    try:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=gen_config,
                        )

                        self.manager.mark_key_cooldown(api_key)
                        if hasattr(response, "usage_metadata") and response.usage_metadata:
                            tokens = response.usage_metadata.total_token_count
                            self.manager.update_tpm(api_key, model_name, tokens)

                        loop.call_soon_threadsafe(future.set_result, (response, model_name))
                        last_exc = None
                        break # Success!

                    except Exception as e:
                        last_exc = e
                        retryable, status = _is_retryable_exception(e)
                        if retryable:
                            status_txt = f" (HTTP {status})" if status else ""
                            logger.warning(
                                f"WORKER: Retryable error{status_txt} with key …{api_key[-4:]}. "
                                f"Retrying after linear backoff. Error type: {type(e).__name__}"
                            )
                            self._sleep_with_jitter(attempt)
                            continue # Go to the next attempt in the loop
                        else:
                            logger.error(f"WORKER: Non-retryable error: {e}", exc_info=True)
                            loop.call_soon_threadsafe(future.set_exception, e)
                            break # Fail the job permanently
                else:
                    # This block runs only if the for loop finishes without a 'break'
                    final_error_msg = f"Failed after {self.max_attempts} attempts."
                    logger.error(f"WORKER: Exhausted all retries. Last error: {last_exc}")
                    err = Exception(f"{final_error_msg} Last known error: {last_exc}")
                    loop.call_soon_threadsafe(future.set_exception, err)

            except Exception as e:
                logger.critical(f"A critical unexpected error occurred in the Gemini worker loop: {e}", exc_info=True)
            
            finally:
                if job is not None:
                    self.work_queue.task_done()
                time.sleep(self.delay_between_calls)

        logger.info("Synchronous Gemini API Worker is shutting down.")