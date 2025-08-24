# graphiti_ingestion/gemini/worker.py

from __future__ import annotations

import queue
import random
import threading
import time
from typing import Any, Iterable, List, Optional, Tuple

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from graphiti_ingestion.gemini.manager import ComprehensiveManager, TaskType

import logging
logger = logging.getLogger(__name__)


def _to_contents(messages: Iterable[Any]) -> List[types.Content]:
    """
    Converts a list of Pydantic `Message` objects into the google-genai SDK's
    `types.Content` objects.

    Args:
        messages: An iterable of `graphiti_core.prompts.models.Message` objects.

    Returns:
        A list of `types.Content` objects ready for the API.
    """
    out: List[types.Content] = []
    for m in messages:
        # ---> THIS IS THE CORRECTED LOGIC <---
        # Access attributes directly on the 'Message' object (m.role)
        # instead of treating it like a dictionary (m.get("role")).
        role = m.role if hasattr(m, "role") else "user"
        if role == "assistant":
            role = "model"
        
        content = m.content if hasattr(m, "content") else ""
        if content is None:
            content = ""
        # ---> END OF CORRECTION <---

        out.append(types.Content(role=role, parts=[types.Part(text=content)]))
    return out


def _is_retryable_exception(exc: BaseException) -> Tuple[bool, Optional[int]]:
    """
    Determines if an exception is retryable.
    """
    if isinstance(exc, genai_errors.ServerError):
        return True, getattr(exc, 'code', 500)
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, 'code', 400) == 429, getattr(exc, 'code', 400)
    msg = str(exc).lower()
    if any(h in msg for h in ("timeout", "connection reset", "unavailable")):
        return True, None
    return False, None


class GeminiAPIWorker(threading.Thread):
    """
    A dedicated, synchronous worker that processes API requests one at a time.
    """

    def __init__(
        self,
        manager: ComprehensiveManager,
        work_queue: queue.Queue,
        delay_between_calls: float = 1.0,
        max_attempts: int = 5,
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
        """Pauses execution using a linear backoff strategy with random jitter."""
        backoff = min(self.max_backoff, self.base_backoff * (attempt_idx + 1))
        time.sleep(backoff * (0.5 + random.random()))

    def run(self) -> None:
        """The main loop of the worker thread."""
        logger.info("Synchronous Gemini API Worker has started.")
        while True:
            job = None
            try:
                job = self.work_queue.get()
                if job is None:
                    break  # Shutdown signal

                original_messages, gen_config, future, loop, retry_count = job
                
                force_best = retry_count > 0
                if force_best:
                    logger.warning(f"Retry attempt #{retry_count + 1}. Forcing best model.")
                
                client_generator = self.manager.get_available_client_details(
                    TaskType.TEXT_TO_TEXT, 
                    force_best_model=force_best
                )

                contents = _to_contents(original_messages)
                last_exc: Optional[BaseException] = None

                for attempt in range(self.max_attempts):
                    api_key, model_name = next(client_generator)
                    logger.info(f"WORKER: Attempt {attempt + 1}/{self.max_attempts} with key …{api_key[-4:]} on model '{model_name}'")
                    
                    try:
                        model_cfg = self.manager.get_model_config(model_name)
                        if model_cfg:
                            token_limit = model_cfg.get("tokens", {}).get("output_limit", 8192)
                            gen_config['max_output_tokens'] = token_limit
                            logger.info(f"Set max_output_tokens to {token_limit} for {model_name}.")

                        client = genai.Client(api_key=api_key)
                        safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE', 'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
                        
                        response = client.models.generate_content(
                            model=model_name, 
                            contents=contents, 
                            generation_config=types.GenerationConfig(**gen_config),
                            safety_settings=safety_settings
                        )
                        self.manager.mark_key_cooldown(api_key)

                        if not getattr(response, "text", None):
                            logger.error(f"WORKER: API returned 200 OK but response was empty. Full Response: {response}")

                        loop.call_soon_threadsafe(future.set_result, (response, model_name))
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        retryable, status = _is_retryable_exception(e)
                        if retryable:
                            logger.warning(f"WORKER: Retryable error (HTTP {status}) with key …{api_key[-4:]}. Retrying.")
                            self._sleep_with_jitter(attempt)
                        else:
                            logger.error(f"WORKER: Non-retryable error: {e}", exc_info=True)
                            loop.call_soon_threadsafe(future.set_exception, e)
                            break
                else:
                    err = Exception(f"Failed after {self.max_attempts} attempts. Last error: {last_exc}")
                    logger.error(f"WORKER: Exhausted all retries. Last error: {last_exc}")
                    loop.call_soon_threadsafe(future.set_exception, err)
            except Exception as e:
                logger.critical(f"Critical error in Gemini worker loop: {e}", exc_info=True)
                if 'future' in locals() and not future.done():
                    loop.call_soon_threadsafe(future.set_exception, e)
            finally:
                if job:
                    self.work_queue.task_done()
                time.sleep(self.delay_between_calls)
        logger.info("Synchronous Gemini API Worker is shutting down.")