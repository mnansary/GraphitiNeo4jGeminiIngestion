# main.py

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from graphiti_ingestion.api.dashboard import router as dashboard_router
from graphiti_ingestion.api.dashboard_websockets import (
    WebSocketLogHandler,
    websocket_manager,
)
from graphiti_ingestion.api.episodes import router as episodes_router
from graphiti_ingestion.config import get_settings
from graphiti_ingestion.services.graphiti_service import (
    GraphitiService,
    get_graphiti_service,
    initialize_graphiti_service,
)
from graphiti_ingestion.services.job_manager import (
    JobManager,
    JobStatus,
    get_job_manager,
)

# --- Configuration & Logging ---
settings = get_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

MAX_CONTENT_RETRIES = 2


# --- Background Worker ---
async def worker(job_manager: JobManager, graphiti_service: GraphitiService):
    """
    The main background worker, now with intelligent failure recovery.
    """
    logger.info("Background worker with advanced failure recovery started.")
    while True:
        try:
            job_details = await job_manager.get_next_job()

            if job_details:
                job_id, data, retry_count = job_details
                logger.info(f"Worker processing job {job_id} (Attempt #{retry_count + 1})")
                try:
                    await graphiti_service.process_and_add_episode(data, retry_count=retry_count)
                    await job_manager.update_job_status(
                        job_id, JobStatus.COMPLETED, "Episode successfully ingested."
                    )
                    logger.info(f"Worker successfully completed job {job_id}.")
                    
                    delay = settings.POST_SUCCESS_DELAY_SECONDS
                    if delay > 0:
                        logger.info(f"Success cooldown: Waiting for {delay} seconds before next job.")
                        await asyncio.sleep(delay)

                except ValueError as e:
                    error_message = str(e)
                    logger.warning(f"Job {job_id} failed with a content error: {error_message}")
                    if retry_count < MAX_CONTENT_RETRIES - 1:
                        new_retry_count = retry_count + 1
                        msg = f"Re-queuing for attempt #{new_retry_count + 1} with a better model."
                        logger.warning(f"Job {job_id}: {msg}")
                        await job_manager.requeue_job_for_retry(job_id, new_retry_count, msg)
                    else:
                        msg = f"Failed permanently after {MAX_CONTENT_RETRIES} attempts."
                        logger.error(f"Job {job_id}: {msg}")
                        await job_manager.update_job_status(job_id, JobStatus.FAILED, msg)
                except Exception as e:
                    msg = f"An unexpected error occurred: {e}"
                    logger.error(f"Worker failed on job {job_id}: {msg}", exc_info=True)
                    await job_manager.update_job_status(job_id, JobStatus.FAILED, msg)
            else:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Background worker received cancellation request.")
            break
        except Exception as e:
            logger.critical(f"A critical error occurred in the main worker loop: {e}", exc_info=True)
            await asyncio.sleep(10)


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("Application starting up...")
    
    root_logger = logging.getLogger()
    websocket_log_handler = WebSocketLogHandler(websocket_manager)
    root_logger.addHandler(websocket_log_handler)
    
    graphiti_service = initialize_graphiti_service()
    job_manager = get_job_manager()
    await graphiti_service.startup()
    
    worker_task = asyncio.create_task(worker(job_manager, graphiti_service))
    
    yield
    
    logger.info("Application shutting down...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Background worker task successfully cancelled.")
    await graphiti_service.shutdown()
    root_logger.removeHandler(websocket_log_handler)
    logger.info("Application shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Graphiti Ingestion Service with Monitoring",
    description="An asynchronous service for ingesting data into a Neo4j knowledge graph.",
    version="0.3.2", # Version bump for the fix
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches all unhandled exceptions and returns a clean 500 error."""
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# ---> THIS IS THE CORRECTED BLOCK <---
# We are putting the hardcoded URL back, as it's simpler and avoids the config error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---> END OF CORRECTION <---

app.include_router(episodes_router)
app.include_router(dashboard_router)