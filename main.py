# main.py

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# --- Basic Logging Configuration ---
settings = get_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Background Worker Task ---
async def worker(job_manager: JobManager, graphiti_service: GraphitiService):
    """
    The main background worker function.

    This worker runs in an infinite loop, periodically polling the file system
    for new jobs in the 'pending' directory. It processes one job at a time
    to ensure sequential and orderly ingestion.
    """
    logger.info("Background file-based worker started.")
    while True:
        try:
            # Atomically fetch the next job from the 'pending' directory.
            # This moves the job to the 'processing' directory.
            job_details = await job_manager.get_next_job()

            if job_details:
                job_id, data = job_details
                logger.info(f"Worker processing job {job_id} from file queue...")

                try:
                    # Core ingestion logic
                    await graphiti_service.process_and_add_episode(data)

                    # On success, move job to the 'completed' directory
                    await job_manager.update_job_status(
                        job_id, JobStatus.COMPLETED, "Episode successfully ingested."
                    )
                    logger.info(f"Worker successfully completed job {job_id}.")

                except Exception as e:
                    # On failure, move job to the 'failed' directory with an error message
                    error_message = f"Failed to process episode: {e}"
                    logger.error(f"Worker failed on job {job_id}: {error_message}", exc_info=True)
                    await job_manager.update_job_status(job_id, JobStatus.FAILED, error_message)
            else:
                # If no pending jobs are found, wait for a few seconds before checking again.
                # This prevents a tight loop that consumes CPU.
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("Background worker received cancellation request. Shutting down gracefully.")
            break
        except Exception as e:
            # Catch any unexpected errors in the worker loop itself
            logger.critical(f"An unexpected error occurred in the worker loop: {e}", exc_info=True)
            # Wait longer before retrying to prevent rapid failure loops
            await asyncio.sleep(10)


# --- Application Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events using FastAPI's
    lifespan context manager.
    """
    logger.info("Application starting up...")

    # Initialize services. Calling get_job_manager() here ensures
    # the necessary job directories are created on startup.
    graphiti_service = initialize_graphiti_service()
    job_manager = get_job_manager()

    # Perform startup tasks for the Graphiti service (e.g., build indexes)
    await graphiti_service.startup()

    # Create and start the background worker task
    worker_task = asyncio.create_task(worker(job_manager, graphiti_service))

    yield  # The application is now running

    logger.info("Application shutting down...")

    # Gracefully cancel and await the worker task
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Background worker task successfully cancelled.")

    # Perform shutdown tasks for the Graphiti service (e.g., close connections)
    await graphiti_service.shutdown()
    logger.info("Application shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Graphiti Ingestion Service",
    description="An asynchronous service to ingest data into a Neo4j knowledge graph using Graphiti.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- MIDDLEWARE & EXCEPTION HANDLERS ---
# The order of middleware matters.

# 1. Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Request Logging and Process Time Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f} ms"
    logger.info(
        f'Request: {request.method} {request.url.path} - Response: {response.status_code} - Time: {process_time:.2f}ms'
    )
    return response


# --- API ROUTERS ---
app.include_router(episodes_router)


@app.get("/", tags=["Health Check"], summary="Basic health check endpoint")
async def read_root():
    """
    Root endpoint for basic service health verification.
    """
    return {"status": "ok", "message": "Graphiti Ingestion Service is running."}