# main.py

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --- API Router Imports ---
from graphiti_ingestion.api.episodes import router as episodes_router
from graphiti_ingestion.api.dashboard import router as dashboard_router

# --- Service and Config Imports ---
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
# --- WebSocket and Logging Imports for Dashboard ---
from graphiti_ingestion.api.dashboard_websockets import (
    websocket_manager,
    WebSocketLogHandler
)


# --- Basic Logging Configuration ---
# We configure the root logger here.
settings = get_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # ---> ADD THIS: Ensure logs are handled by default handlers (like console)
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --- Background Worker Task (No changes from previous step) ---
async def worker(job_manager: JobManager, graphiti_service: GraphitiService):
    """
    The main background worker function, polling the file system for jobs.
    """
    logger.info("Background file-based worker started.")
    while True:
        try:
            job_details = await job_manager.get_next_job()
            if job_details:
                job_id, data = job_details
                logger.info(f"Worker processing job {job_id} from file queue...")
                try:
                    await graphiti_service.process_and_add_episode(data)
                    await job_manager.update_job_status(
                        job_id, JobStatus.COMPLETED, "Episode successfully ingested."
                    )
                    logger.info(f"Worker successfully completed job {job_id}.")
                    delay = settings.POST_SUCCESS_DELAY_SECONDS
                    if delay > 0:
                        logger.info(f"Success cooldown: Waiting for {delay} seconds before next job.")
                        await asyncio.sleep(delay)
                except Exception as e:
                    error_message = f"Failed to process episode: {e}"
                    logger.error(f"Worker failed on job {job_id}: {error_message}", exc_info=True)
                    await job_manager.update_job_status(job_id, JobStatus.FAILED, error_message)
            else:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Background worker received cancellation request. Shutting down gracefully.")
            break
        except Exception as e:
            logger.critical(f"An unexpected error occurred in the worker loop: {e}", exc_info=True)
            await asyncio.sleep(10)


# --- Application Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events.
    """
    logger.info("Application starting up...")

    # ---> ADD THIS: Configure logging to stream to the dashboard
    root_logger = logging.getLogger()
    websocket_log_handler = WebSocketLogHandler(websocket_manager)
    root_logger.addHandler(websocket_log_handler)
    logger.info("WebSocket logging handler configured and attached to root logger.")
    
    # Initialize services
    graphiti_service = initialize_graphiti_service()
    job_manager = get_job_manager()
    await graphiti_service.startup()

    # Start the background worker task
    worker_task = asyncio.create_task(worker(job_manager, graphiti_service))

    yield  # Application is now running

    logger.info("Application shutting down...")
    
    # Gracefully shut down the worker and other services
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Background worker task successfully cancelled.")
    await graphiti_service.shutdown()

    # ---> ADD THIS: Cleanly remove the custom log handler
    root_logger.removeHandler(websocket_log_handler)
    logger.info("Application shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Graphiti Ingestion Service with Monitoring",
    description="An asynchronous service to ingest data into a Neo4j knowledge graph, with a real-time monitoring dashboard.",
    version="0.2.0", # Version bump!
    lifespan=lifespan,
)

# --- Static Files Mounting ---
# This allows FastAPI to serve CSS, JS, etc., from the 'static' directory.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- MIDDLEWARE & EXCEPTION HANDLERS (No changes) ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f} ms"
    # Exclude dashboard polling from request logs for cleaner output
    if not request.url.path.startswith('/dashboard'):
        logger.info(
            f'Request: {request.method} {request.url.path} - Response: {response.status_code} - Time: {process_time:.2f}ms'
        )
    return response


# --- API ROUTERS ---
# Add both the original episodes router and the new dashboard router
app.include_router(episodes_router)
app.include_router(dashboard_router)


@app.get("/", tags=["Health Check"], summary="Basic health check endpoint")
async def read_root():
    """Root endpoint for basic service health verification."""
    return {"status": "ok", "message": "Graphiti Ingestion Service is running."}