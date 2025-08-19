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
from graphiti_ingestion.services.task_queue import TaskQueue, get_task_queue

# --- Basic Logging Configuration ---
settings = get_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Background Worker Task ---
async def worker(task_queue: TaskQueue, graphiti_service: GraphitiService):
    """
    The main background worker function.
    """
    logger.info("Background worker started.")
    while True:
        try:
            job = await task_queue.get_job()
            job_id = job["job_id"]
            data = job["data"]

            logger.info(f"Worker processing job {job_id}...")
            await task_queue.update_job_status(job_id, "processing")

            try:
                await graphiti_service.process_and_add_episode(data)
                await task_queue.update_job_status(
                    job_id, "completed", "Episode successfully ingested."
                )
                logger.info(f"Worker successfully completed job {job_id}.")
            except Exception as e:
                error_message = f"Failed to process episode: {e}"
                logger.error(f"Worker failed on job {job_id}: {error_message}", exc_info=True)
                await task_queue.update_job_status(job_id, "failed", error_message)
            finally:
                task_queue.mark_task_done()

        except asyncio.CancelledError:
            logger.info("Background worker received cancellation request. Shutting down.")
            break
        except Exception as e:
            logger.critical(f"An unexpected error occurred in the worker loop: {e}", exc_info=True)
            await asyncio.sleep(5)


# --- Application Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events.
    """
    logger.info("Application starting up...")
    
    graphiti_service = initialize_graphiti_service()
    task_queue = get_task_queue()

    await graphiti_service.startup()

    worker_task = asyncio.create_task(worker(task_queue, graphiti_service))
    
    yield
    
    logger.info("Application shutting down...")
    
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Background worker task successfully cancelled.")

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
# The order of middleware matters. The first one added is the first to process the request.

# 1. Global Exception Handler
# This will catch any unhandled exception and return a clean JSON response.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# 2. CORS Middleware
# Allows web frontends to communicate with this API.
# For production, you should restrict origins to your specific frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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
# Include the API router after middleware is defined.
app.include_router(episodes_router)


@app.get("/", tags=["Health Check"])
async def read_root():
    """
    Root endpoint for basic health check.
    """
    return {"status": "ok", "message": "Graphiti Ingestion Service is running."}