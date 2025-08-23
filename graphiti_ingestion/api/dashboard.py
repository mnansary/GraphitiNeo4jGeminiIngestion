# graphiti_ingestion/api/dashboard.py

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..services.job_manager import JobManager, get_job_manager
from .dashboard_websockets import websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
    responses={404: {"description": "Not found"}},
)

# Define the path to the static directory relative to the project root
# This assumes your 'static' folder is at the same level as 'main.py'
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"


@router.get(
    "/",
    summary="Serve the dashboard's main HTML page",
    description="Serves the single-page application for the monitoring dashboard.",
)
async def get_dashboard_page():
    """
    Endpoint to serve the `index.html` file for the dashboard.
    FastAPI's FileResponse handles sending the file to the client's browser.
    """
    if not INDEX_HTML_PATH.is_file():
        logger.error(f"Dashboard HTML file not found at: {INDEX_HTML_PATH}")
        return {"error": "Dashboard UI not found"}, 500
    return FileResponse(INDEX_HTML_PATH)


@router.websocket("/ws/dashboard")
async def websocket_endpoint(
    websocket: WebSocket,
    job_manager: JobManager = Depends(get_job_manager),
):
    """
    The main WebSocket endpoint for the dashboard.

    It handles the lifecycle of a client connection, listens for incoming
    commands from the frontend, and serves as the entry point for broadcasting

    real-time updates.
    """
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Wait for a message from the client
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
                action = data.get("action")

                # Handle the initial request to load all job data
                if action == "get_all_jobs":
                    logger.info("Dashboard requested all job statuses.")
                    all_jobs = await job_manager.get_all_job_statuses()
                    # Send the job list back to the client
                    await websocket.send_json({
                        "type": "all_jobs",
                        "payload": all_jobs
                    })

            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON via WebSocket: {raw_data}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected.")
    finally:
        # Ensure the connection is removed from the manager on disconnect
        websocket_manager.disconnect(websocket)