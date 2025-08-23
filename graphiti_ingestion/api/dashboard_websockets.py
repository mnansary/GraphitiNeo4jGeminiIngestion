# graphiti_ingestion/api/dashboard_websockets.py

import asyncio
import logging
from typing import List
import json
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages active WebSocket connections for the dashboard.

    This class provides a centralized way to track all connected clients and
    broadcast messages to them, such as live log updates or job status changes.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop = asyncio.get_event_loop()

    async def connect(self, websocket: WebSocket):
        """Accepts and stores a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Dashboard client connected via WebSocket.")

    def disconnect(self, websocket: WebSocket):
        """Removes a WebSocket connection."""
        self.active_connections.remove(websocket)
        logger.info("Dashboard client disconnected.")

    async def broadcast(self, message: str):
        """Sends a message to all connected clients."""
        # Create a list of tasks to send messages concurrently
        tasks = [connection.send_text(message) for connection in self.active_connections]
        # Wait for all messages to be sent, but don't fail if one client has an issue
        await asyncio.gather(*tasks, return_exceptions=True)

    def broadcast_threadsafe(self, message: str):
        """
        Sends a message from a non-async context (like a standard logging thread)
        by scheduling the broadcast on the main event loop.
        """
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)


# --- Create a singleton instance to be used across the application ---
websocket_manager = WebSocketManager()


class WebSocketLogHandler(logging.Handler):
    """
    A custom Python logging handler that streams log records over a WebSocket.

    This handler is added to the root logger. Whenever a log message is emitted
    anywhere in the application, this handler's `emit` method is called. It
    formats the log record and uses the WebSocketManager to broadcast it to all
    connected dashboard clients in real-time.
    """
    def __init__(self, manager: WebSocketManager):
        super().__init__()
        self.manager = manager
        # Set a professional log format for the dashboard stream
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord):
        """
        Formats the log record and broadcasts it thread-safely.
        """
        try:
            # Format the log record into a string
            msg = self.format(record)
            # Create a JSON structure for the frontend to easily parse
            log_data = {"type": "log", "payload": msg}
            # Use the thread-safe method to broadcast from the logging thread
            self.manager.broadcast_threadsafe(json.dumps(log_data))
        except Exception:
            # If broadcasting fails, fall back to handling the error locally
            self.handleError(record)