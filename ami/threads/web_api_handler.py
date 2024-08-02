import json
import threading
from enum import Enum, auto
from queue import Queue
from typing import TypeAlias

import bottle

from ..logger import get_main_thread_logger
from .thread_control import ThreadControllerStatus

PayloadType: TypeAlias = dict[str, str]


class ControlCommands(Enum):
    """Enumerates the commands for the system control."""

    SHUTDOWN = auto()
    PAUSE = auto()
    RESUME = auto()
    SAVE_CHECKPOINT = auto()


class WebApiHandler:
    """Web API handler class.

    This class provides a simple Web API for controlling the ThreadController.

    Examples:
        GET /api/status: Get the status of the system.
        > curl http://localhost:8080/api/status
        {"status": "active"} # or "paused" or "stopped"

        POST /api/pause: Pause the system. (status: active -> paused)
        > curl -X POST http://localhost:8080/api/pause
        {"result": "ok"}

        POST /api/resume: Resume the system. (status: paused -> active)
        > curl -X POST http://localhost:8080/api/resume
        {"result": "ok"}

        POST /api/shutdown: Shutdown the system. (status: active or paused -> stopped)
        > curl -X POST http://localhost:8080/api/shutdown
        {"result": "ok"}
    """

    def __init__(
        self,
        controller_status: ThreadControllerStatus,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        """Initialize the WebApiHandler.

        Args:
            controller_status: To watch the thread controller status.
            host: Hostname to run the API server on.
            port: Port to run the API server on.
        """
        self._logger = get_main_thread_logger(self.__class__.__name__)
        self._controller_status = controller_status
        self._host = host
        self._port = port

        self._register_handlers()
        self._handler_thread = threading.Thread(target=self.run, daemon=True)

        self._received_commands_queue: Queue[ControlCommands] = Queue()

    def run(self) -> None:
        """Run the API server."""
        failed = True
        while failed:
            try:
                self._logger.info(f"Serving system command at '{self._host}:{self._port}'")
                bottle.run(host=self._host, port=self._port)
                failed = False
            except OSError:
                self._logger.info(f"Address '{self._host}:{self._port}' is already used, increment port number...")
                self._port += 1

    def run_in_background(self) -> None:
        """Run the API server in background thread."""
        self._handler_thread.start()

    def has_commands(self) -> bool:
        """Checks if the handler is receiving the system control commands."""
        return not self._received_commands_queue.empty()

    def receive_command(self) -> ControlCommands:
        """Receives a system control commands from queue.

        Returns:
            ControlCommands: Received commands from network.

        Raises:
            Empty: If the handler is receiving no commands.
        """
        return self._received_commands_queue.get_nowait()

    def _register_handlers(self) -> None:
        """Register API handlers."""

        bottle.get("/api/status", callback=self._get_status)
        bottle.post("/api/pause", callback=self._post_pause)
        bottle.post("/api/resume", callback=self._post_resume)
        bottle.post("/api/shutdown", callback=self._post_shutdown)
        bottle.post("/api/save-checkpoint", callback=self._post_save_checkpoint)

        bottle.error(404)(self._error_404)
        bottle.error(405)(self._error_405)
        bottle.error(500)(self._error_500)

        self._registered = True

    def _get_status(self) -> PayloadType:
        status: str
        if self._controller_status.is_shutdown():
            status = "stopped"
        elif self._controller_status.is_paused():
            status = "paused"
        else:
            status = "active"

        self._logger.info(f"_get_status: returning status = {status}")
        return {"status": status}

    def _post_pause(self) -> PayloadType:
        self._logger.info("Pausing threads")
        self._received_commands_queue.put(ControlCommands.PAUSE)
        return {"result": "ok"}

    def _post_resume(self) -> PayloadType:
        self._logger.info("Resuming threads")
        self._received_commands_queue.put(ControlCommands.RESUME)
        return {"result": "ok"}

    def _post_shutdown(self) -> PayloadType:
        self._logger.info("Shutting down threads")
        self._received_commands_queue.put(ControlCommands.SHUTDOWN)
        return {"result": "ok"}

    def _post_save_checkpoint(self) -> PayloadType:
        self._logger.info("Saving a checkpoint.")
        self._received_commands_queue.put(ControlCommands.SAVE_CHECKPOINT)
        return {"result": "ok"}

    def _error_404(self, error: bottle.HTTPError) -> str:
        request = bottle.request
        self._logger.error(f"404: {request.method} {request.path} is invalid API endpoint")

        bottle.response.headers["Content-Type"] = "application/json"
        return json.dumps({"error": "Invalid API endpoint"})

    def _error_405(self, error: bottle.HTTPError) -> str:
        request = bottle.request
        self._logger.error(f"405: {request.method} {request.path} is invalid API method")

        bottle.response.headers["Content-Type"] = "application/json"
        return json.dumps({"error": "Invalid API method"})

    def _error_500(self, error: bottle.HTTPError) -> str:
        request = bottle.request
        self._logger.error(f"500: {request.method} {request.path} caused an error")

        bottle.response.headers["Content-Type"] = "application/json"
        return json.dumps({"error": "Internal server error"})
