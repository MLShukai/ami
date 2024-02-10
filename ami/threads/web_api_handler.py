import json
from typing import TypeAlias

import bottle

from ..logger import get_main_thread_logger
from .thread_control import ThreadController

PayloadType: TypeAlias = dict[str, str]


class WebApiHandler:
    """Web API handler class."""

    def __init__(self, controller: ThreadController, host: str = "localhost", port: int = 8080) -> None:
        self._logger = get_main_thread_logger(self.__class__.__name__)
        self._controller = controller
        self._host = host
        self._port = port

        self._register_handlers()

    def run(self):
        bottle.run(host=self._host, port=self._port)

    def _register_handlers(self) -> None:
        """Register API handlers."""

        bottle.get("/api/status", callback=self._get_status)
        bottle.post("/api/pause", callback=self._post_pause)
        bottle.post("/api/resume", callback=self._post_resume)
        bottle.post("/api/shutdown", callback=self._post_shutdown)

        bottle.error(404)(self._error_404)
        bottle.error(405)(self._error_405)
        bottle.error(500)(self._error_500)

        self._registered = True

    def _get_status(self) -> PayloadType:
        status: str = ""
        if self._controller.is_shutdown():
            status = "stopped"
        elif not self._controller.is_resumed():
            status = "paused"
        else:
            status = "active"

        self._logger.info(f"_get_status: returning status = {status}")
        return {"result": "ok", "status": status}

    def _post_pause(self) -> PayloadType:
        self._logger.info("Pausing threads")
        self._controller.pause()
        return {"result": "ok", "status": "paused"}

    def _post_resume(self) -> PayloadType:
        self._logger.info("Resuming threads")
        self._controller.resume()
        return {"result": "ok", "status": "active"}

    def _post_shutdown(self) -> PayloadType:
        self._logger.info("Shutting down threads")
        self._controller.shutdown()
        return {"result": "ok", "status": "stopped"}

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
