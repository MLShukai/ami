import json
from logging import Logger
from typing import Self

import bottle

from ..logger import get_main_thread_logger
from .base_threads import BaseMainThread
from .thread_control import ThreadController


class WebApiHandler:
    """Web API handler class."""

    _logger: Logger

    def __init__(self, controller: ThreadController):
        self._logger = get_main_thread_logger(self.__class__.__name__)
        self._controller = controller

        self._register_handlers()

    def run(self):
        bottle.run(host="localhost", port=8080)

    def _register_handlers(self) -> None:
        """Register API handlers."""

        bottle.get("/api/status", callback=self._get_status)
        bottle.post("/api/start", callback=self._post_start)
        bottle.post("/api/pause", callback=self._post_pause)
        bottle.post("/api/resume", callback=self._post_resume)
        bottle.post("/api/shutdown", callback=self._post_shutdown)

        bottle.error(404)(self._error_404)
        bottle.error(405)(self._error_405)
        bottle.error(500)(self._error_500)

        self._registered = True

    def _get_status(self) -> dict:
        # TODO: Check if training and inference threads are active
        if self._controller.is_shutdown():
            return {"status": "stopped"}
        elif False:
            return {"status": "paused"}
        else:
            self._logger.info("Status: active")
            return {"status": "active"}

    def _post_start(self) -> dict:
        # TODO: Start training and inference threads
        self._logger.info("Starting threads")
        return {"result": "ok", "status": "active"}

    def _post_pause(self) -> dict:
        # TODO: Pause training and inference threads
        self._logger.info("Pausing threads")
        return {"result": "ok", "status": "paused"}

    def _post_resume(self) -> dict:
        # TODO: Resume training and inference threads
        self._logger.info("Resuming threads")
        return {"result": "ok", "status": "active"}

    def _post_shutdown(self) -> dict:
        # TODO: Shutdown training and inference threads
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


class MainThread(BaseMainThread):
    """Main thread class."""

    _logger: Logger
    _controller: ThreadController
    _handler: WebApiHandler

    def __init__(self) -> None:
        super().__init__()

        self._logger = get_main_thread_logger(self.__class__.__name__)
        self._controller = ThreadController()
        self._handler = WebApiHandler(self._controller)

    def worker(self) -> None:
        self._handler.run()

    def on_shared_object_pool_attached(self) -> None:
        pass
