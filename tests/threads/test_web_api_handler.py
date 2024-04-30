from collections.abc import Generator

import bottle
import pytest
from webtest import TestApp

from ami.threads.thread_control import ThreadController, ThreadControllerStatus
from ami.threads.web_api_handler import WebApiHandler


class TestWebApiHandler:
    @pytest.fixture
    def controller(self) -> ThreadController:
        return ThreadController()

    @pytest.fixture
    def handler(self, controller) -> WebApiHandler:
        status = ThreadControllerStatus(controller)
        handler = WebApiHandler(status, host="localhost", port=8080)
        return handler

    @pytest.fixture
    def app(self, handler) -> Generator[TestApp, None, None]:
        app = bottle.default_app()

        # Test server for testing WSGI server
        yield TestApp(app)

        app.reset()

    def test_get_status(self, app: TestApp, controller: ThreadController) -> None:
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "active"}

        controller.pause()
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "paused"}

        controller.shutdown()
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "stopped"}

    def test_post_pause(self, app: TestApp, handler: WebApiHandler):
        assert handler.receive_pause() is False
        response = app.post("http://localhost:8080/api/pause")
        assert response.status_code == 200
        assert response.json == {"result": "ok"}
        assert handler.receive_pause() is True
        assert handler.receive_pause() is False

    def test_post_resume(self, app: TestApp, handler: WebApiHandler):
        assert handler.receive_resume() is False
        response = app.post("http://localhost:8080/api/resume")
        assert response.status_code == 200
        assert response.json == {"result": "ok"}
        assert handler.receive_resume() is True
        assert handler.receive_resume() is False

    def test_post_shutdown(self, app: TestApp, handler: WebApiHandler):
        assert handler.receive_shutdown() is False
        response = app.post("http://localhost:8080/api/shutdown")
        assert response.status_code == 200
        assert response.json == {"result": "ok"}
        assert handler.receive_shutdown() is True
        assert handler.receive_shutdown() is False

    def test_error_404(self, app) -> None:
        response = app.get("http://localhost:8080/api/invalid", status=404)
        assert "error" in response.json

    def test_error_405(self, app) -> None:
        response = app.post("http://localhost:8080/api/status", status=405)
        assert "error" in response.json
