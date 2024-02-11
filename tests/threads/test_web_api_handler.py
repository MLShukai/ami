import pytest
# import app
from typing import Generator
import bottle
from webtest import TestApp

from ami.threads.thread_control import ThreadController
from ami.threads.web_api_handler import WebApiHandler


class TestWebApiHandler:
    @pytest.fixture
    def app(self) -> Generator[TestApp, None, None]:
        controller = ThreadController()
        handler = WebApiHandler(controller, host="localhost", port=8080)
        app = bottle.default_app()

        # Test server for testing WSGI server
        yield TestApp(app)

        app.reset()


    def test_get_status(self, app: TestApp) -> None:
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "active"}


    def test_post_pause_resume(self, app: TestApp) -> None:
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "active"}

        response = app.post("http://localhost:8080/api/pause")
        assert response.status_code == 200
        assert response.json == {"result": "ok", "status": "paused"}

        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "paused"}

        response = app.post("http://localhost:8080/api/resume")
        assert response.status_code == 200
        assert response.json == {"result": "ok", "status": "active"}

        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "active"}

    def test_post_shutdown(self, app) -> None:
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "active"}

        response = app.post("http://localhost:8080/api/shutdown")
        assert response.status_code == 200
        assert response.json == {"result": "ok", "status": "stopped"}

        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "stopped"}

    def test_post_pause_shutdown(self, app) -> None:
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "active"}

        response = app.post("http://localhost:8080/api/pause")
        assert response.status_code == 200
        assert response.json == {"result": "ok", "status": "paused"}

        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "paused"}

        response = app.post("http://localhost:8080/api/shutdown")
        assert response.status_code == 200
        assert response.json == {"result": "ok", "status": "stopped"}

        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "stopped"}

    def test_error_404(self, app) -> None:
        response = app.get("http://localhost:8080/api/invalid", status=404)
        assert "error" in response.json

    def test_error_405(self, app) -> None:
        response = app.post("http://localhost:8080/api/status", status=405)
        assert "error" in response.json
