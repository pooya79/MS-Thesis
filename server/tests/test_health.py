from fastapi.testclient import TestClient

from server.app.main import app


def test_health_returns_401_when_unauthenticated_non_html() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}
