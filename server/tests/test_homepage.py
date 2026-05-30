from fastapi.testclient import TestClient

from server.app.main import app


def test_login_page_is_public() -> None:
    with TestClient(app) as client:
        response = client.get("/login")

    assert response.status_code == 200
    assert "Enter the application password to continue." in response.text
    assert "/static/css/login.css" in response.text
    assert "auth-layout" in response.text


def test_home_redirects_to_login_when_unauthenticated() -> None:
    with TestClient(app) as client:
        response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/login?next=%2F")


def test_login_rejects_invalid_password() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/login",
            data={"password": "wrong-password", "next": "/"},
            follow_redirects=False,
        )

    assert response.status_code == 401
    assert "Invalid password." in response.text


def test_login_allows_access_with_valid_password() -> None:
    with TestClient(app) as client:
        login_response = client.post(
            "/login",
            data={"password": "test-password", "next": "/"},
            follow_redirects=False,
        )
        home_response = client.get("/")
        health_response = client.get("/health")

    assert login_response.status_code == 303
    assert login_response.headers["location"] == "/"
    assert home_response.status_code == 200
    assert "Research command center" in home_response.text
    assert "/static/css/shell.css" in home_response.text
    assert "Log out" in home_response.text
    assert 'data-sidebar-backdrop hidden' in home_response.text
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}


def test_static_assets_are_publicly_readable() -> None:
    with TestClient(app) as client:
        response = client.get("/static/css/tokens.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert "--color-bg-primary" in response.text


def test_shell_assets_include_sidebar_toggle_behavior() -> None:
    with TestClient(app) as client:
        css = client.get("/static/css/shell.css")
        js = client.get("/static/js/shell.js")

    assert css.status_code == 200
    assert js.status_code == 200
    assert "sidebar-backdrop" in css.text
    assert "setSidebarOpen" in js.text
    assert "aria-expanded" in js.text


def test_logout_clears_authenticated_session() -> None:
    with TestClient(app) as client:
        login_response = client.post(
            "/login",
            data={"password": "test-password", "next": "/"},
            follow_redirects=False,
        )
        logout_response = client.post("/logout", follow_redirects=False)
        home_response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)

    assert login_response.status_code == 303
    assert logout_response.status_code == 303
    assert logout_response.headers["location"] == "/login"
    assert home_response.status_code == 303
    assert home_response.headers["location"].startswith("/login?next=%2F")


def test_docs_are_protected() -> None:
    with TestClient(app) as client:
        response = client.get("/docs", headers={"accept": "text/html"}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/login?next=%2Fdocs")
