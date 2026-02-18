import secrets
from urllib.parse import urlencode

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from server.app.core.config import get_settings


def verify_password(submitted_password: str) -> bool:
    settings = get_settings()
    return secrets.compare_digest(submitted_password, settings.app_password)


def is_html_request(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def is_public_path(request: Request) -> bool:
    path = request.url.path
    method = request.method

    if path == "/login" and method in {"GET", "POST"}:
        return True

    if method in {"GET", "HEAD"} and (path == "/static" or path.startswith("/static/")):
        return True

    return False


def sanitize_next_path(next_path: str | None) -> str:
    if not next_path:
        return "/"
    if not next_path.startswith("/"):
        return "/"
    if next_path.startswith("//"):
        return "/"
    return next_path


class PasswordAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if is_public_path(request):
            return await call_next(request)

        if request.session.get("authenticated") is True:
            return await call_next(request)

        if is_html_request(request):
            next_path = request.url.path
            if request.url.query:
                next_path = f"{next_path}?{request.url.query}"
            params = urlencode({"next": next_path})
            return RedirectResponse(url=f"/login?{params}", status_code=303)

        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"},
        )
