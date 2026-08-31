from __future__ import annotations

import hashlib
import hmac
import os
import time
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

SESSION_COOKIE = "speech_lab_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60


def require_app_password() -> str:
    password = os.environ.get("APP_PASSWORD", "")
    if not password:
        raise RuntimeError("APP_PASSWORD must be set before starting the server")
    return password


def make_session_token(password: str, *, now: int | None = None) -> str:
    timestamp = str(int(time.time() if now is None else now))
    signature = hmac.new(password.encode(), timestamp.encode(), hashlib.sha256).hexdigest()
    return f"{timestamp}.{signature}"


def verify_session_token(token: str | None, password: str, *, now: int | None = None) -> bool:
    if not token:
        return False
    try:
        raw_timestamp, signature = token.split(".", maxsplit=1)
        timestamp = int(raw_timestamp)
    except (ValueError, TypeError):
        return False
    current = int(time.time() if now is None else now)
    if timestamp > current or current - timestamp > SESSION_TTL_SECONDS:
        return False
    expected = hmac.new(password.encode(), raw_timestamp.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def safe_next_path(value: str) -> str:
    return value if value.startswith("/") and not value.startswith("//") else "/"


class PasswordAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, password: str) -> None:
        super().__init__(app)
        self.password = password

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path == "/login" or path.startswith("/static/"):
            return await call_next(request)
        if verify_session_token(request.cookies.get(SESSION_COOKIE), self.password):
            return await call_next(request)
        if request.method == "GET" and not path.startswith("/api/"):
            target = path
            if request.url.query:
                target = f"{target}?{request.url.query}"
            return RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
