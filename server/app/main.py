from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from server.app.core.auth import PasswordAuthMiddleware
from server.app.core.config import get_settings
from server.app.routers.web import router as web_router

settings = get_settings()
static_dir = Path(__file__).resolve().parent / "static"

app = FastAPI(title=settings.app_name)
app.add_middleware(PasswordAuthMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.app_auth_secret,
    max_age=settings.session_max_age_seconds,
    same_site="lax",
    https_only=settings.environment == "prod",
)

app.mount(
    "/static",
    StaticFiles(directory=str(static_dir)),
    name="static",
)
app.include_router(web_router)
