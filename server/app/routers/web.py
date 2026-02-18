from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from server.app.core.auth import sanitize_next_path, verify_password
from server.app.core.config import get_settings

settings = get_settings()
templates = Jinja2Templates(directory=str(Path(settings.template_dir)))

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/login", response_class=HTMLResponse, name="login_page")
def login_page(request: Request, next: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "app_name": settings.app_name,
            "error": None,
            "next_path": sanitize_next_path(next),
        },
    )


@router.post("/login", response_class=HTMLResponse, name="login_submit")
async def login_submit(request: Request) -> Response:
    form = await request.form()
    password = str(form.get("password", ""))
    next_path = sanitize_next_path(str(form.get("next", "")))

    if verify_password(password):
        request.session["authenticated"] = True
        return RedirectResponse(url=next_path, status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "app_name": settings.app_name,
            "error": "Invalid password.",
            "next_path": next_path,
        },
        status_code=401,
    )


@router.post("/logout", name="logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@router.get("/", response_class=HTMLResponse)
def homepage(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "active_nav": "home",
            "subtitle": "Archive findings, demo implementations, and publish research blogs.",
        },
    )
