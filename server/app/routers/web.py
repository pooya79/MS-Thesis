from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from server.app.core.auth import sanitize_next_path, verify_password
from server.app.core.config import get_settings
from server.app.services.speech_degradation_demo import (
    DemoValidationError,
    FILE_KINDS,
    cleanup_session_demos,
    demo_file_path,
    generate_demo,
)

settings = get_settings()
templates = Jinja2Templates(directory=str(Path(settings.template_dir)))

router = APIRouter()


def speech_degradation_context(
    request: Request,
    error: str | None = None,
    result: dict[str, Any] | None = None,
    metadata_json: str | None = None,
    form_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "app_name": settings.app_name,
        "active_nav": "experiments",
        "active_sidebar": "speech_degradation",
        "error": error,
        "result": result,
        "metadata_json": metadata_json,
        "form_values": form_values
        or {
            "noise_enabled": False,
            "snr_bucket": "5:10",
            "gain_db": "0",
            "clipping_enabled": False,
            "channel_path": "narrowband",
            "codec": "g711_alaw",
            "network_enabled": False,
        },
    }


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


@router.get("/experiments/speech-degradation", response_class=HTMLResponse)
def speech_degradation_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="speech_degradation.html",
        context=speech_degradation_context(request),
    )


@router.post("/experiments/speech-degradation/generate", response_class=HTMLResponse)
async def generate_speech_degradation_demo(
    request: Request,
    audio_file: UploadFile = File(...),
    noise_enabled: str | None = Form(None),
    snr_bucket: str = Form("5:10"),
    gain_db: str = Form("0"),
    clipping_enabled: str | None = Form(None),
    channel_path: str = Form("narrowband"),
    codec: str = Form("g711_alaw"),
    network_enabled: str | None = Form(None),
) -> HTMLResponse:
    form_values = {
        "noise_enabled": noise_enabled is not None,
        "snr_bucket": snr_bucket,
        "gain_db": gain_db,
        "clipping_enabled": clipping_enabled is not None,
        "channel_path": channel_path,
        "codec": codec,
        "network_enabled": network_enabled is not None,
    }
    try:
        content = await audio_file.read()
        existing_demo_ids = list(request.session.get("degradation_demo_ids", []))
        demo = generate_demo(content, audio_file.filename or "", form_values, existing_demo_ids)
        demo_ids = cleanup_session_demos([demo.demo_id, *existing_demo_ids])
        request.session["degradation_demo_ids"] = demo_ids
        metadata_json = demo.demo_dir.joinpath("metadata.json").read_text(encoding="utf-8")
        result = {
            "demo_id": demo.demo_id,
            "input_url": f"/experiments/speech-degradation/files/{demo.demo_id}/input",
            "clean_target_url": f"/experiments/speech-degradation/files/{demo.demo_id}/clean_target",
            "degraded_url": f"/experiments/speech-degradation/files/{demo.demo_id}/degraded",
            "metadata": demo.metadata,
        }
        status_code = 200
        error = None
    except (DemoValidationError, ValueError, RuntimeError) as exc:
        result = None
        metadata_json = None
        status_code = 400
        error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="speech_degradation.html",
        context=speech_degradation_context(
            request,
            error=error,
            result=result,
            metadata_json=metadata_json,
            form_values=form_values,
        ),
        status_code=status_code,
    )


@router.get("/experiments/speech-degradation/files/{demo_id}/{kind}")
def speech_degradation_file(request: Request, demo_id: str, kind: str) -> Response:
    if kind not in FILE_KINDS:
        return Response(status_code=404)
    if demo_id not in set(request.session.get("degradation_demo_ids", [])):
        return Response(status_code=404)
    path = demo_file_path(demo_id, kind)
    if not path.exists():
        return Response(status_code=404)
    return FileResponse(path, media_type="audio/wav", filename=path.name)
