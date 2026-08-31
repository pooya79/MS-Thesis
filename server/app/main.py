from __future__ import annotations

import hmac
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from server.app.core.auth import (
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    PasswordAuthMiddleware,
    make_session_token,
    require_app_password,
    safe_next_path,
)
from server.app.core.config import get_settings
from server.app.services.transcription import get_registry

settings = get_settings()
app_dir = Path(__file__).resolve().parent
templates = Environment(
    loader=FileSystemLoader(app_dir / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)

app = FastAPI(title=settings.title)
app.add_middleware(PasswordAuthMiddleware, password=require_app_password())
app.mount("/static", StaticFiles(directory=app_dir / "static"), name="static")


@app.get("/login", response_class=HTMLResponse)
def login_page(next: str = "/") -> str:
    return templates.get_template("login.html").render(error=None, next=safe_next_path(next))


@app.post("/login")
def login(request: Request, password: str = Form(...), next: str = Form("/")) -> Response:
    app_password = require_app_password()
    destination = safe_next_path(next)
    if not hmac.compare_digest(password.encode(), app_password.encode()):
        content = templates.get_template("login.html").render(
            error="Incorrect password.",
            next=destination,
        )
        return HTMLResponse(content, status_code=401)
    response = RedirectResponse(destination, status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        make_session_token(app_password),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
    )
    return response


@app.post("/logout")
def logout(request: Request) -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(
        SESSION_COOKIE,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
    )
    return response


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return templates.get_template("index.html").render(settings=settings)


@app.get("/api/models")
def list_models() -> dict[str, object]:
    return {
        "models": [
            {
                "id": model.id,
                "label": model.label,
                "description": model.description,
                "backend": model.backend,
            }
            for model in settings.models
        ]
    }


@app.get("/api/models/{model_id}/status")
def model_status(model_id: str) -> dict[str, object]:
    try:
        status = get_registry().status(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "model_id": status.model_id,
        "label": status.label,
        "state": status.state,
        "device": status.device,
    }


def _convert_to_wav(source: Path, destination: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
                "-ac", "1", "-ar", str(settings.sample_rate), str(destination),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise ValueError("ffmpeg is required to decode uploaded audio") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("the uploaded file is not a readable audio recording") from exc


@app.post("/api/transcriptions")
async def transcribe(model_id: str = Form(...), audio: UploadFile = File(...)) -> dict[str, object]:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Choose or record an audio file")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="asr-upload-") as temp_dir:
        source = Path(temp_dir) / "upload"
        size = 0
        with source.open("wb") as handle:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio must be smaller than {settings.max_upload_mb} MB",
                    )
                handle.write(chunk)
        await audio.close()
        wav_path = Path(temp_dir) / "audio.wav"
        try:
            await run_in_threadpool(_convert_to_wav, source, wav_path)
            result = await run_in_threadpool(get_registry().transcribe, model_id, wav_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (FileNotFoundError, RuntimeError, OSError, ImportError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "text": result.text,
        "model_id": result.model_id,
        "device": result.device,
        "duration_seconds": result.duration_seconds,
        "processing_seconds": result.processing_seconds,
    }
