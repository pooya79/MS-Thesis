"""Evaluate Ivira Avanegar on a mixed ASR test dataset.

The evaluator uses Avanegar's synchronous short-audio endpoint. Requests are
sequential and checkpointed individually, and provider-reported units are
tracked as the authoritative run-usage counter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import os
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
import jiwer
from mutagen import File as MutagenFile

from ml.asr.eval_openrouter_stt import (
    BudgetReached,
    EvalRow,
    append_jsonl,
    log_event,
    normalize_for_scoring,
    read_mixed_test_rows,
    read_predictions,
    utc_now,
    write_json_atomic,
)


AVANEGAR_BASE_URL = "https://partai.gw.isahab.ir"
AVANEGAR_ENDPOINT = "/avanegar/avanegar/request"
DEFAULT_MODEL = "default"
LOG = logging.getLogger(__name__)


def parse_positive_decimal(raw: str, *, name: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a decimal number") from exc
    if not value.is_finite() or value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be greater than zero")
    return value


def audio_duration_seconds(path: Path) -> float:
    audio = MutagenFile(path)
    if audio is None or audio.info is None:
        raise ValueError(f"could not determine audio duration: {path}")
    duration = float(audio.info.length)
    if duration <= 0:
        raise ValueError(f"audio duration must be positive: {path}")
    return duration


def configure_logging(output_dir: Path) -> None:
    log_path = output_dir / "logs" / "ivira_avanegar.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _response_data(payload: dict[str, Any]) -> tuple[str, Decimal]:
    try:
        request_data = payload["data"]["data"]
        ai_response = request_data["aiResponse"]
        text = ai_response["result"]["text"]
        raw_units = ai_response["meta"]["units"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Avanegar response is missing data.data.aiResponse result or units") from exc
    if not isinstance(text, str):
        raise ValueError("Avanegar response transcript is not a string")
    if isinstance(raw_units, bool):
        raise ValueError("Avanegar response units are not numeric")
    try:
        units = Decimal(str(raw_units))
    except InvalidOperation as exc:
        raise ValueError("Avanegar response units are not numeric") from exc
    if not units.is_finite() or units < 0:
        raise ValueError("Avanegar response units must be a non-negative number")
    return text, units


class AvanegarClient:
    def __init__(
        self,
        gateway_token: str,
        *,
        timeout: float,
        attempts: int,
        retry_delay: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.client = httpx.Client(
            base_url=AVANEGAR_BASE_URL,
            headers={"gateway-token": gateway_token, "accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> AvanegarClient:
        self.client.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.client.__exit__(*args)

    def transcribe(self, model: str, row: EvalRow) -> dict[str, Any]:
        form = {
            "model": model,
            "srt": "false",
            "inverseNormalizer": "false",
            "timestamp": "false",
            "spokenPunctuation": "false",
            "punctuation": "false",
            "numSpeakers": "0",
            "diarize": "false",
        }
        content_type = mimetypes.guess_type(row.audio_path.name)[0] or "application/octet-stream"
        for attempt in range(1, self.attempts + 1):
            try:
                # Multipart streams must be reopened so retries resend the
                # complete clip rather than an exhausted file object.
                with row.audio_path.open("rb") as audio:
                    response = self.client.post(
                        AVANEGAR_ENDPOINT,
                        data=form,
                        files={"audio": (row.audio_path.name, audio, content_type)},
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Avanegar returned non-object JSON")
                text, units = _response_data(payload)
                return {
                    "text": text,
                    "units": str(units),
                    "provider_response": payload,
                    "response_headers": {
                        name: response.headers[name]
                        for name in ("x-request-id", "x-trace-id", "trace-id")
                        if name in response.headers
                    },
                }
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                retryable = status is None or status == 429 or status >= 500
                if attempt >= self.attempts or not retryable:
                    raise
                delay = self.retry_delay * (2 ** (attempt - 1))
                LOG.warning(
                    "transcription request failed attempt=%s/%s status=%s retry_in=%.1fs error=%s; provider units may already have been charged",
                    attempt,
                    self.attempts,
                    status,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")


def prediction_record(
    *,
    model: str,
    row: EvalRow,
    response: dict[str, Any],
    duration_seconds: float,
) -> dict[str, Any]:
    raw_prediction = str(response["text"])
    reference_normalized = normalize_for_scoring(row.reference)
    prediction_normalized = normalize_for_scoring(raw_prediction)
    return {
        "timestamp": utc_now(),
        "model": model,
        "row_index": row.index,
        "path": row.path,
        "source_dataset": row.source_dataset,
        "reference": row.reference,
        "prediction": raw_prediction,
        "reference_normalized": reference_normalized,
        "prediction_normalized": prediction_normalized,
        "wer": float(jiwer.wer(reference_normalized, prediction_normalized)),
        "cer": float(jiwer.cer(reference_normalized, prediction_normalized)),
        "audio_duration_seconds": duration_seconds,
        "provider_units": float(Decimal(str(response["units"]))),
        "response_headers": response["response_headers"],
        "provider_response": response["provider_response"],
        "request_options": {
            "punctuation": False,
            "spokenPunctuation": False,
            "inverseNormalizer": False,
            "diarize": False,
            "timestamp": False,
            "srt": False,
            "numSpeakers": 0,
        },
    }


def score_records(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    references = [str(record["reference_normalized"]) for record in records]
    predictions = [str(record["prediction_normalized"]) for record in records]
    return {
        "examples": len(records),
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
        "audio_duration_seconds": float(
            sum(float(record["audio_duration_seconds"]) for record in records)
        ),
        "provider_units": float(
            sum(Decimal(str(record["provider_units"])) for record in records)
        ),
    }


def build_metrics(
    records: Sequence[dict[str, Any]], expected_rows: int, model: str
) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[str(record["source_dataset"])].append(record)
    result: dict[str, Any] = {
        "updated_at": utc_now(),
        "model": model,
        "complete": len(records) == expected_rows,
        "expected_examples": expected_rows,
        "completed_examples": len(records),
        "datasets": [
            {"source_dataset": name, **score_records(dataset_records)}
            for name, dataset_records in sorted(by_dataset.items())
        ],
    }
    if records:
        result["overall"] = score_records(records)
    return result


def write_predictions_tsv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    import csv

    fields = [
        "model",
        "row_index",
        "path",
        "source_dataset",
        "reference",
        "prediction",
        "reference_normalized",
        "prediction_normalized",
        "wer",
        "cer",
        "audio_duration_seconds",
        "provider_units",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def evaluate(
    *,
    dataset_root: Path,
    output_dir: Path,
    max_run_units: Decimal,
    model: str = DEFAULT_MODEL,
    max_audio_seconds: float = 60,
    resume: bool = False,
    timeout: float = 120,
    attempts: int = 3,
    retry_delay: float = 2,
    gateway_token: str | None = None,
    client_factory: Callable[..., AvanegarClient] = AvanegarClient,
    duration_reader: Callable[[Path], float] = audio_duration_seconds,
) -> int:
    rows = read_mixed_test_rows(dataset_root)
    manifest_sha256 = hashlib.sha256((dataset_root / "test.tsv").read_bytes()).hexdigest()
    immutable = {
        "dataset_root": str(dataset_root.resolve()),
        "test_tsv_sha256": manifest_sha256,
        "model": model,
        "max_audio_seconds": max_audio_seconds,
        "punctuation": False,
        "spoken_punctuation": False,
        "inverse_normalizer": False,
        "diarize": False,
        "timestamp": False,
        "srt": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(immutable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config_path = output_dir / "run_config.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    events_path = output_dir / "events.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    if resume:
        if not config_path.is_file():
            raise ValueError(f"cannot resume without {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("fingerprint") != fingerprint:
            raise ValueError("resume configuration or test.tsv differs from the original run")
    else:
        config = {
            "created_at": utc_now(),
            "fingerprint": fingerprint,
            **immutable,
            "max_run_units": str(max_run_units),
            "requests_are_sequential": True,
            "usage_note": "Avanegar response meta.units is tracked exactly; the public API document does not expose an account-balance endpoint or a USD conversion.",
            "budget_note": "The run-unit cap is checked between requests, so one in-flight request may cross it. Use a dedicated token/provider quota if available for a remote hard cap.",
        }
        write_json_atomic(config_path, config)

    records = read_predictions(predictions_path)
    completed = {int(record["row_index"]) for record in records}
    if len(completed) != len(records):
        raise ValueError("predictions.jsonl contains duplicate row results")
    run_units = sum(Decimal(str(record["provider_units"])) for record in records)
    secret = gateway_token or os.environ.get("IVIRA_GATEWAY_TOKEN")
    if not secret:
        raise ValueError("IVIRA_GATEWAY_TOKEN is not set")

    stopped_for_budget = False
    failures = 0
    log_event(
        events_path,
        "run_resumed" if resume else "run_started",
        completed=len(records),
        provider_units=str(run_units),
        max_run_units=str(max_run_units),
    )
    LOG.info(
        "evaluation start model=%s examples=%s completed=%s provider_units=%s cap=%s",
        model,
        len(rows),
        len(records),
        run_units,
        max_run_units,
    )
    with client_factory(
        secret, timeout=timeout, attempts=attempts, retry_delay=retry_delay
    ) as client:
        try:
            for row in rows:
                if row.index in completed:
                    continue
                LOG.info(
                    "budget row=%s provider_units=%s/%s",
                    row.index,
                    run_units,
                    max_run_units,
                )
                log_event(
                    events_path,
                    "budget_checked",
                    row_index=row.index,
                    provider_units=str(run_units),
                    max_run_units=str(max_run_units),
                )
                if run_units >= max_run_units:
                    raise BudgetReached(
                        f"provider units {run_units} reached run cap {max_run_units}"
                    )
                try:
                    duration = duration_reader(row.audio_path)
                    if duration >= max_audio_seconds:
                        raise ValueError(
                            f"clip duration {duration:.3f}s must be below Avanegar's {max_audio_seconds:g}s short-audio limit"
                        )
                    LOG.info(
                        "transcribing row=%s/%s dataset=%s path=%s duration_seconds=%.3f punctuation=false spoken_punctuation=false",
                        row.index + 1,
                        len(rows),
                        row.source_dataset,
                        row.path,
                        duration,
                    )
                    log_event(
                        events_path,
                        "request_started",
                        row_index=row.index,
                        source_dataset=row.source_dataset,
                        path=row.path,
                        duration_seconds=duration,
                        punctuation=False,
                        spoken_punctuation=False,
                    )
                    response = client.transcribe(model, row)
                    record = prediction_record(
                        model=model,
                        row=row,
                        response=response,
                        duration_seconds=duration,
                    )
                except Exception as exc:
                    failures += 1
                    LOG.exception("transcription failed row=%s path=%s", row.index, row.path)
                    log_event(
                        events_path,
                        "request_failed",
                        row_index=row.index,
                        path=row.path,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    continue
                append_jsonl(predictions_path, record)
                records.append(record)
                completed.add(row.index)
                run_units += Decimal(str(record["provider_units"]))
                LOG.info(
                    "transcribed row=%s prediction=%r request_units=%s run_units=%s",
                    row.index,
                    record["prediction"],
                    record["provider_units"],
                    run_units,
                )
                log_event(
                    events_path,
                    "request_succeeded",
                    row_index=row.index,
                    prediction=record["prediction"],
                    request_units=str(record["provider_units"]),
                    run_units=str(run_units),
                    response_headers=record["response_headers"],
                )
        except BudgetReached as exc:
            stopped_for_budget = True
            LOG.warning("stopping safely: %s", exc)
            log_event(events_path, "budget_stop", reason=str(exc))
        finally:
            metrics = build_metrics(records, len(rows), model)
            metrics["stopped_for_budget"] = stopped_for_budget
            metrics["failed_requests"] = failures
            metrics["provider_units_used"] = float(run_units)
            metrics["max_run_units"] = float(max_run_units)
            write_json_atomic(output_dir / "metrics.json", metrics)
            write_predictions_tsv(output_dir / "predictions.tsv", records)
            log_event(
                events_path,
                "run_finished",
                complete=metrics["complete"],
                stopped_for_budget=stopped_for_budget,
                failed_requests=failures,
                provider_units=str(run_units),
            )
            LOG.info(
                "evaluation finish complete=%s failures=%s stopped_for_budget=%s provider_units=%s",
                metrics["complete"],
                failures,
                stopped_for_budget,
                run_units,
            )
    return 0 if metrics["complete"] else 2 if stopped_for_budget else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Ivira Avanegar on a mixed test.tsv with punctuation disabled, exact predictions, per-source WER/CER, resumable logs, and a provider-unit guard."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Mixed dataset containing test.tsv and clips/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for predictions, metrics, checkpoints, and logs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["default", "microphony", "telephony"],
        help="Avanegar acoustic model (default: default; provider also documents microphony and telephony).",
    )
    parser.add_argument(
        "--max-run-units",
        required=True,
        help="Required positive cap on provider-reported units during this run; one request can overshoot.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=60,
        help="Reject clips at or above this duration before upload (default: 60, matching the short-audio API).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the identical run and skip checkpointed clips.",
    )
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds (default: 120).")
    parser.add_argument("--attempts", type=int, default=3, help="Maximum attempts for retryable requests (default: 3).")
    parser.add_argument("--retry-delay", type=float, default=2, help="Initial exponential-backoff delay in seconds (default: 2).")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.attempts < 1 or args.retry_delay < 0:
        parser.error("--timeout must be > 0, --attempts >= 1, and --retry-delay >= 0")
    if args.max_audio_seconds <= 0 or args.max_audio_seconds > 60:
        parser.error("--max-audio-seconds must be > 0 and <= 60")
    try:
        return evaluate(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            max_run_units=parse_positive_decimal(args.max_run_units, name="--max-run-units"),
            model=args.model,
            max_audio_seconds=args.max_audio_seconds,
            resume=args.resume,
            timeout=args.timeout,
            attempts=args.attempts,
            retry_delay=args.retry_delay,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
