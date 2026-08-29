"""Evaluate OpenRouter speech-to-text models on a mixed ASR test dataset.

The evaluator is intentionally sequential: request-level cost is checkpointed after
every successful transcription, making spend and resume behaviour easy to audit.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
import jiwer

from ml.speech_data.text_normalization import normalize_persian_asr_text


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalRow:
    index: int
    path: str
    audio_path: Path
    reference: str
    source_dataset: str


class BudgetReached(RuntimeError):
    """Raised when the configured local or remote spend guard is reached."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_usd(raw: str, *, name: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a decimal number") from exc
    if not value.is_finite() or value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be greater than zero")
    return value


def normalize_for_scoring(text: str) -> str:
    return normalize_persian_asr_text(text) or ""


def resolve_audio_path(dataset_root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"audio path must be relative: {raw_path}")
    root = dataset_root.resolve()
    candidates = [dataset_root / "clips" / relative, dataset_root / relative]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            return resolved
    raise FileNotFoundError(f"audio was not found under {dataset_root}: {raw_path}")


def read_mixed_test_rows(dataset_root: Path) -> list[EvalRow]:
    test_path = dataset_root / "test.tsv"
    if not test_path.is_file():
        raise FileNotFoundError(f"missing mixed test manifest: {test_path}")
    rows: list[EvalRow] = []
    with test_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"path", "sentence", "source_dataset"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{test_path} must contain path, sentence, and source_dataset columns")
        for index, row in enumerate(reader):
            raw_path = (row.get("path") or "").strip()
            source_dataset = (row.get("source_dataset") or "").strip()
            if not raw_path or not source_dataset:
                raise ValueError(f"empty path or source_dataset at {test_path}:{index + 2}")
            rows.append(
                EvalRow(
                    index=index,
                    path=raw_path,
                    audio_path=resolve_audio_path(dataset_root, raw_path),
                    reference=row.get("sentence") or "",
                    source_dataset=source_dataset,
                )
            )
    if not rows:
        raise ValueError(f"{test_path} contains no examples")
    return rows


def audio_format(path: Path) -> str:
    extension = path.suffix.lower().lstrip(".")
    if extension == "wave":
        extension = "wav"
    if extension not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"unsupported OpenRouter STT audio format for {path}; expected one of "
            f"{', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )
    return extension


def config_fingerprint(dataset_root: Path, models: Sequence[str], language: str) -> tuple[str, str]:
    manifest = (dataset_root / "test.tsv").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    payload = {
        "dataset_root": str(dataset_root.resolve()),
        "manifest_sha256": manifest_sha256,
        "models": list(models),
        "language": language,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, manifest_sha256


def configure_logging(output_dir: Path) -> None:
    log_path = output_dir / "logs" / "openrouter_stt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def log_event(events_path: Path, event: str, **fields: Any) -> None:
    append_jsonl(events_path, {"timestamp": utc_now(), "event": event, **fields})


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    predictions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            predictions.append(record)
    return predictions


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    attempts: int,
    retry_delay: float,
    **kwargs: Any,
) -> dict[str, Any]:
    for attempt in range(1, attempts + 1):
        try:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"OpenRouter returned a non-object JSON response from {url}")
            return payload
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            retryable = status is None or status == 429 or status >= 500
            if attempt >= attempts or not retryable:
                raise
            delay = retry_delay * (2 ** (attempt - 1))
            LOG.warning("request failed attempt=%s/%s status=%s retry_in=%.1fs error=%s", attempt, attempts, status, delay, exc)
            time.sleep(delay)
    raise AssertionError("unreachable")


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float,
        attempts: int,
        retry_delay: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.client = httpx.Client(
            base_url=OPENROUTER_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> OpenRouterClient:
        self.client.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.client.__exit__(*args)

    def key_status(self) -> dict[str, Any]:
        payload = request_json(
            self.client,
            "GET",
            "/key",
            attempts=self.attempts,
            retry_delay=self.retry_delay,
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("OpenRouter /key response has no data object")
        return data

    def transcribe(self, model: str, row: EvalRow, language: str) -> dict[str, Any]:
        payload = {
            "model": model,
            "input_audio": {
                "data": base64.b64encode(row.audio_path.read_bytes()).decode("ascii"),
                "format": audio_format(row.audio_path),
            },
            "language": language,
            "temperature": 0,
        }
        response = request_json(
            self.client,
            "POST",
            "/audio/transcriptions",
            attempts=self.attempts,
            retry_delay=self.retry_delay,
            json=payload,
        )
        if not isinstance(response.get("text"), str):
            raise ValueError("OpenRouter transcription response has no text string")
        usage = response.get("usage")
        if not isinstance(usage, dict) or not isinstance(usage.get("cost"), (int, float)):
            raise ValueError("OpenRouter transcription response has no numeric usage.cost")
        cost = float(usage["cost"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("OpenRouter transcription response has invalid usage.cost")
        return response


def prediction_record(model: str, row: EvalRow, response: dict[str, Any]) -> dict[str, Any]:
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
        "cost_usd": float(response["usage"]["cost"]),
        "usage": response["usage"],
    }


def score_records(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    references = [str(record["reference_normalized"]) for record in records]
    predictions = [str(record["prediction_normalized"]) for record in records]
    return {
        "examples": len(records),
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
        "cost_usd": float(sum(Decimal(str(record["cost_usd"])) for record in records)),
    }


def build_metrics(
    records: Sequence[dict[str, Any]], models: Sequence[str], expected_rows: int
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[str(record["model"])].append(record)
    model_metrics: list[dict[str, Any]] = []
    for model in models:
        model_records = by_model[model]
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in model_records:
            by_dataset[str(record["source_dataset"])].append(record)
        entry: dict[str, Any] = {
            "model": model,
            "complete": len(model_records) == expected_rows,
            "expected_examples": expected_rows,
            "completed_examples": len(model_records),
            "datasets": [
                {"source_dataset": dataset, **score_records(dataset_records)}
                for dataset, dataset_records in sorted(by_dataset.items())
            ],
        }
        if model_records:
            entry["overall"] = score_records(model_records)
        model_metrics.append(entry)
    return {
        "updated_at": utc_now(),
        "complete": all(item["complete"] for item in model_metrics),
        "total_cost_usd": float(sum(Decimal(str(r["cost_usd"])) for r in records)),
        "models": model_metrics,
    }


def write_predictions_tsv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "model", "row_index", "path", "source_dataset", "reference", "prediction",
        "reference_normalized", "prediction_normalized", "wer", "cer", "cost_usd",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def validate_models(models: Sequence[str]) -> list[str]:
    cleaned = [model.strip() for model in models]
    if not cleaned or any(not model for model in cleaned):
        raise ValueError("at least one non-empty model is required")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("model names must be unique")
    return cleaned


def evaluate(
    *,
    dataset_root: Path,
    models: Sequence[str],
    output_dir: Path,
    max_run_cost: Decimal,
    language: str = "fa",
    min_key_remaining: Decimal | None = None,
    resume: bool = False,
    timeout: float = 120,
    attempts: int = 3,
    retry_delay: float = 2,
    api_key: str | None = None,
    client_factory: Callable[..., OpenRouterClient] = OpenRouterClient,
) -> int:
    models = validate_models(models)
    rows = read_mixed_test_rows(dataset_root)
    fingerprint, manifest_sha256 = config_fingerprint(dataset_root, models, language)
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
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("fingerprint") != fingerprint:
            raise ValueError("resume configuration or test.tsv differs from the original run")
    else:
        write_json_atomic(
            config_path,
            {
                "created_at": utc_now(),
                "fingerprint": fingerprint,
                "dataset_root": str(dataset_root.resolve()),
                "test_tsv_sha256": manifest_sha256,
                "models": models,
                "language": language,
                "max_run_cost_usd": str(max_run_cost),
                "min_key_remaining_usd": str(min_key_remaining) if min_key_remaining else None,
                "requests_are_sequential": True,
                "budget_note": "The local cap is checked between requests; one in-flight request may cross it. Use an OpenRouter key spending limit for a server-enforced hard cap.",
            },
        )

    records = read_predictions(predictions_path)
    completed = {(str(record["model"]), int(record["row_index"])) for record in records}
    if len(completed) != len(records):
        raise ValueError("predictions.jsonl contains duplicate model/row results")
    run_cost = sum(Decimal(str(record["cost_usd"])) for record in records)
    secret = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not secret:
        raise ValueError("OPENROUTER_API_KEY is not set")

    log_event(
        events_path,
        "run_started" if not resume else "run_resumed",
        completed=len(records),
        run_cost_usd=str(run_cost),
        max_run_cost_usd=str(max_run_cost),
        min_key_remaining_usd=str(min_key_remaining) if min_key_remaining else None,
    )
    LOG.info("evaluation start models=%s examples=%s completed=%s run_cost_usd=%s", len(models), len(rows), len(records), run_cost)
    stopped_for_budget = False
    failures = 0
    try:
        with client_factory(secret, timeout=timeout, attempts=attempts, retry_delay=retry_delay) as client:
            for model in models:
                for row in rows:
                    if (model, row.index) in completed:
                        continue
                    if run_cost >= max_run_cost:
                        raise BudgetReached(f"run cost {run_cost} reached cap {max_run_cost}")
                    key_status = client.key_status()
                    key_usage = key_status.get("usage")
                    key_remaining = key_status.get("limit_remaining")
                    LOG.info(
                        "budget model=%s row=%s run_cost_usd=%s cap_usd=%s key_usage_usd=%s key_limit_remaining_usd=%s",
                        model, row.index, run_cost, max_run_cost, key_usage, key_remaining,
                    )
                    log_event(events_path, "budget_checked", model=model, row_index=row.index, run_cost_usd=str(run_cost), max_run_cost_usd=str(max_run_cost), key_usage_usd=key_usage, key_limit_remaining_usd=key_remaining)
                    if min_key_remaining is not None:
                        if key_remaining is None:
                            raise BudgetReached("key has no limit_remaining value; cannot enforce --min-key-remaining-usd")
                        if Decimal(str(key_remaining)) <= min_key_remaining:
                            raise BudgetReached(f"key remaining {key_remaining} reached minimum {min_key_remaining}")
                    LOG.info("transcribing model=%s row=%s/%s dataset=%s path=%s", model, row.index + 1, len(rows), row.source_dataset, row.path)
                    log_event(events_path, "request_started", model=model, row_index=row.index, source_dataset=row.source_dataset, path=row.path)
                    try:
                        response = client.transcribe(model, row, language)
                        record = prediction_record(model, row, response)
                    except Exception as exc:
                        failures += 1
                        LOG.exception("transcription failed model=%s row=%s path=%s", model, row.index, row.path)
                        log_event(events_path, "request_failed", model=model, row_index=row.index, path=row.path, error_type=type(exc).__name__, error=str(exc))
                        continue
                    append_jsonl(predictions_path, record)
                    records.append(record)
                    completed.add((model, row.index))
                    cost = Decimal(str(record["cost_usd"]))
                    run_cost += cost
                    LOG.info("transcribed model=%s row=%s prediction=%r request_cost_usd=%s run_cost_usd=%s", model, row.index, record["prediction"], cost, run_cost)
                    log_event(events_path, "request_succeeded", model=model, row_index=row.index, prediction=record["prediction"], request_cost_usd=str(cost), run_cost_usd=str(run_cost), usage=record["usage"])
    except BudgetReached as exc:
        stopped_for_budget = True
        LOG.warning("stopping safely: %s", exc)
        log_event(events_path, "budget_stop", reason=str(exc), run_cost_usd=str(run_cost))
    finally:
        metrics = build_metrics(records, models, len(rows))
        metrics["stopped_for_budget"] = stopped_for_budget
        metrics["failed_requests"] = failures
        write_json_atomic(output_dir / "metrics.json", metrics)
        write_predictions_tsv(output_dir / "predictions.tsv", records)
        log_event(events_path, "run_finished", complete=metrics["complete"], stopped_for_budget=stopped_for_budget, failed_requests=failures, run_cost_usd=str(run_cost))
        LOG.info("evaluation finish complete=%s failures=%s stopped_for_budget=%s run_cost_usd=%s", metrics["complete"], failures, stopped_for_budget, run_cost)
    return 0 if metrics["complete"] else 2 if stopped_for_budget else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more OpenRouter speech-to-text models on a mixed test.tsv, with resumable predictions, per-source WER/CER, detailed logs, and spend guards."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Mixed dataset containing test.tsv and clips/.")
    parser.add_argument("--model", action="append", required=True, help="OpenRouter STT model slug. Repeat for every model.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for predictions, metrics, checkpoints, and logs.")
    parser.add_argument("--max-run-cost-usd", required=True, help="Required positive local USD cap; checked before every request (one request can overshoot).")
    parser.add_argument("--min-key-remaining-usd", help="Stop when the current key's limit_remaining reaches this positive USD amount. Requires a spending limit on the key.")
    parser.add_argument("--language", default="fa", help="ISO-639-1 transcription language (default: fa).")
    parser.add_argument("--resume", action="store_true", help="Resume the identical run and skip checkpointed model/clip pairs.")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds (default: 120).")
    parser.add_argument("--attempts", type=int, default=3, help="Maximum attempts for retryable requests (default: 3).")
    parser.add_argument("--retry-delay", type=float, default=2, help="Initial exponential-backoff delay in seconds (default: 2).")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.attempts < 1 or args.retry_delay < 0:
        parser.error("--timeout must be > 0, --attempts >= 1, and --retry-delay >= 0")
    max_run_cost = parse_usd(args.max_run_cost_usd, name="--max-run-cost-usd")
    min_key_remaining = (
        parse_usd(args.min_key_remaining_usd, name="--min-key-remaining-usd")
        if args.min_key_remaining_usd is not None
        else None
    )
    try:
        return evaluate(
            dataset_root=args.dataset_root,
            models=args.model,
            output_dir=args.output_dir,
            max_run_cost=max_run_cost,
            language=args.language,
            min_key_remaining=min_key_remaining,
            resume=args.resume,
            timeout=args.timeout,
            attempts=args.attempts,
            retry_delay=args.retry_delay,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
