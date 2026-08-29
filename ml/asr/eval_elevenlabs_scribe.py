"""Evaluate ElevenLabs Scribe on a mixed ASR test dataset.

Requests are sequential and checkpointed individually. USD cost is estimated
from audio duration because ElevenLabs does not return per-request cost in the
STT body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from decimal import Decimal
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
    parse_usd,
    read_mixed_test_rows,
    read_predictions,
    utc_now,
    write_json_atomic,
)


ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = "scribe_v2"
DEFAULT_PRICE_PER_HOUR_USD = Decimal("0.22")
LOG = logging.getLogger(__name__)


def audio_duration_seconds(path: Path) -> float:
    audio = MutagenFile(path)
    if audio is None or audio.info is None:
        raise ValueError(f"could not determine audio duration: {path}")
    duration = float(audio.info.length)
    if duration <= 0:
        raise ValueError(f"audio duration must be positive: {path}")
    return duration


def configure_logging(output_dir: Path) -> None:
    log_path = output_dir / "logs" / "elevenlabs_scribe.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


class ElevenLabsClient:
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
            base_url=ELEVENLABS_BASE_URL,
            headers={"xi-api-key": api_key},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> ElevenLabsClient:
        self.client.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.client.__exit__(*args)

    def transcribe(self, model: str, row: EvalRow, language: str, seed: int) -> dict[str, Any]:
        form = {
            "model_id": model,
            "language_code": language,
            "tag_audio_events": "false",
            "diarize": "false",
            "timestamps_granularity": "none",
            "temperature": "0",
            "seed": str(seed),
        }
        for attempt in range(1, self.attempts + 1):
            try:
                # Reopen the file for each attempt so a retry never uploads an
                # exhausted stream after the first multipart request.
                with row.audio_path.open("rb") as audio:
                    raw_response = self.client.post(
                        "/speech-to-text",
                        data=form,
                        files={
                            "file": (
                                row.audio_path.name,
                                audio,
                                "application/octet-stream",
                            )
                        },
                    )
                raw_response.raise_for_status()
                payload = raw_response.json()
                if not isinstance(payload, dict):
                    raise ValueError(
                        "ElevenLabs returned non-object JSON from /speech-to-text"
                    )
                headers = raw_response.headers
                break
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                status = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                retryable = status is None or status == 429 or status >= 500
                if attempt >= self.attempts or not retryable:
                    raise
                delay = self.retry_delay * (2 ** (attempt - 1))
                LOG.warning(
                    "transcription request failed attempt=%s/%s status=%s retry_in=%.1fs error=%s; provider billing may already have occurred",
                    attempt,
                    self.attempts,
                    status,
                    delay,
                    exc,
                )
                time.sleep(delay)
        else:
            raise AssertionError("unreachable")
        if not isinstance(payload.get("text"), str):
            raise ValueError("ElevenLabs transcription response has no text string")
        payload["_response_headers"] = {
            name: headers[name]
            for name in ("request-id", "x-trace-id")
            if name in headers
        }
        return payload


def prediction_record(
    *,
    model: str,
    row: EvalRow,
    response: dict[str, Any],
    duration_seconds: float,
    estimated_cost: Decimal,
) -> dict[str, Any]:
    raw_prediction = str(response["text"])
    reference_normalized = normalize_for_scoring(row.reference)
    prediction_normalized = normalize_for_scoring(raw_prediction)
    provider_response = {key: value for key, value in response.items() if key != "_response_headers"}
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
        "estimated_cost_usd": float(estimated_cost),
        "response_headers": response.get("_response_headers", {}),
        "provider_response": provider_response,
    }


def score_records(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    references = [str(record["reference_normalized"]) for record in records]
    predictions = [str(record["prediction_normalized"]) for record in records]
    return {
        "examples": len(records),
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
        "audio_duration_seconds": float(sum(float(record["audio_duration_seconds"]) for record in records)),
        "estimated_cost_usd": float(
            sum(Decimal(str(record["estimated_cost_usd"])) for record in records)
        ),
    }


def build_metrics(records: Sequence[dict[str, Any]], expected_rows: int, model: str) -> dict[str, Any]:
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
        "model", "row_index", "path", "source_dataset", "reference", "prediction",
        "reference_normalized", "prediction_normalized", "wer", "cer",
        "audio_duration_seconds", "estimated_cost_usd",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def evaluate(
    *,
    dataset_root: Path,
    output_dir: Path,
    max_estimated_cost: Decimal,
    model: str = DEFAULT_MODEL,
    language: str = "fa",
    seed: int = 0,
    price_per_hour: Decimal = DEFAULT_PRICE_PER_HOUR_USD,
    resume: bool = False,
    timeout: float = 120,
    attempts: int = 3,
    retry_delay: float = 2,
    api_key: str | None = None,
    client_factory: Callable[..., ElevenLabsClient] = ElevenLabsClient,
    duration_reader: Callable[[Path], float] = audio_duration_seconds,
) -> int:
    rows = read_mixed_test_rows(dataset_root)
    manifest_sha256 = hashlib.sha256((dataset_root / "test.tsv").read_bytes()).hexdigest()
    immutable = {
        "dataset_root": str(dataset_root.resolve()),
        "test_tsv_sha256": manifest_sha256,
        "model": model,
        "language": language,
        "seed": seed,
        "price_per_hour_usd": str(price_per_hour),
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
            "max_estimated_cost_usd": str(max_estimated_cost),
            "requests_are_sequential": True,
            "cost_note": "USD is estimated from duration and configured price; no account-credit endpoint is queried.",
            "budget_note": "The estimated-cost cap is checked before each request. Configure an API-key quota in ElevenLabs for a remote hard cap.",
        }
        write_json_atomic(config_path, config)

    records = read_predictions(predictions_path)
    completed = {int(record["row_index"]) for record in records}
    if len(completed) != len(records):
        raise ValueError("predictions.jsonl contains duplicate row results")
    estimated_cost = sum(Decimal(str(record["estimated_cost_usd"])) for record in records)
    secret = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not secret:
        raise ValueError("ELEVENLABS_API_KEY is not set")

    stopped_for_budget = False
    failures = 0
    with client_factory(secret, timeout=timeout, attempts=attempts, retry_delay=retry_delay) as client:
        log_event(
            events_path,
            "run_resumed" if resume else "run_started",
            completed=len(records),
            estimated_cost_usd=str(estimated_cost),
        )
        LOG.info(
            "evaluation start model=%s examples=%s completed=%s estimated_cost_usd=%s",
            model, len(rows), len(records), estimated_cost,
        )
        try:
            for row in rows:
                if row.index in completed:
                    continue
                duration = duration_reader(row.audio_path)
                request_estimate = price_per_hour * Decimal(str(duration)) / Decimal("3600")
                projected_cost = estimated_cost + request_estimate
                LOG.info(
                    "budget row=%s estimated_cost_usd=%s projected_cost_usd=%s cap_usd=%s",
                    row.index, estimated_cost, projected_cost, max_estimated_cost,
                )
                log_event(
                    events_path, "budget_checked", row_index=row.index,
                    estimated_cost_usd=str(estimated_cost),
                    request_estimated_cost_usd=str(request_estimate),
                    projected_cost_usd=str(projected_cost), max_estimated_cost_usd=str(max_estimated_cost),
                )
                if projected_cost > max_estimated_cost:
                    raise BudgetReached(
                        f"next request would raise estimated cost to {projected_cost}, above cap {max_estimated_cost}"
                    )
                LOG.info(
                    "transcribing row=%s/%s dataset=%s path=%s duration_seconds=%.3f",
                    row.index + 1, len(rows), row.source_dataset, row.path, duration,
                )
                log_event(
                    events_path, "request_started", row_index=row.index,
                    source_dataset=row.source_dataset, path=row.path, duration_seconds=duration,
                )
                try:
                    response = client.transcribe(model, row, language, seed)
                except Exception as exc:
                    failures += 1
                    LOG.exception("transcription failed row=%s path=%s", row.index, row.path)
                    log_event(
                        events_path, "request_failed", row_index=row.index, path=row.path,
                        error_type=type(exc).__name__, error=str(exc),
                    )
                    continue
                record = prediction_record(
                    model=model, row=row, response=response, duration_seconds=duration,
                    estimated_cost=request_estimate,
                )
                append_jsonl(predictions_path, record)
                records.append(record)
                completed.add(row.index)
                estimated_cost += request_estimate
                LOG.info(
                    "transcribed row=%s prediction=%r estimated_run_cost_usd=%s",
                    row.index, record["prediction"], estimated_cost,
                )
                log_event(
                    events_path, "request_succeeded", row_index=row.index,
                    prediction=record["prediction"], estimated_run_cost_usd=str(estimated_cost),
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
            write_json_atomic(output_dir / "metrics.json", metrics)
            write_predictions_tsv(output_dir / "predictions.tsv", records)
            log_event(
                events_path, "run_finished", complete=metrics["complete"],
                stopped_for_budget=stopped_for_budget, failed_requests=failures,
                estimated_cost_usd=str(estimated_cost),
            )
            LOG.info(
                "evaluation finish complete=%s failures=%s stopped_for_budget=%s estimated_cost_usd=%s",
                metrics["complete"], failures, stopped_for_budget, estimated_cost,
            )
    return 0 if metrics["complete"] else 2 if stopped_for_budget else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ElevenLabs Scribe v2 on a mixed test.tsv with exact predictions, per-source WER/CER, resumable logs, and an estimated-cost guard."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Mixed dataset containing test.tsv and clips/.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for predictions, metrics, checkpoints, and logs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["scribe_v2"], help="ElevenLabs batch STT model (default: scribe_v2).")
    parser.add_argument("--language", default="fa", help="ISO-639-1 or ISO-639-3 language code (default: fa).")
    parser.add_argument("--seed", type=int, default=0, help="Best-effort deterministic Scribe seed, 0-2147483647 (default: 0).")
    parser.add_argument("--max-estimated-cost-usd", required=True, help="Required positive estimated USD cap; the next clip is not sent if its duration would exceed this cap.")
    parser.add_argument("--price-per-hour-usd", default=str(DEFAULT_PRICE_PER_HOUR_USD), help="Scribe price used only for USD estimates (default: 0.22). Override for your contract/current price.")
    parser.add_argument("--resume", action="store_true", help="Resume the identical run and skip checkpointed clips.")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds (default: 120).")
    parser.add_argument("--attempts", type=int, default=3, help="Maximum attempts for retryable requests (default: 3).")
    parser.add_argument("--retry-delay", type=float, default=2, help="Initial exponential-backoff delay in seconds (default: 2).")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.attempts < 1 or args.retry_delay < 0:
        parser.error("--timeout must be > 0, --attempts >= 1, and --retry-delay >= 0")
    if not 0 <= args.seed <= 2_147_483_647:
        parser.error("--seed must be between 0 and 2147483647")
    try:
        max_cost = parse_usd(args.max_estimated_cost_usd, name="--max-estimated-cost-usd")
        price = parse_usd(args.price_per_hour_usd, name="--price-per-hour-usd")
        return evaluate(
            dataset_root=args.dataset_root, output_dir=args.output_dir,
            max_estimated_cost=max_cost,
            model=args.model, language=args.language, seed=args.seed, price_per_hour=price,
            resume=args.resume,
            timeout=args.timeout, attempts=args.attempts, retry_delay=args.retry_delay,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
