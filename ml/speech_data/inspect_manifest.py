from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def inspect_manifest(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    counters = {
        "profile": Counter(),
        "channel_path": Counter(),
        "codec": Counter(),
        "codec_bitrate": Counter(),
        "reverb_mode": Counter(),
        "network_impairment": Counter(),
        "target_bandwidth": Counter(),
    }
    missing_files = 0
    unreadable_files = 0
    length_mismatches = 0
    total_duration = 0.0
    snr_values: list[float] = []
    loss_values: list[float] = []

    for row in rows:
        for key, counter in counters.items():
            if key == "network_impairment":
                value = (row.get("network_impairment") or {}).get("mode") or "none"
            else:
                value = row.get(key) or "none"
            counter[str(value)] += 1
        if row.get("snr_db") is not None:
            snr_values.append(float(row["snr_db"]))
        network = row.get("network_impairment") or {}
        if network.get("observed_loss_rate") is not None:
            loss_values.append(float(network["observed_loss_rate"]))
        clean_path = Path(row["clean_path"])
        degraded_path = Path(row["degraded_path"])
        if not clean_path.exists() or not degraded_path.exists():
            missing_files += 1
            continue
        try:
            clean_info = sf.info(str(clean_path))
            degraded_info = sf.info(str(degraded_path))
            total_duration += degraded_info.duration
            if clean_info.frames != degraded_info.frames or clean_info.samplerate != degraded_info.samplerate:
                length_mismatches += 1
        except sf.LibsndfileError:
            unreadable_files += 1

    return {
        "manifest": str(path),
        "pairs": len(rows),
        "total_hours": total_duration / 3600,
        "missing_files": missing_files,
        "unreadable_files": unreadable_files,
        "length_mismatches": length_mismatches,
        "distributions": {name: dict(counter) for name, counter in counters.items()},
        "snr_db": {
            "count": len(snr_values),
            "min": min(snr_values) if snr_values else None,
            "max": max(snr_values) if snr_values else None,
            "mean": sum(snr_values) / len(snr_values) if snr_values else None,
        },
        "observed_loss_rate": {
            "count": len(loss_values),
            "min": min(loss_values) if loss_values else None,
            "max": max(loss_values) if loss_values else None,
            "mean": sum(loss_values) / len(loss_values) if loss_values else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a generated speech-enhancement pair manifest.")
    parser.add_argument("manifest", help="Path to se_train_pairs.jsonl or se_valid_pairs.jsonl.")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_manifest(Path(args.manifest)), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
