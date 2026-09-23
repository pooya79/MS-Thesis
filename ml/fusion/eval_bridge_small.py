"""Evaluate the Whisper Small PQ+RI bridge on configured test datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ml.fusion.evaluate_ablation import run

DEFAULT_CONFIG = Path("configs/speech_enhancement/bridge_small_eval.yaml")


def load_eval_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Small bridge evaluation config must be a mapping")
    for section in ("model", "data", "eval"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
    data = config["data"]
    if data.get("split") != "test" or not isinstance(data.get("datasets"), list) or not data["datasets"]:
        raise ValueError("data.datasets must list held-out test datasets")
    if len(data["datasets"]) != len(set(data["datasets"])):
        raise ValueError("duplicate test datasets")
    for key in ("asr_checkpoint", "bridge_checkpoint", "processor"):
        if not config["model"].get(key):
            raise ValueError(f"model.{key} is required")
    if int(config["eval"].get("max_new_tokens", 0)) < 1:
        raise ValueError("eval.max_new_tokens must be positive")
    return config


def suite_config(config: dict) -> dict:
    """Bind the explicit evaluation YAML to the actual prepared enhancer identity."""
    path = Path(config["eval"]["inputs_dir"]) / "final_tests.yaml"
    generated = yaml.safe_load(path.read_text())
    expected = {"root_dir": str(Path(config["data"]["root_dir"]).resolve()),
                "datasets": config["data"]["datasets"], "split": "test"}
    if generated["data"] != expected:
        raise ValueError("evaluation datasets differ from prepared test audio")
    if generated["asr_checkpoint"] != config["model"]["asr_checkpoint"]:
        raise ValueError("evaluation ASR checkpoint differs from prepared inputs")
    for name in ("bridge", "enhanced_only"):
        if generated["methods"][name]["checkpoint"] != config["model"]["bridge_checkpoint"]:
            raise ValueError("evaluation bridge checkpoint differs from prepared inputs")
    generated["processor"] = config["model"]["processor"]
    generated["max_new_tokens"] = config["eval"]["max_new_tokens"]
    return generated


def write_comparison(output: Path, archive_root: Path = Path("report")) -> None:
    """Place archived metrics beside scores from the common current test set."""
    summary = json.loads((output / "summary.json").read_text())
    rows = []
    for name, result in summary.items():
        rows.append({"model": f"current/{name}", "source": str(output / f"{name}.metrics.json"),
                     "examples": result["examples"], "wer": result["wer"], "cer": result["cer"],
                     "datasets": result["dataset_metrics"], "matched_current_test_set": True})
    for path in sorted(archive_root.glob("*/metrics.json")):
        result = json.loads(path.read_text())
        if not isinstance(result.get("dataset_metrics"), list):
            continue
        datasets = {Path(item["dataset"]).name: {key: item[key] for key in ("examples", "wer", "cer")}
                    for item in result["dataset_metrics"]}
        rows.append({"model": f"archive/{path.parent.name}", "source": str(path),
                     "examples": result.get("examples"), "wer": result.get("wer"),
                     "cer": result.get("cer"), "datasets": datasets,
                     "matched_current_test_set": False})
    (output / "comparison.json").write_text(json.dumps({
        "note": "Current methods share test_manifest.jsonl. Archived metrics may use different filtered clips; compare counts before interpreting differences.",
        "methods": rows}, ensure_ascii=False, indent=2))
    datasets = list(summary["bridge"]["dataset_metrics"])
    lines = ["# Whisper Small bridge comparison", "",
             "Current baseline, bridge, and enhanced-only use the same test_manifest.jsonl. "
             "Archived rows may have different filtered clips; counts are shown for context.", ""]
    for dataset in datasets:
        lines.extend([f"## {dataset}", "", "| Model | Examples | WER | CER |", "| --- | ---: | ---: | ---: |"])
        for row in rows:
            score = row["datasets"].get(dataset)
            if score is not None:
                lines.append(f"| {row['model']} | {score['examples']} | {score['wer']:.4f} | {score['cer']:.4f} |")
        lines.append("")
    (output / "comparison.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="evaluation YAML with test datasets, checkpoints, and output directory")
    parser.add_argument("--output", type=Path, default=None, help="new result directory; overrides eval.output_dir/eval.name")
    parser.add_argument("--device", default=None, help="torch device; overrides eval.device")
    args = parser.parse_args(argv)
    config = load_eval_config(args.config)
    output = args.output or Path(config["eval"]["output_dir"]) / str(config["eval"]["name"])
    run(suite_config(config), output, None, args.device or config["eval"]["device"])
    write_comparison(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
