"""Evaluate frozen choices on one reported, consistently filtered test cohort."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from jiwer import cer, wer

from ml.fusion.bridging import BridgingModule, observation_addition
from ml.fusion.bridging_experiment import Recognizer, filterbank, skipped_record, write_skipped
from ml.utils.audio import load_audio, resample_audio
from ml.utils.progress import ProgressReporter

TEST_DATASETS = ("cv-corpus-25.0", "AGFarsdat_test_normalized")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config["data"]["split"] != "test" or tuple(config["data"]["datasets"]) != TEST_DATASETS:
        raise ValueError("this final-test protocol requires original CV25 and AGFarsdat test splits")
    if int(config["max_new_tokens"]) < 1:
        raise ValueError("max_new_tokens must be positive")
    return config


def test_rows(config: dict[str, Any], skipped: list[dict] | None = None) -> list[dict[str, str]]:
    """Build one cohort while recording and omitting inconsistent clip rows."""
    from ml.asr.train_whisper_small import resolve_audio_path
    skipped = skipped if skipped is not None else []
    rows = []
    for dataset in config["data"]["datasets"]:
        directory = Path(config["data"]["root_dir"]) / dataset
        with (directory / "test.tsv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not {"path", "sentence"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{directory}/test.tsv needs path and sentence columns")
            raw_rows = list(reader)
            path_counts = Counter(str(row.get("path", "")).strip() for row in raw_rows)
            usable_in_dataset = 0
            for line_number, row in enumerate(raw_rows, start=2):
                path = str(row.get("path") or "").strip()
                sentence = str(row.get("sentence") or "").strip()
                identity = {"id": f"{dataset}/{path or f'line-{line_number}'}", "split": "test"}
                if not path or not sentence:
                    skipped.append(skipped_record(identity, "empty_test_field", f"{directory}/test.tsv:{line_number}"))
                    continue
                if path_counts[path] > 1:
                    skipped.append(skipped_record(identity, "duplicate_test_path", f"{directory}/test.tsv:{line_number}"))
                    continue
                audio = resolve_audio_path(directory, path).resolve()
                if not audio.is_file():
                    skipped.append(skipped_record(identity, "missing_test_audio", str(audio)))
                    continue
                try:
                    relative = Path(path)
                    if relative.is_absolute():
                        relative = audio.relative_to(directory.resolve())
                except ValueError as exc:
                    skipped.append(skipped_record(identity, "test_path_outside_dataset", str(exc)))
                    continue
                if relative.parts and relative.parts[0] == "clips":
                    relative = Path(*relative.parts[1:])
                if ".." in relative.parts:
                    skipped.append(skipped_record(identity, "test_path_outside_dataset", path))
                    continue
                rows.append({"id": f"{dataset}/{path}", "dataset": dataset, "audio_path": str(audio),
                             "relative_path": str(relative), "reference": sentence})
                usable_in_dataset += 1
        if not usable_in_dataset:
            raise ValueError(f"no usable test clips remain in: {directory}")
    return rows


def read_wave(path: str | Path) -> torch.Tensor:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    wave, rate = load_audio(path)
    result = torch.from_numpy(resample_audio(wave, rate, 16000))
    if not 560 <= result.numel() <= 480000 or not torch.isfinite(result).all():
        raise ValueError(f"{path}: require finite 35 ms–30 s audio; no per-method skipping/truncation")
    return result


def enhanced_path(config: dict[str, Any], row: dict[str, str]) -> Path:
    return Path(config["bridge_enhanced_root"]) / row["dataset"] / Path(row["relative_path"]).with_suffix(".wav")


def preflight_rows(config: dict[str, Any], rows: list[dict[str, str]], needs_enhanced: bool,
                   skipped: list[dict]) -> list[dict[str, str]]:
    usable = []
    for row in rows:
        try:
            original = read_wave(row["audio_path"])
            if needs_enhanced:
                enhanced = read_wave(enhanced_path(config, row))
                if enhanced.shape != original.shape:
                    raise ValueError("enhanced waveform length differs from original")
        except (OSError, RuntimeError, ValueError) as exc:
            skipped.append(skipped_record(row, "invalid_test_audio", str(exc)))
            continue
        usable.append(row)
    for dataset in config["data"]["datasets"]:
        if not any(row["dataset"] == dataset for row in usable):
            raise ValueError(f"no usable common-cohort clips remain for: {dataset}")
    return usable


def build_decoder(spec: dict[str, Any], config: dict[str, Any], device: str):
    """Return a common waveform decoder; all outputs use identical raw references."""
    kind = spec["kind"]
    if kind == "fusion":
        from ml.fusion.eval_fusion import load_fusion_model
        from ml.asr.whisper_features import waveform_to_log_mel
        from transformers import WhisperTokenizer
        model, _ = load_fusion_model(Path(spec["checkpoint"]),
            base_asr_checkpoint=config["asr_checkpoint"], model_name=config["processor"])
        model.to(device).eval()
        tokenizer = WhisperTokenizer.from_pretrained(config["processor"])

        def decode(wave, _enhanced):
            mel = waveform_to_log_mel(wave, model_name=config["processor"])[None].to(device)
            tokens = model.generate(mel, view_mode=spec.get("view_mode", "fusion"),
                gate_override=spec.get("gate_override"), language="Persian", task="transcribe",
                max_new_tokens=config["max_new_tokens"], do_sample=False, num_beams=1, suppress_tokens=[])
            return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        return decode
    asr = Recognizer(config["asr_checkpoint"], device, config["max_new_tokens"])
    asr.model.generation_config.suppress_tokens = []
    if kind == "asr":
        return lambda wave, _enhanced: asr(wave)
    if kind != "bridge":
        raise ValueError(f"unknown method kind: {kind}")
    checkpoint = torch.load(spec["checkpoint"], weights_only=True, map_location="cpu")
    provenance = checkpoint["provenance"]
    if provenance["asr_checkpoint"] != config["asr_checkpoint"]:
        raise ValueError("bridge and ASR baseline must use the same frozen checkpoint")
    if provenance["enhancer_id"] != config["bridge_enhancer_id"]:
        raise ValueError("test enhanced audio must use the bridge training enhancer")
    bridge = BridgingModule(**checkpoint["model_config"]).to(device).eval()
    bridge.load_state_dict(checkpoint["state_dict"])

    def decode(wave, enhanced):
        if "omega" in spec:
            omega = torch.tensor(float(spec["omega"]))
        else:
            omega = bridge(filterbank(wave)[None].to(device), filterbank(enhanced)[None].to(device))["omega"].cpu().squeeze(0)
        return asr(observation_addition(wave, enhanced, omega))
    return decode


def run(config: dict[str, Any], output: Path, methods: list[str] | None, device: str) -> None:
    selected = methods or list(config["methods"])
    if len(selected) != len(set(selected)) or not set(selected).issubset(config["methods"]):
        raise ValueError("select unique method names from the config")
    skipped: list[dict] = []
    rows = test_rows(config, skipped)
    needs_enhanced = any(config["methods"][m]["kind"] == "bridge" for m in selected)
    # Preflight the common cohort before loading models or writing scores.
    rows = preflight_rows(config, rows, needs_enhanced, skipped)
    output.mkdir(parents=True, exist_ok=False)
    write_skipped(output / "skipped_inputs.jsonl", skipped)
    manifest = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (output / "test_manifest.jsonl").write_text(manifest)
    digest = hashlib.sha256(manifest.encode()).hexdigest()
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    summary = {}
    progress = ProgressReporter("final-evaluation", len(selected) * len(rows), output / "progress.json")
    completed = 0
    for name in selected:
        print(f"evaluating {name}: {len(rows)} original test clips", flush=True)
        spec = config["methods"][name]
        decode = build_decoder(spec, config, device)
        predictions = []
        with (output / f"{name}.predictions.jsonl").open("w") as stream, torch.inference_mode():
            for row in rows:
                wave = read_wave(row["audio_path"])
                enhanced = read_wave(enhanced_path(config, row)) if spec["kind"] == "bridge" else None
                result = {**row, "hypothesis": decode(wave, enhanced)}
                predictions.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                progress.update(completed, f"method={name} clip={row['id']}")
        del decode
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        def score(items):
            refs, hyps = [x["reference"] for x in items], [x["hypothesis"] for x in items]
            return {"examples": len(items), "wer": wer(refs, hyps), "cer": cer(refs, hyps)}
        metrics = {**score(predictions), "skipped_inputs": len(skipped),
                   "test_manifest_sha256": digest, "method": spec,
                   "dataset_metrics": {d: score([x for x in predictions if x["dataset"] == d])
                                       for d in config["data"]["datasets"]}}
        summary[name] = metrics
        (output / f"{name}.metrics.json").write_text(json.dumps(metrics, indent=2))
        (output / "summary.json").write_text(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/speech_enhancement/cv25_tiny/final_tests.yaml"),
                        help="YAML final-test suite; paths relative to working directory")
    parser.add_argument("--output", type=Path, required=True, help="new result directory (existing paths refused)")
    parser.add_argument("--methods", nargs="+", default=None, help="method names; default all configured methods")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cuda:0")
    args = parser.parse_args(argv)
    run(load_config(args.config), args.output, args.methods, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
