"""Build isolated official CV25 splits and reuse their existing degraded variants."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
import tarfile

SPLITS = ("train", "dev", "test")
CLEAN = "cv-corpus-25.0"
DEGRADED = "cv-corpus-25.0-degraded-v2"


def official_tables(archive: Path) -> dict[str, bytes]:
    tables = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            path = Path(member.name)
            if path.parent.name == "fa" and path.name in {f"{s}.tsv" for s in SPLITS}:
                if path.stem in tables:
                    raise ValueError(f"duplicate archive split: {path}")
                tables[path.stem] = stream.extractfile(member).read()
                if len(tables) == len(SPLITS):
                    break
    if set(tables) != set(SPLITS):
        raise ValueError("archive must contain Persian train/dev/test TSVs")
    return tables


def audio_path(root: Path, value: str) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [root / "clips" / raw, root / raw, raw]
    for candidate in candidates:
        for suffix in (candidate.suffix, ".flac", ".wav"):
            alternate = candidate.with_suffix(suffix)
            if alternate.is_file():
                return alternate.resolve()
    raise FileNotFoundError(value)


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def prepare(archive: Path, source_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    tables = official_tables(archive)
    rows = {s: list(csv.DictReader(io.StringIO(tables[s].decode("utf-8-sig")), delimiter="\t")) for s in SPLITS}
    membership = {}
    for split, group in rows.items():
        if not group:
            raise ValueError(f"empty official split: {split}")
        for row in group:
            key = Path(row["path"]).stem
            if key in membership:
                raise ValueError(f"duplicate/overlapping official clip: {key}")
            membership[key] = (split, row)
    clean = output / CLEAN
    degraded = output / DEGRADED
    (clean / "clips").mkdir(parents=True)
    (degraded / "clips").mkdir(parents=True)
    (output / "original-tsvs").mkdir()
    missing_audio = {}
    for split, group in rows.items():
        (output / "original-tsvs" / f"{split}.tsv").write_bytes(tables[split])
        for row in group:
            try:
                source = audio_path(source_root / CLEAN, row["path"])
            except FileNotFoundError:
                name = Path(row["path"]).name
                missing_audio[name] = row
                row["path"] = name
                continue
            target = clean / "clips" / source.name
            os.link(source, target)  # No audio duplication; fail explicitly across filesystems.
            row["path"] = target.name
    if missing_audio:
        print(f"Recovering {len(missing_audio)} original clips from archive", flush=True)
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                name = Path(member.name).name
                if member.isfile() and Path(member.name).parent.name == "clips" and name in missing_audio:
                    with (clean / "clips" / name).open("xb") as handle:
                        handle.write(stream.extractfile(member).read())
                    del missing_audio[name]
                    if not missing_audio:
                        break
        if missing_audio:
            raise FileNotFoundError(f"archive missing selected audio: {sorted(missing_audio)}")
    for split, group in rows.items():
        write_tsv(clean / f"{split}.tsv", group)
    selected = {s: [] for s in ("train", "dev")}
    covered = {s: set() for s in selected}
    counts = Counter()
    mapping = source_root / DEGRADED / "degraded_to_clean.jsonl"
    digest = hashlib.sha256()
    seen = set()
    with mapping.open("rb") as handle, (degraded / "degraded_to_clean.jsonl").open("w") as out:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            item = json.loads(line)
            key = Path(item["clean_path"]).stem
            match = membership.get(key)
            if match is None or match[0] == "test":
                counts["excluded_variants"] += 1
                continue
            split, row = match
            source = audio_path(source_root / DEGRADED, item["degraded_path"])
            name = source.name
            if name in seen:
                raise ValueError(f"duplicate degraded filename: {name}")
            seen.add(name)
            os.link(source, degraded / "clips" / name)
            original = dict(item)
            item.update(split=split, clean_path=str((clean / "clips" / row["path"]).resolve()),
                        degraded_path=str((degraded / "clips" / name).resolve()),
                        degraded_tsv_path=name, sentence=row["sentence"],
                        source_tsv=str((clean / f"{split}.tsv").resolve()), source_path=row["path"],
                        original_mapping=original)
            # Preserve generation metadata unchanged; original_mapping records old assignment.
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            selected[split].append({"path": name, "sentence": row["sentence"]})
            covered[split].add(key)
            counts[f"{split}_variants"] += 1
            counts["reassigned_variants"] += original["split"] != split
    for split, group in selected.items():
        if not group:
            raise ValueError(f"no degraded variants for official {split}")
        write_tsv(degraded / f"{split}.tsv", group)
    ag = source_root / "AGFarsdat_test_normalized"
    if ag.is_dir():
        (output / ag.name).symlink_to(ag.resolve(), target_is_directory=True)
    report = {"archive": str(archive.resolve()), "source_root": str(source_root.resolve()),
              "official_tsv_sha256": {s: hashlib.sha256(tables[s]).hexdigest() for s in SPLITS},
              "source_mapping_sha256": digest.hexdigest(), "clean_counts": {s: len(rows[s]) for s in SPLITS},
              "degraded_source_counts": {s: len(covered[s]) for s in covered},
              "missing_degraded_sources": {s: sorted(k for k, (sp, _) in membership.items() if sp == s and k not in covered[s]) for s in covered},
              "counts": dict(counts), "complete": True}
    (output / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--archive", type=Path, default=Path("data/common-voice-scripted-speech-25-0-persia-510d6f1e.tar.gz"), help="original Persian CV25 tar.gz path")
    parser.add_argument("--source-root", type=Path, default=Path("data"), help="existing clean/degraded dataset root")
    parser.add_argument("--output", type=Path, default=Path("data/cv25-official"), help="new isolated dataset root on the same filesystem; must not exist")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.archive, args.source_root, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
