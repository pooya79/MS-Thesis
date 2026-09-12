# Prepare the official CV25 cohort

From the repository root on the dataset server:

```bash
uv run python -m ml.speech_data.prepare_cv25_official --help
uv run python -m ml.speech_data.prepare_cv25_official
```

`--archive` (Path) defaults to
`data/common-voice-scripted-speech-25-0-persia-510d6f1e.tar.gz`;
`--source-root` (Path) defaults to `data`; `--output` (Path) defaults to
`data/cv25-official` and must not exist. The output and source audio must be on
the same filesystem. A failed partial output must be preserved/renamed before retry.

The script reads the Persian train/dev/test TSVs directly from the archive,
rejects duplicate or overlapping clip IDs, and retains the official transcripts
and speaker metadata. It reuses the existing converted clean audio, matching
by original filename stem and allowing WAV/FLAC replacements. Hard links give
the cohort its own paths without copying audio. Do not modify linked audio in place.
Selected clean clips absent from the converted dataset are recovered as original
MP3 files from the archive. Original TSV bytes and their SHA256 hashes are retained for audit.

All existing degraded mapping rows are searched, regardless of their old split.
Variants of official train/dev sources are reassigned to the source's official
split; all other variants, including official test sources, are excluded.
The output TSV and mapping transcripts come from the official TSV. Original
mapping records (including seeds, codec metadata, old split and transcript) are
preserved in `original_mapping`; nested generation metadata remains unchanged.
Missing selected audio and duplicate degraded filenames fail preparation.
`selection.json` records counts, missing degraded source IDs and the source
mapping hash. Check coverage before training: no new degradation is generated.

The output contains clean train/dev/test and degraded train/dev datasets plus
a link to the existing AGFarsdat normalized test dataset. The Tiny configs point
to this root and new `cv25-tiny-official` model/artifact directories. Continue
with [the complete experiment sequence](cv25-tiny-run-all.md).

## Server cohort audit, 2026-09-12

Prepared on `user01@213.233.184.201` under `~/MS-Thesis/data/cv25-official`:

| Split | Official clean clips | Existing degraded variants | Sources with variants |
| --- | ---: | ---: | ---: |
| train | 30,105 | 57,684 | 28,842 |
| dev | 10,454 | 20,012 | 10,006 |
| test | 10,519 | excluded | excluded |

These are parsed TSV record counts; quoted multiline fields make `wc -l`
overcount dev/test records. Recovered 1,711 missing clean clips from the archive.
Existing degraded coverage is missing for 1,263 train and 448 dev sources;
these remain available as clean examples. No extra degradations were generated.
541,176 unrelated degraded variants were excluded. All selected variants already
had the correct split, so no split reassignment was needed in this archive.
The original expanded datasets and earlier study artifacts remain intact.

Detailed paths, missing source IDs and hashes are in the server's `selection.json`.
The run launcher also snapshots this report and the configs into its log directory.
