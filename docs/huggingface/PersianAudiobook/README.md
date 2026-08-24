---
language:
  - fa
pretty_name: PersianAudiobook
license: other
gated: true
extra_gated_fields:
  Briefly describe your intended use and research purpose: text
task_categories:
  - automatic-speech-recognition
  - text-to-speech
size_categories:
  - 10K<n<100K
configs:
  - config_name: refined-subset
    data_files:
      - split: train
        path: refined-subset/train-*.parquet
---

# PersianAudiobook

PersianAudiobook is a collection of short Persian audiobook speech clips paired with conservatively refined pseudo-transcriptions. The initial release contains 39,454 accepted examples representing 218.564 hours of mono, 16 kHz speech. It is intended for speech-recognition research, audiobook-domain language modeling, speech representation learning, and carefully reviewed text-to-speech research.

Raw data was collected from IranSeda audiobooks.

The transcriptions are machine-generated and machine-refined rather than human-verified ground truth. Users should account for residual recognition, punctuation, normalization, speaker, and content errors.

## Dataset statistics

| Measure | Value |
|---|---:|
| Accepted examples | 39,454 |
| Source recordings | 136 |
| Total duration | 218.564 hours |
| Total FLAC size before Parquet packaging | 13.161 GiB |
| Mean clip duration | 19.943 s |
| Median clip duration | 20.178 s |
| 5th–95th percentile duration | 15.532–24.200 s |
| Minimum–maximum duration | 2.150–25.000 s |
| Total whitespace-delimited transcript tokens | 1,504,450 |
| Unique transcriptions | 39,163 |
| Audio | mono, 16 kHz, PCM-16 FLAC |

The refinement run evaluated 44,820 targets, accepted 39,454, rejected 5,366, and recorded 40 operational failures. Rejected, uncertain, invalid, and operationally failed targets are not published in this config.

## Data creation pipeline

![Long-audio data creation pipeline](./assets/long-audio-data-creation-pipeline.png)

1. **Source collection and validation:** catalogue records and audio downloads were saved in manifests with source identifiers and SHA-256 checksums. Missing, unreadable, or checksum-inconsistent inputs were rejected.
2. **Speech segmentation:** source recordings were decoded for analysis and segmented with Silero VAD, silence boundaries, and an energy-based fallback. Final clips are non-overlapping mono 16 kHz PCM-16 FLAC files, generally targeting 20 seconds with a hard maximum of 25 seconds in this release.
3. **Pseudo-transcription:** every completed clip was independently transcribed by a Persian fine-tuned Whisper Medium checkpoint. Audio boundaries were fixed before transcription; word timestamps were not used to construct clips.
4. **Deterministic normalization:** Persian characters, spacing, and supported text conventions were normalized before refinement. Empty or invalid normalized outputs were rejected.
5. **Conservative LLM refinement:** a Qwen3.6-27B AWQ model received each target transcript with limited preceding and following context. Its prompt permitted high-confidence spelling, orthography, spacing, punctuation, and obvious word-substitution corrections while forbidding paraphrasing, completion across clip boundaries, unsupported additions, and changes to numeric tokens.
6. **Quality filtering:** structured-output validation, uncertainty rejection, numeric-token preservation, normalized edit-distance limits, audio-manifest consistency, and provenance checks determined whether a refined example was accepted. Full refinement audit records remain outside the published training rows.
7. **Parquet publication:** accepted FLAC bytes and metadata are embedded in typed Hugging Face Parquet shards with 100-row groups for efficient loading and streaming.

The model-training box in the figure represents a downstream use of the accepted data and is not required to reproduce the published dataset.

## Data fields

Each row contains:

- `audio`: typed Hugging Face audio value containing the original FLAC bytes and clip path;
- `id`: stable clip identifier;
- `sentence`: accepted refined Persian pseudo-transcription;
- `source_id`: stable identifier shared by clips from one original recording;
- `duration_sec`, `start_sec`, and `end_sec`: clip duration and original-recording boundaries;
- `speech_seconds` and `speech_ratio`: segmentation-stage speech measurements;
- `boundary_type`: selected segmentation-boundary category;
- `clip_checksum`: SHA-256 clip checksum from the segmentation manifest.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "<your-hf-account>/PersianAudiobook",
    "refined-subset",
    split="train",
    token=True,
)
```

For metadata-only inspection without decoding audio:

```python
from datasets import Audio, load_dataset

dataset = load_dataset(
    "<your-hf-account>/PersianAudiobook",
    "refined-subset",
    split="train",
    token=True,
)
dataset = dataset.cast_column("audio", Audio(decode=False))
```

## Appropriate uses

- Persian speech recognition and audiobook-domain adaptation;
- analysis of Persian pseudo-label refinement and text normalization;
- self-supervised or weakly supervised speech representation learning;
- acoustic and linguistic corpus analysis;
- carefully reviewed TTS experiments where the applicable rights, voice considerations, and residual transcript errors are acceptable.

## Limitations and responsible use

- Transcriptions have not been exhaustively verified by human annotators.
- Punctuation and orthographic refinement may not exactly represent every spoken detail.
- The dataset does not currently provide speaker identities, speaker-consent annotations, demographic labels, or a speaker-disjoint evaluation split.
- Audiobooks may contain sensitive, historical, political, religious, violent, or otherwise objectionable material inherited from their source works.
- Do not use the data to impersonate speakers, infer sensitive speaker attributes, or create misleading synthetic speech.
- The initial config is a selected and filtered subset. A later full config may have different coverage and error characteristics.

## Reproducibility

The preparation archive records configuration digests, model identities, inference parameters, source and clip checksums, segmentation measurements, normalization results, refinement decisions, validation metrics, and accepted/rejected manifests. The published Parquet rows retain the fields needed to trace examples back to those internal audit artifacts without exposing server-local paths or raw model prompts.

## License and attribution

This is a restricted-access research dataset. No license to redistribute the underlying audio is granted, and all rights in the source recordings and literary works remain with their respective rightsholders. Hugging Face collects each requester's username, email address, and brief intended use through the access form. The dataset publisher contacts each requester and manually approves access only after verifying the requester's permissions. Approval does not grant ownership or redistribution rights. Do not bypass access controls, share credentials, or redistribute any content.
