# IranSeda Long-Audio to Whisper Training Dataset Pipeline

## Purpose and status

This document specifies the preparation pipeline for turning the long MP3 files
downloaded by the IranSeda scrapers into a pseudo-labelled Persian ASR dataset.
The labels are produced by a fine-tuned Whisper Medium model and may be
orthographically cleaned by an LLM.

This remains the design for the complete preparation pipeline. The downloader
is responsible only for acquiring and verifying raw audio. The reusable VAD
and chunk-export stage is now implemented as
`ml.speech_data.long_audio_asr_pipeline.segment_audio`; see
[`long-audio-asr-pipeline-guide.md`](long-audio-asr-pipeline-guide.md). The
transcription and later dataset-publication stages remain future work.

The fine-tuned Whisper model is not required to produce word timestamps. Final
audio boundaries are chosen before transcription by voice activity detection
(VAD), silence gaps, and an energy-based fallback. Whisper then transcribes each
finished clip independently, so its complete output belongs to that clip.

## Design goals

- Preserve downloaded MP3s as immutable, traceable source material.
- Produce speech clips that fit comfortably inside Whisper's 30-second input
  window.
- Prefer natural silence boundaries instead of fixed-duration cuts.
- Keep the audio and transcript aligned without relying on word timestamps.
- Prevent LLM cleanup from paraphrasing or inventing speech.
- Record enough provenance to reproduce, inspect, reject, or regenerate every
  example.
- Prevent clips from the same original recording from leaking across dataset
  splits.
- Keep pseudo-labelled training data separate from human-labelled evaluation
  data.

## End-to-end flow

```text
IranSeda discovery manifests
        +
downloads.jsonl and source MP3s
        |
        v
validate source and decode mono 16 kHz working audio
        |
        v
VAD, silence detection, and optional music/overlap filtering
        |
        v
construct final non-overlapping clips (target 20 s, preferred 15-25 s)
        |
        v
transcribe every final clip with fine-tuned Whisper Medium
        |
        v
deterministic Persian normalization
        |
        v
strict LLM orthographic cleanup with structured output
        |
        v
quality gates and rejection manifest
        |
        v
lossless clips + segments.jsonl + train/dev/test TSV files
        |
        v
human audit and training-set release
```

Each phase should consume a saved manifest and atomically write its own output.
That makes an interrupted run resumable and allows transcription or cleanup to
be repeated without redownloading or re-segmenting audio.

## 1. Inputs and immutable source data

Run the existing discovery and download commands as described in
[`scraper_guides/iranseda-scrapers.md`](scraper_guides/iranseda-scrapers.md).
The preparation pipeline reads `downloads.jsonl` from an audiobook or radio
discovery root. Each record supplies the stable source ID, source kind, MP3
path, URL, checksum, and acquisition timestamps.

Do not rewrite or replace the downloaded MP3 during preparation. Treat its
recorded SHA-256 as the source version. A derived record must retain at least:

- source ID and source kind;
- source MP3 path and checksum;
- source discovery manifest;
- preparation configuration version;
- segmentation, ASR, normalization, and LLM versions.

Exclude a source before processing if it is missing, its checksum no longer
matches, it cannot be decoded, or its discovery metadata marks it ineligible.
Record the reason rather than silently skipping it.

## 2. Working-audio decoding

Decode each source to mono 16 kHz PCM for analysis. A representative FFmpeg
operation is:

```bash
ffmpeg -i source.mp3 \
  -map 0:a:0 \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  working/source.wav
```

The implementation should invoke FFmpeg directly and check its exit status; it
should not build commands through a shell. Record the FFmpeg version, decoded
sample rate, channel count, frame count, and duration. Reject zero-length,
non-finite, or unreadable output.

The working WAV may be deleted after all derived clips have been verified. It
is a cache, not the source of record.

Do not apply lossy denoising, aggressive loudness normalization, or a codec
round-trip during this phase. The dataset should preserve the acoustic domain
that the model is intended to learn. Any enhancement experiment belongs in a
separate, explicitly versioned pipeline.

## 3. Speech and non-speech analysis

Run a VAD over the working audio and retain its frame-level or interval-level
scores. Silero VAD is a suitable initial implementation, but its exact model
identifier and version must be recorded. FFmpeg `silencedetect` can be useful
for diagnostics, but VAD scores are preferable for reproducible speech
selection.

Initial thresholds to validate on a representative IranSeda sample are:

| Setting | Initial value | Purpose |
| --- | ---: | --- |
| minimum detected speech | 0.25 s | suppress isolated VAD spikes |
| merge speech across silence | 0.5 s | avoid excessive fragmentation |
| useful boundary silence | 0.3 s | prefer phrase boundaries |
| boundary padding | 0.15 s | avoid clipping initial/final phonemes |

These are starting points, not universal constants. Save them in configuration
and tune them independently for radio and audiobooks if their acoustics differ.

Radio requires additional care. Reject or flag intervals dominated by music,
station jingles, advertisements, severe cross-talk, or non-Persian speech.
Metadata classification alone cannot reliably remove these regions. If an
automatic music or overlap detector is introduced, record its model, score,
threshold, and decision in the segment manifest.

Speaker diarization is optional. When available, prefer boundaries at speaker
changes and avoid combining unrelated speakers only to reach a duration target.
The absence of diarization must not block the basic VAD pipeline.

## 4. Segment construction without word timestamps

Segment the source before asking Whisper for text. Use these default duration
rules:

| Constraint | Default |
| --- | ---: |
| target duration | 20 s |
| preferred minimum | 15 s |
| preferred maximum | 25 s |
| hard maximum | 28 s |
| minimum accepted speech clip | 2 s |

The preferred minimum is not a hard requirement. A natural 7-second utterance
is better training data than a 15-second clip padded with silence or joined to
unrelated speech. The hard maximum leaves margin below Whisper's 30-second
window.

For a segment beginning at `start`:

1. Collect candidate silence midpoints between `start + 15 s` and
   `start + 25 s`.
2. Choose the suitable candidate closest to `start + 20 s`.
3. If no candidate exists, consider a speaker-change boundary in the same
   interval.
4. If neither exists, locate the lowest-energy short region between
   approximately `start + 18 s` and `start + 25 s`.
5. If continuous speech still offers no safe point, make a hard cut no later
   than 25 seconds and mark it for stricter review.
6. Add up to 100-200 ms of boundary padding without exceeding the source or
   creating overlapping final clips.
7. Keep a shorter tail only if it contains sufficient speech and intelligible
   content; otherwise reject or merge it at an earlier natural boundary.

Conceptual pseudocode:

```python
TARGET_SECONDS = 20.0
PREFERRED_MIN_SECONDS = 15.0
PREFERRED_MAX_SECONDS = 25.0

while source_audio_remains:
    candidates = silence_boundaries(
        start + PREFERRED_MIN_SECONDS,
        start + PREFERRED_MAX_SECONDS,
    )
    if candidates:
        end = closest(candidates, start + TARGET_SECONDS)
        boundary_type = "silence"
    else:
        end = lowest_energy_boundary(start + 18.0, start + 25.0)
        boundary_type = "energy_fallback"

    emit_non_overlapping_clip(start, end, boundary_type)
    start = end
```

Inference overlap is unnecessary when every final clip is transcribed
independently. Do not create overlapping training clips: duplicated audio can
bias training and can leak nearly identical examples across sampling batches.

Every proposed interval should first be written to a segmentation manifest.
Include its VAD speech ratio, silence statistics, boundary type, duration, and
any content-filter scores. This permits threshold changes without losing the
audit trail.

## 5. Exporting final audio clips

Export final intervals from the decoded lossless working audio, not by copying
arbitrary MP3 frames. Use mono 16 kHz PCM WAV or lossless FLAC. FLAC is
preferred when storage is important.

Names must be stable and derived from the source ID plus a zero-padded segment
index, for example:

```text
radio-123456_000042.flac
book-9876-attachment-54321_000003.flac
```

Re-running the same configuration against the same source checksum must produce
the same IDs, boundaries, and audio. Before reuse, verify the derived file's
duration and checksum against the manifest.

## 6. Whisper pseudo-transcription

Transcribe each final clip independently with the fine-tuned Whisper Medium
checkpoint. Configure Persian explicitly and use the transcription task. The
model does not need word timestamps because the label is for the entire final
clip.

Record enough inference configuration to reproduce the output:

- checkpoint path and checkpoint checksum or immutable revision;
- processor/tokenizer identifier;
- language and task;
- decoding method, beam size, temperature, and fallback policy;
- generation length limit;
- library versions;
- raw decoded text;
- any available sequence confidence, average log probability, no-speech
  probability, language score, or generation diagnostics.

Use deterministic decoding for dataset generation unless a deliberate
multi-hypothesis selection experiment is documented. Never silently replace a
previous transcript generated by a different checkpoint or decoding setup.

If the inference wrapper exposes no confidence values, the pipeline can still
operate. Mark those values as unavailable and rely more heavily on speech
ratio, transcript heuristics, LLM edit checks, and human auditing.

Transcription failures go to a retry/rejection manifest with the exception and
model configuration. They must not create an empty `sentence` entry.

## 7. Deterministic Persian normalization

Apply deterministic normalization before using an LLM where possible. Reuse
the repository's Persian ASR normalization policy so training and evaluation
do not disagree. Typical operations include:

- canonical Persian `ی` and `ک` forms;
- Unicode normalization;
- consistent whitespace and zero-width non-joiner handling;
- removal of formatting characters that do not represent speech;
- the project's chosen punctuation policy.

Do not remove spoken repetitions, incomplete phrases, filled pauses, names, or
numbers merely because they look unusual. Save the raw Whisper output and the
deterministically normalized value separately.

## 8. Constrained LLM cleanup

The LLM is an orthographic cleaner, not a second transcription system. It does
not hear the audio and therefore must not fill omissions, correct facts,
paraphrase, summarize, translate, or make the sentence more fluent.

Use structured output with at least:

```json
{
  "cleaned_text": "...",
  "uncertain": false,
  "change_categories": ["whitespace", "persian_characters"]
}
```

A suitable prompt contract is:

```text
Normalize the supplied Persian ASR transcript only orthographically.
Do not add, remove, reorder, infer, summarize, translate, or paraphrase spoken
content. Preserve repetitions, disfluencies, incomplete phrases, names,
numbers, and the speaker's grammar. If a boundary appears incomplete, leave it
incomplete. Return only the required structured object and set uncertain=true
when a safe normalization cannot be determined.
```

Version the full system prompt, user template, model identifier, model
parameters, and output schema. Store the input and output for every accepted or
rejected example.

Automatically reject or route to manual review when:

- the LLM reports uncertainty;
- it returns invalid structured output;
- normalized edit distance exceeds a configured threshold;
- token or character count changes unusually;
- digits or written numbers are added, removed, or changed;
- named entities appear to change;
- a new phrase or completed sentence appears;
- the result is empty or in the wrong language.

The default behavior on LLM failure should be to retain the deterministic
transcript for review, not to accept an invented repair. Whether such a sample
is allowed into training must be an explicit configuration choice.

## 9. Quality gates

Quality filtering should produce an accepted manifest and a rejected manifest
with machine-readable reasons. Useful gates include:

- valid lossless audio that is mono, 16 kHz, finite, and non-empty;
- duration within the configured hard limits;
- adequate detected-speech ratio;
- non-empty normalized transcript;
- Persian language consistency;
- plausible characters or tokens per second;
- no obvious repeated-phrase Whisper hallucination;
- acceptable ASR confidence when available;
- acceptable LLM edit distance and no protected-token changes;
- no music-dominated or severe-overlap flag;
- no duplicated or near-duplicated source interval.

Do not select a confidence threshold solely by inspecting the pseudo-label
model's own scores. Calibrate thresholds on a manually transcribed sample and
measure label error versus data retained.

Examples created with `energy_fallback` or `hard_cut` boundaries should receive
stricter thresholds or mandatory sampling during manual review, because a word
may have been split at the boundary.

## 10. Leakage-safe dataset splits

Assign each original source to a split before publishing any derived clip. All
clips from that source inherit the same split.

- For radio, group by at least episode ID. Consider grouping an entire program,
  series, or nearby dates when repeated presenters and material would otherwise
  cross splits.
- For audiobooks, group by book. Consider narrator-level grouping when the
  evaluation goal requires unseen speakers.
- Never randomly split the final clips themselves.
- Never add held-out thesis evaluation recordings to pseudo-label training.

Use a fixed seed and a stable hash of the grouping key for deterministic split
assignment. Record the grouping policy and ratios. Inspect hours, speakers or
narrators, sources, and content categories per split rather than checking clip
counts alone.

The primary `test.tsv` should be human-labelled. A pseudo-labelled test set can
be useful for pipeline diagnostics but must not be reported as an unbiased ASR
evaluation set.

## 11. Output layout

The prepared dataset follows the repository ASR convention:

```text
data/iranseda/prepared/
├── train.tsv
├── dev.tsv
├── test.tsv
├── segments.jsonl
├── rejected.jsonl
├── preparation_summary.json
└── clips/
    ├── radio-123456_000000.flac
    ├── radio-123456_000001.flac
    └── book-9876-attachment-54321_000000.flac
```

Every TSV must contain at least `path` and `sentence`:

```tsv
path	sentence
radio-123456_000000.flac	متن نهایی همین قطعه
radio-123456_000001.flac	متن قطعه بعدی
```

Paths are relative to `clips/`, matching the project's dataset resolver.
Additional columns such as `id`, `duration`, or `source_id` are allowed, but
`segments.jsonl` remains the complete provenance record.

A representative accepted segment record is:

```json
{
  "id": "radio-123456_000042",
  "source_id": "123456",
  "source_kind": "radio",
  "source_path": "clips/11/2026-07-01/123456.mp3",
  "source_checksum": "sha256:...",
  "path": "clips/radio-123456_000042.flac",
  "clip_checksum": "sha256:...",
  "start_sec": 120.15,
  "end_sec": 140.72,
  "duration_sec": 20.57,
  "speech_ratio": 0.91,
  "boundary_type": "silence",
  "raw_transcript": "...",
  "normalized_transcript": "...",
  "sentence": "...",
  "asr_checkpoint": "models/asr/whisper-medium/runs/example/checkpoint-5000",
  "llm_model": "...",
  "llm_prompt_version": "v1",
  "llm_uncertain": false,
  "split": "train"
}
```

A rejected record should preserve the same identity and relevant measurements,
plus one or more stable reason codes such as `low_speech_ratio`,
`music_dominated`, `empty_transcript`, `llm_large_edit`, `hard_cut_review`, or
`audio_decode_failed`.

## 12. Human audit

Pseudo-label quality cannot be established by the labelling model or LLM alone.
Before training, manually inspect a stratified sample containing:

- random accepted clips from each source kind and split;
- the lowest-confidence accepted transcripts;
- clips closest to every quality threshold;
- every boundary type, especially energy fallback and hard cuts;
- large but still accepted LLM edits;
- short clips and clips near the maximum duration;
- music, noise, multiple speakers, proper names, and numbers.

Record word or character errors, boundary truncation, music/overlap errors, and
LLM semantic changes. Use the audit to adjust thresholds, then regenerate the
dataset under a new preparation version. Do not edit derived artifacts in
place without recording the correction.

Maintain a small, fully human-corrected development/test set for model
selection. Improvements measured only against pseudo-labels generated by the
teacher model are not reliable evidence of improved ASR.

## 13. Training use

Treat the accepted output as pseudo-labelled data:

- begin with the highest-confidence subset;
- mix it with trusted human-labelled Persian data;
- consider sampling or loss weights that prevent pseudo-labels from dominating;
- compare experiments at fixed amounts of accepted audio;
- evaluate every experiment on the same human-labelled held-out set;
- track the teacher checkpoint and exact dataset preparation version with the
  student checkpoint.

Self-training a model on its own predictions can reinforce omissions,
hallucinations, and normalization habits. The LLM cleanup stage improves text
consistency but cannot establish what the audio actually contained. Expanding
the dataset should therefore be driven by human-set performance, not only by
the quantity of generated hours.

## 14. Reproducibility and resume behavior

The complete future preparation command should accept a configuration file
containing all segmentation, model, cleanup, filtering, and split parameters.
The implemented segmentation stage already applies the corresponding source,
configuration-digest, resume, and atomic-manifest requirements. The complete
pipeline should:

- record the configuration and its digest in every run;
- use explicit seeds for split assignment and any stochastic operation;
- checkpoint after each source or bounded group of clips;
- reuse an artifact only when its source checksum and stage configuration
  digest match;
- write manifests atomically;
- support retrying rejected operational failures separately from quality
  rejections;
- print counts, accepted/rejected duration, and current source progress;
- never overwrite a different preparation version without an explicit flag.

The preparation summary should include source counts and hours, accepted and
rejected clip counts and hours, duration distribution, rejection reasons,
boundary types, LLM edit statistics, and per-split totals.

## 15. Validation checklist

Before using a generated release for training, verify:

- source checksums still match `downloads.jsonl`;
- all accepted clip paths exist and all rejected paths are excluded from TSVs;
- every clip is readable mono 16 kHz lossless audio;
- actual durations match manifest timestamps within the chosen tolerance;
- clips do not exceed the hard maximum;
- transcript fields are non-empty and normalized;
- final clip intervals from one source do not overlap unintentionally;
- no original source grouping key occurs in more than one split;
- held-out evaluation sources are absent from training;
- model, prompt, threshold, FFmpeg, and configuration versions are recorded;
- the manual audit meets the predefined label-quality target.

Automated tests for the future script should cover deterministic segmentation,
silence-boundary selection, continuous-speech fallback, short tails, audio
shape/range safety, stable IDs, manifest fields, LLM schema failures, protected
number changes, source-group split isolation, resumability, CLI help, and TSV
path resolution.
