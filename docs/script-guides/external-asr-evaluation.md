# External ASR Evaluation

## Evaluate OpenRouter Speech-to-Text Models

Evaluate multiple OpenRouter STT models against the mixed test dataset while
retaining every exact prediction and tracking actual API cost:

```bash
export OPENROUTER_API_KEY='YOUR_DEDICATED_EVALUATION_KEY'

uv run python -m ml.asr.eval_openrouter_stt \
  --dataset-root data/mixed-persian-test \
  --model openai/whisper-large-v3 \
  --model openai/gpt-4o-mini-transcribe \
  --max-run-cost-usd 5.00 \
  --min-key-remaining-usd 1.00 \
  --output-dir artifacts/openrouter-stt/mixed-persian-test
```

Use OpenRouter model slugs that advertise the `transcription` output modality.
The API key is read only from `OPENROUTER_API_KEY`; do not put it in a config,
command argument, or committed file. For the strongest cost protection, create
a dedicated OpenRouter key with its own server-side spending limit. The required
`--max-run-cost-usd` is also checked locally before every sequential request,
and `--min-key-remaining-usd` stops when the key's reported `limit_remaining`
reaches that reserve. Because OpenRouter reports exact cost only after a request,
the local run cap alone can be exceeded by one in-flight request; the dedicated
key limit is the hard remote guard.

The output is checkpointed after each successful request. `predictions.jsonl`
is the append-only source of truth and includes the exact model response,
reference, normalized scoring strings, per-clip WER/CER, source dataset, full
usage object, and request cost. `predictions.tsv` is a convenient tabular view;
`metrics.json` reports corpus-level WER/CER and cost for every model, overall and
for every `source_dataset`. `events.jsonl` records each budget check and request
transition, while `logs/openrouter_stt.log` is the human-readable live log.
Interrupted or budget-stopped evaluations can continue with `--resume` and the
same dataset, ordered model list, language, and output directory. A completed
run exits 0, API failures/incomplete results exit 1, and a safe budget stop exits
2. WER/CER use the repository's Persian ASR normalization, but raw references
and predictions are always preserved for auditability.

```bash
uv run python -m ml.asr.eval_openrouter_stt \
  --dataset-root data/mixed-persian-test \
  --model openai/whisper-large-v3 \
  --model openai/gpt-4o-mini-transcribe \
  --max-run-cost-usd 5.00 \
  --min-key-remaining-usd 1.00 \
  --output-dir artifacts/openrouter-stt/mixed-persian-test \
  --resume
```

OpenRouter's current-key endpoint is queried before each transcription. The log
therefore shows the key's cumulative usage and remaining key limit alongside
this evaluator's independently checkpointed run cost.

### Re-score OpenRouter output with strict normalization

Recompute WER/CER from an existing OpenRouter evaluation without making API
requests:

```bash
uv run python -m ml.asr.rescore_openrouter_stt \
  --output-dir artifacts/openrouter-stt/mixed-persian-test
```

The command always reads the raw `reference` and `prediction` values from
`predictions.jsonl`; it does not reuse the evaluator's existing normalized
fields. Both sides receive NFKC and Persian character normalization, followed
by removal of every Unicode punctuation (`P*`) and format (`Cf`) character.
The latter covers zero-width non-joiner, zero-width joiner, zero-width space,
word joiner, and legacy zero-width no-break-space representations of
half-spaces. Whitespace is then collapsed.

The original evaluation artifacts remain unchanged. In
`predictions_strict_normalized.jsonl`, both the `reference`/`prediction` fields
and their `*_normalized` aliases contain only the strictly normalized text;
the raw values remain available in the original `predictions.jsonl`.
Per-example WER/CER is included alongside the text, while corpus WER/CER grouped
by model and `source_dataset` is written to `metrics_strict_normalized.json`.

## Evaluate ElevenLabs Scribe v2

Evaluate ElevenLabs Scribe v2 on the same mixed Persian test dataset:

```bash
export ELEVENLABS_API_KEY='YOUR_DEDICATED_EVALUATION_KEY'

uv run python -m ml.asr.eval_elevenlabs_scribe \
  --dataset-root data/mixed-persian-test \
  --max-estimated-cost-usd 5.00 \
  --output-dir artifacts/elevenlabs-scribe/mixed-persian-test
```

The evaluator uses the synchronous `POST /v1/speech-to-text` endpoint with
`scribe_v2`, Persian (`fa`), temperature 0, seed 0, and audio-event tagging and
diarization disabled. This keeps the output focused on ASR text and makes the
run as reproducible as the service permits. The key is read only from
`ELEVENLABS_API_KEY`.

The evaluator does not query ElevenLabs' user or subscription endpoints, so the
API key only needs speech-to-text access. For the strongest protection, create
a dedicated ElevenLabs API key with its own credit quota in the ElevenLabs
dashboard; that quota is enforced remotely.

The STT response does not report exact request cost in USD. The evaluator
therefore computes and clearly labels an estimate from each clip's duration.
`--price-per-hour-usd` defaults to the current public Scribe v2 API price of
`0.22`, and should be overridden if your contract or current price differs.
Before sending a clip, `--max-estimated-cost-usd` checks its projected cost, so
this estimate-based cap is not crossed by an in-flight request.

`predictions.jsonl` preserves the exact raw prediction, raw provider response,
response request IDs, reference, normalized scoring strings, source dataset,
per-clip WER/CER, duration, and estimated cost.
`predictions.tsv` provides the main fields in tabular form. `metrics.json`
reports corpus-level WER/CER, duration, and estimated cost overall and for every
`source_dataset`; `events.jsonl` and
`logs/elevenlabs_scribe.log` provide machine-readable and live operational
logs. Exit codes are 0 for complete, 1 for incomplete/API failures, and 2 for a
safe budget stop.

Resume an interrupted or budget-stopped run after raising the estimated-cost cap:

```bash
uv run python -m ml.asr.eval_elevenlabs_scribe \
  --dataset-root data/mixed-persian-test \
  --max-estimated-cost-usd 10.00 \
  --output-dir artifacts/elevenlabs-scribe/mixed-persian-test \
  --resume
```

The dataset, manifest, model, language, seed, and price assumption must match
the original run. Completed clips are skipped safely. The estimated-cost cap
may be changed when resuming.

## Evaluate Ivira Avanegar

Evaluate Avanegar on the same mixed Persian test dataset using its
[documented synchronous short-audio API](https://api.ivira.ai/partai/avanegar?type=document):

```bash
export IVIRA_GATEWAY_TOKEN='YOUR_DEDICATED_EVALUATION_TOKEN'

uv run python -m ml.asr.eval_ivira_avanegar \
  --dataset-root data/mixed-persian-test \
  --model default \
  --max-run-units 10000 \
  --output-dir artifacts/ivira-avanegar/mixed-persian-test
```

The token is read only from `IVIRA_GATEWAY_TOKEN`. The evaluator sends clips
sequentially to `POST /avanegar/avanegar/request`, and rejects clips of 60
seconds or longer before upload because the documented synchronous endpoint is
for audio below one minute. The request always sends both `punctuation=false`
and `spokenPunctuation=false`; these settings are fixed and cannot accidentally
be enabled from the CLI. SRT, timestamps, inverse normalization, diarization,
and speaker separation are also disabled.

Avanegar reports `units` in each successful API response. The exact response
and per-request units are checkpointed in `predictions.jsonl`, and cumulative
units are written to `metrics.json`, `events.jsonl`, and
`logs/ivira_avanegar.log`. `--max-run-units` is checked before every request.
Because the next request's units are unknown until it succeeds, one in-flight
request can cross the local cap. The public API document does not expose an
account-balance endpoint or a conversion from units to currency, so the script
does not invent a remaining-credit or USD figure. Use a dedicated gateway token
with a provider-side quota, if your Ivira account supports one, for a remote
hard cap.

`predictions.jsonl` preserves the exact raw prediction, complete raw provider
response, response IDs, reference, normalized scoring strings, source dataset,
per-clip WER/CER, duration, request options, and billed units.
`predictions.tsv` is the tabular view. `metrics.json` reports total and
per-`source_dataset` WER/CER, duration, and units. Exit codes are 0 for complete,
1 for incomplete/API failures, and 2 for a safe unit-cap stop.

Resume after interruption or after raising the local unit cap:

```bash
uv run python -m ml.asr.eval_ivira_avanegar \
  --dataset-root data/mixed-persian-test \
  --model default \
  --max-run-units 20000 \
  --output-dir artifacts/ivira-avanegar/mixed-persian-test \
  --resume
```

The dataset, manifest, model, duration limit, and fixed processing options must
match the original run. Completed clips are skipped; `--max-run-units` may be
raised when resuming.
