# Whisper Small bridge comparison

Current baseline, bridge, and enhanced-only use the same test_manifest.jsonl. Archived rows may have different filtered clips; counts are shown for context.

## AGFarsdat_test_normalized

| Model | Examples | WER | CER |
| --- | ---: | ---: | ---: |
| current/baseline | 1 | 0.5000 | 0.5556 |
| current/bridge | 1 | 0.0000 | 0.0000 |
| current/enhanced_only | 1 | 0.5000 | 0.5556 |
| archive/fastconformer | 10044 | 0.3107 | 0.1481 |
| archive/fastconformer-deg | 10044 | 0.3048 | 0.1409 |
| archive/whisper | 10044 | 0.3563 | 0.1973 |
| archive/whisper-deg | 10044 | 0.2580 | 0.1262 |
| archive/whisper-fusion-v1 | 10044 | 0.2945 | 0.1379 |
| archive/whisper-fusion-v2 | 10044 | 0.2538 | 0.1238 |
| archive/whisper-iranseda | 10044 | 0.2666 | 0.1296 |
| archive/whisper-medium | 10044 | 0.2199 | 0.1092 |
| archive/whisper-small-noise-only | 10044 | 0.3267 | 0.1749 |
| archive/whisper-small-pmct | 10044 | 0.3001 | 0.1598 |

## cv-corpus-25.0

| Model | Examples | WER | CER |
| --- | ---: | ---: | ---: |
| current/baseline | 1 | 0.0000 | 0.0000 |
| current/bridge | 1 | 0.0000 | 0.0000 |
| current/enhanced_only | 1 | 0.2500 | 0.2353 |
| archive/fastconformer | 10519 | 0.1168 | 0.1646 |
| archive/fastconformer-deg | 10519 | 0.1001 | 0.1610 |
| archive/whisper | 10518 | 0.1152 | 0.0406 |
| archive/whisper-deg | 10518 | 0.0784 | 0.0322 |
| archive/whisper-fusion-v1 | 10519 | 0.1837 | 0.1787 |
| archive/whisper-fusion-v2 | 10519 | 0.0906 | 0.1608 |
| archive/whisper-iranseda | 10518 | 0.0819 | 0.0329 |
| archive/whisper-medium | 10518 | 0.0384 | 0.0165 |
| archive/whisper-small-noise-only | 10518 | 0.0786 | 0.0317 |
| archive/whisper-small-pmct | 10518 | 0.0725 | 0.0279 |

## fleurs-normalized

| Model | Examples | WER | CER |
| --- | ---: | ---: | ---: |
| current/baseline | 1 | 0.2000 | 0.3333 |
| current/bridge | 1 | 0.2000 | 0.3333 |
| current/enhanced_only | 1 | 0.4000 | 0.4583 |
| archive/fastconformer | 871 | 0.2922 | 0.0994 |
| archive/fastconformer-deg | 871 | 0.2940 | 0.1022 |
| archive/whisper | 871 | 0.2105 | 0.0613 |
| archive/whisper-deg | 871 | 0.1982 | 0.0636 |
| archive/whisper-fusion-v1 | 871 | 0.2155 | 0.0668 |
| archive/whisper-fusion-v2 | 871 | 0.1961 | 0.0623 |
| archive/whisper-iranseda | 871 | 0.1378 | 0.0409 |
| archive/whisper-medium | 871 | 0.1714 | 0.0542 |
| archive/whisper-small-noise-only | 871 | 0.1926 | 0.0572 |
| archive/whisper-small-pmct | 871 | 0.2023 | 0.0615 |

## PersianSpeech_test

| Model | Examples | WER | CER |
| --- | ---: | ---: | ---: |
| current/baseline | 1 | 0.0000 | 0.0000 |
| current/bridge | 1 | 0.0000 | 0.0000 |
| current/enhanced_only | 1 | 0.2500 | 0.2632 |
| archive/fastconformer | 24 | 0.3070 | 0.1350 |
| archive/fastconformer-deg | 24 | 0.3141 | 0.1356 |
| archive/whisper | 24 | 0.3453 | 0.1621 |
| archive/whisper-deg | 24 | 0.3525 | 0.1562 |
| archive/whisper-fusion-v1 | 24 | 0.2878 | 0.1133 |
| archive/whisper-fusion-v2 | 24 | 0.3118 | 0.1291 |
| archive/whisper-iranseda | 24 | 0.2686 | 0.0982 |
| archive/whisper-medium | 24 | 0.3165 | 0.1399 |
| archive/whisper-small-noise-only | 24 | 0.3357 | 0.1464 |
| archive/whisper-small-pmct | 24 | 0.3165 | 0.1057 |

## persian-speech-corpus-test

| Model | Examples | WER | CER |
| --- | ---: | ---: | ---: |
| current/baseline | 1 | 0.2000 | 0.1304 |
| current/bridge | 1 | 0.0000 | 0.0000 |
| current/enhanced_only | 1 | 0.4000 | 0.3478 |
| archive/fastconformer | 396 | 0.3369 | 0.1222 |
| archive/fastconformer-deg | 396 | 0.3359 | 0.1268 |
| archive/whisper | 391 | 0.3017 | 0.1548 |
| archive/whisper-deg | 391 | 0.3090 | 0.1522 |
| archive/whisper-fusion-v1 | 396 | 0.3726 | 0.2191 |
| archive/whisper-fusion-v2 | 396 | 0.3517 | 0.2107 |
| archive/whisper-iranseda | 391 | 0.2587 | 0.1416 |
| archive/whisper-medium | 391 | 0.3117 | 0.1729 |
| archive/whisper-small-noise-only | 391 | 0.3047 | 0.1540 |
| archive/whisper-small-pmct | 391 | 0.3050 | 0.1513 |
