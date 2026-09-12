---
language:
- fa
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: ${base_model}
tags:
- whisper
- persian
- speech-recognition
---

# ${title}

A Persian speech-recognition checkpoint based on `${base_model}`.
This repository contains the inference weights, model configuration, and saved
processor/tokenizer. It does not contain training audio or optimizer state.

## Access

${access_instructions}

Authenticate on the machine where you will run inference using your own Hugging
Face account and a token with read access to this repository:

```bash
hf auth login
```

## Installation and inference

Use Python 3.10 or newer and a PyTorch build suitable for your CPU or CUDA GPU:

```bash
python -m pip install 'transformers>=5.9,<6' torch librosa soundfile huggingface_hub
```

Save the following as `transcribe.py`, put an audio file at `speech.wav`, then run
`python transcribe.py`. The example reads the token saved by `hf auth login`.
For reproducibility, replace `main` with a commit SHA from this repository.

```python
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

repo_id = "${repo_id}"
revision = "main"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(repo_id, revision=revision, token=True)
model = WhisperForConditionalGeneration.from_pretrained(
    repo_id, revision=revision, token=True
).to(device).eval()

# librosa converts to mono and resamples to the required 16 kHz.
audio, _ = librosa.load("speech.wav", sr=16000, mono=True)
if len(audio) == 0:
    raise ValueError("Audio is empty")

# This simple example handles one recording of at most 30 seconds.
if len(audio) > 30 * 16000:
    raise ValueError("Use the long-audio example below for recordings over 30 seconds")
inputs = processor(
    audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
).to(device)
with torch.inference_mode():
    ids = model.generate(**inputs, language="fa", task="transcribe")
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

For recordings longer than 30 seconds, replace the `inputs = ...` block and the
length guard above with Whisper's sequential long-form generation:

```python
inputs = processor(
    audio, sampling_rate=16000, return_tensors="pt",
    return_attention_mask=True, truncation=False, padding="longest"
).to(device)
with torch.inference_mode():
    ids = model.generate(
        **inputs, language="fa", task="transcribe", return_timestamps=True
    )
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

Long recordings require more memory. Run inference on your own machine or GPU
server; storing weights on the Hub does not itself provision an inference API.

## Training provenance

${provenance_note}

${training_details}

These values describe the saved configuration, not an independently audited
record of every training sample. `publication.json` records the model key and export name,
selected inference files, and the saved training-configuration SHA-256.

## Evaluation and limitations

No WER/CER scores have been verified for this exact published export. No claim
is made that this checkpoint outperforms another run or the base model.
Validate it on your intended domain before use. Noise, accents, overlapping
speakers, music, and silence can produce errors or hallucinated text.
Review transcriptions before using them as research annotations.

## License

Consult the [base model](${base_model_url}) for its license. A redistribution
license for this fine-tuned checkpoint has not been declared here; training-data
rights and the intended license should be reviewed by the model owner before
adding license metadata.
