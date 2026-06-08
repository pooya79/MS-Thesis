# Standalone FastConformer-CTC (Persian) — no NeMo required

A minimal, dependency-light reimplementation of the **CTC branch** of
`nvidia/stt_fa_fastconformer_hybrid_large`, so you can run **and fine-tune**
the checkpoint without installing NeMo.

The `.nemo` file is a tar archive holding a config YAML, a PyTorch
`state_dict`, and a SentencePiece tokenizer. This package rebuilds the
FastConformer encoder + CTC head as plain `nn.Module`s whose parameter names
match NeMo exactly, then loads those weights with `strict` matching. The RNNT
(transducer) decoder/joint weights in the checkpoint are ignored — CTC alone
is enough to transcribe and is far simpler (no autoregressive loop).

## Files
| file | purpose |
|------|---------|
| `features.py` | mel-spectrogram preprocessor (port of `AudioToMelSpectrogramPreprocessor`) |
| `conformer.py` | FastConformer encoder (subsampling, rel-pos attention, conformer blocks) |
| `model.py` | `.nemo` unpacking, CTC head, greedy decode, `transcribe()` |
| `run_example.py` | load + transcribe wav files |
| `verify_against_nemo.py` | numerical equivalence check vs real NeMo (run where NeMo is installed) |

## Install & run
```bash
pip install -r requirements.txt

# get the checkpoint once
python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('nvidia/stt_fa_fastconformer_hybrid_large', \
  'stt_fa_fastconformer_hybrid_large.nemo'))"

python run_example.py /path/to/stt_fa_fastconformer_hybrid_large.nemo my_audio.wav
```

## Loading in code
```python
from model import FastConformerCTC
model = FastConformerCTC.from_nemo("stt_fa_fastconformer_hybrid_large.nemo")
print(model.transcribe(["a.wav", "b.wav"]))
```

## Convert to a plain-PyTorch bundle (no NeMo, no tar)
The `.nemo` is a tar archive that must be unpacked on every load. Repack the
CTC branch we use (encoder + CTC head weights, the relevant config, and the
tokenizer bytes) into a single self-contained `.pt` once, then load from it:
```bash
python convert.py stt_fa_fastconformer_hybrid_large.nemo stt_fa_fastconformer_ctc.pt --verify
```
```python
from model import FastConformerCTC
model = FastConformerCTC.from_pretrained("stt_fa_fastconformer_ctc.pt")
print(model.transcribe(["a.wav"]))
```
The bundle drops the unused RNNT decoder/joint weights, so it is smaller than
the `.nemo` and loads without unpacking a tar. `convert_nemo_to_pt(nemo, out)`
is also importable from `model.py` for use in scripts.

| file | purpose |
|------|---------|
| `convert.py` | repack a `.nemo` into a CTC-only `.pt` bundle |

## Fine-tuning
`FastConformerCTC` is a standard `nn.Module`. Training loop sketch:
```python
import torch.nn.functional as F

model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# waveforms: (B, T_samp), wav_lens: (B,), targets: list[list[int]] token ids
log_probs, enc_len = model(waveforms, wav_lens)        # (B, T, V+1)
targets_flat = torch.cat([torch.tensor(t) for t in targets])
target_lens = torch.tensor([len(t) for t in targets])
loss = F.ctc_loss(
    log_probs.transpose(0, 1),      # (T, B, V+1)
    targets_flat, enc_len, target_lens,
    blank=model.blank_id, zero_infinity=True,
)
loss.backward(); opt.step(); opt.zero_grad()
```
Tokenize text with `model.tokenizer.EncodeAsIds(text)`. For real training add
a dataloader, batching by length, SpecAugment, and an LR schedule. Freeze the
encoder (`for p in model.encoder.parameters(): p.requires_grad = False`) to
fine-tune only the head on small data.

## Scope / faithfulness
Ported for the exact config of this checkpoint: non-streaming, `rel_pos`
relative attention, `dw_striding` 8× subsampling, batch-norm conv, full
(unlimited) attention context. Not ported: streaming caches, local/longformer
attention, stochastic depth, adapters, the RNNT branch. Inference path only
(no dithering/SpecAugment in the preprocessor).

Run `verify_against_nemo.py` in an environment that has NeMo to confirm the
standalone log-probs match the reference within ~1e-3.
```
python verify_against_nemo.py /path/to/stt_fa_fastconformer_hybrid_large.nemo
```
