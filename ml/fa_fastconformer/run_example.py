"""Example: load nvidia/stt_fa_fastconformer_hybrid_large (CTC branch) and
transcribe Persian audio — without installing NeMo.

Get the checkpoint once (any of these):
  # via huggingface_hub
  python -c "from huggingface_hub import hf_hub_download; \
             print(hf_hub_download('nvidia/stt_fa_fastconformer_hybrid_large', \
             'stt_fa_fastconformer_hybrid_large.nemo'))"
  # or download the .nemo from the NGC / HF model card manually.

Then:
  python run_example.py /path/to/stt_fa_fastconformer_hybrid_large.nemo audio1.wav audio2.wav
"""

import sys

import torch

from model import FastConformerCTC


def main():
    nemo_path = sys.argv[1]
    audio_files = sys.argv[2:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FastConformerCTC.from_nemo(nemo_path, map_location="cpu").to(device)
    print(f"Loaded. vocab={model.ctc_decoder._num_classes - 1} (+blank), device={device}")

    if not audio_files:
        print("No audio passed; model loaded successfully.")
        return

    transcripts = model.transcribe(audio_files, batch_size=4, device=device)
    for f, t in zip(audio_files, transcripts):
        print(f"{f}\n  -> {t}\n")


if __name__ == "__main__":
    main()
