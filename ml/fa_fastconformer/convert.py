"""Convert a hybrid ``.nemo`` checkpoint into a plain-PyTorch ``.pt`` bundle.

The ``.nemo`` file is a tar archive (config YAML + ``model_weights.ckpt`` +
SentencePiece tokenizer). This repacks only the CTC branch we use — encoder +
CTC head weights, the relevant config, and the tokenizer bytes — into a single
self-contained ``.pt`` file that loads without NeMo and without unpacking a tar.

Usage:
  python convert.py /path/to/stt_fa_fastconformer_hybrid_large.nemo out.pt

Then in code:
  from model import FastConformerCTC
  model = FastConformerCTC.from_pretrained("out.pt")
  print(model.transcribe(["a.wav"]))
"""

import argparse
import os
import sys

from model import FastConformerCTC, convert_nemo_to_pt


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Repack a FastConformer hybrid .nemo into a CTC-only PyTorch .pt bundle."
    )
    parser.add_argument("nemo_path", help="Path to the source .nemo checkpoint.")
    parser.add_argument("out_path", help="Destination .pt bundle path.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load the written bundle back to confirm it reconstructs the model.",
    )
    args = parser.parse_args(argv)

    out = convert_nemo_to_pt(args.nemo_path, args.out_path)
    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"wrote {out} ({size_mb:.1f} MiB)")

    if args.verify:
        model = FastConformerCTC.from_pretrained(out, map_location="cpu")
        vocab = model.ctc_decoder._num_classes - 1
        print(f"verified: bundle loads, vocab={vocab} (+blank)")


if __name__ == "__main__":
    sys.exit(main())
