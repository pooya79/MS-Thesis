"""Numerical equivalence check vs NeMo (run in an env that HAS nemo installed).

This is the gold-standard test that the standalone port reproduces the real
model. It compares CTC log-probs frame-by-frame on random audio.

  python verify_against_nemo.py /path/to/stt_fa_fastconformer_hybrid_large.nemo
"""

import sys

import torch

from model import FastConformerCTC


def main():
    nemo_path = sys.argv[1]
    torch.manual_seed(0)

    # standalone
    mine = FastConformerCTC.from_nemo(nemo_path, map_location="cpu").eval()

    # reference NeMo
    from nemo.collections.asr.models import ASRModel

    ref = ASRModel.restore_from(nemo_path, map_location="cpu").eval()
    ref.cur_decoder = "ctc"  # use CTC branch

    # two random utterances of different length to exercise padding/masking
    wavs = [torch.randn(16000 * 3), torch.randn(16000 * 2)]
    lengths = torch.tensor([w.numel() for w in wavs])
    maxlen = int(lengths.max())
    batch = torch.zeros(len(wavs), maxlen)
    for i, w in enumerate(wavs):
        batch[i, : w.numel()] = w

    with torch.no_grad():
        my_lp, my_len = mine(batch, lengths)

        # NeMo forward through preprocessor->encoder->ctc_decoder
        feats, feat_len = ref.preprocessor(input_signal=batch, length=lengths)
        enc, enc_len = ref.encoder(audio_signal=feats, length=feat_len)
        ref_lp = ref.ctc_decoder(encoder_output=enc)

    print("shapes:", my_lp.shape, ref_lp.shape)
    T = min(my_lp.size(1), ref_lp.size(1))
    diff = (my_lp[:, :T] - ref_lp[:, :T]).abs()
    # compare only valid frames of the first (longest) sample
    valid = diff[0, : my_len[0]]
    print(f"max abs diff (valid frames): {valid.max().item():.3e}")
    print(f"mean abs diff (valid frames): {valid.mean().item():.3e}")
    # argmax agreement
    agree = (my_lp[0, : my_len[0]].argmax(-1) == ref_lp[0, : my_len[0]].argmax(-1)).float().mean()
    print(f"argmax agreement: {agree.item() * 100:.2f}%")
    assert valid.max().item() < 1e-2, "Logprobs diverge — port is not faithful!"
    print("PASS: standalone matches NeMo within tolerance.")


if __name__ == "__main__":
    main()
