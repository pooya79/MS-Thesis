"""Standalone FastConformer-CTC model for nvidia/stt_fa_fastconformer_hybrid_large.

Uses ONLY the CTC branch of the hybrid checkpoint (encoder + ctc_decoder),
which is enough for transcription and is far simpler than the RNNT branch
(no autoregressive joint loop). The RNNT decoder/joint weights in the .nemo
are ignored.

No NeMo dependency. Requirements: torch, torchaudio, librosa, sentencepiece,
pyyaml, soundfile (for audio I/O).
"""

import os
import tarfile
import tempfile

import torch
import torch.nn as nn
import yaml

from conformer import ConformerEncoder
from features import MelSpectrogramPreprocessor


# --------------------------------------------------------------------------- #
# CTC head (nemo ConvASRDecoder)
# --------------------------------------------------------------------------- #
class ConvASRDecoder(nn.Module):
    """1x1 conv projection to vocab+1 (blank). Matches nemo ConvASRDecoder."""

    def __init__(self, feat_in, num_classes):
        super().__init__()
        self._num_classes = num_classes + 1  # +1 blank
        self.decoder_layers = nn.Sequential(nn.Conv1d(feat_in, self._num_classes, kernel_size=1, bias=True))

    def forward(self, encoder_output):  # encoder_output: (B, D, T)
        return torch.log_softmax(self.decoder_layers(encoder_output).transpose(1, 2), dim=-1)


# --------------------------------------------------------------------------- #
# .nemo unpacking
# --------------------------------------------------------------------------- #
def unpack_nemo(nemo_path, extract_dir):
    """Extract a .nemo tar archive (gz or plain) into extract_dir."""
    with tarfile.open(nemo_path, "r:*") as tar:
        tar.extractall(path=extract_dir)
    return extract_dir


def _find_file(root, predicate):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if predicate(fn):
                return os.path.join(dirpath, fn)
    return None


def _read_nemo_parts(extract_dir):
    """Locate (config dict, weights path, tokenizer path) in an extracted .nemo."""
    cfg_path = _find_file(extract_dir, lambda f: f == "model_config.yaml") or _find_file(
        extract_dir, lambda f: f.endswith(".yaml")
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    weights_path = _find_file(extract_dir, lambda f: f.endswith(".ckpt")) or _find_file(
        extract_dir, lambda f: f == "model_weights.ckpt"
    )
    tok_path = _find_file(extract_dir, lambda f: f.endswith(".model"))  # sentencepiece
    if tok_path is None:
        raise RuntimeError("No SentencePiece .model file found inside the .nemo archive")
    return cfg, weights_path, tok_path


# Bundle format tag stored in / checked on the .pt files produced below.
BUNDLE_FORMAT = "fastconformer-ctc-v1"


def convert_nemo_to_pt(nemo_path, out_path, map_location="cpu"):
    """Repack a hybrid ``.nemo`` checkpoint into a single CTC-only ``.pt`` bundle.

    The bundle is a torch-saved dict with the trimmed config (only the sections
    the standalone model reads), the encoder + CTC-head ``state_dict``, and the
    serialized SentencePiece tokenizer. Loading it back via
    :meth:`FastConformerCTC.from_pretrained` needs neither a tar unpack nor NeMo.
    The RNNT decoder/joint weights are dropped, so the file is much smaller.
    """
    tmp = tempfile.mkdtemp(prefix="nemo_fa_")
    unpack_nemo(nemo_path, tmp)
    cfg, weights_path, tok_path = _read_nemo_parts(tmp)

    state = torch.load(weights_path, map_location=map_location, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    wanted = {
        k: v for k, v in state.items() if k.startswith("encoder.") or k.startswith("ctc_decoder.")
    }
    if not wanted:
        raise RuntimeError(f"No encoder./ctc_decoder. weights found in {weights_path}")

    # keep only the config sections _build_modules reads
    cfg_subset = {
        "preprocessor": cfg["preprocessor"],
        "encoder": cfg["encoder"],
        "aux_ctc": {"decoder": cfg["aux_ctc"]["decoder"]},
    }
    with open(tok_path, "rb") as f:
        tokenizer_proto = f.read()

    bundle = {
        "format": BUNDLE_FORMAT,
        "config": cfg_subset,
        "state_dict": wanted,
        "tokenizer_proto": tokenizer_proto,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(bundle, out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FastConformerCTC(nn.Module):
    def __init__(self, preprocessor, encoder, ctc_decoder, tokenizer):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.ctc_decoder = ctc_decoder
        self.tokenizer = tokenizer  # sentencepiece processor
        self.blank_id = ctc_decoder._num_classes - 1

    # -- module assembly --------------------------------------------------- #
    @staticmethod
    def _build_modules(cfg):
        """Build (preprocessor, encoder, ctc_decoder) from a config mapping.

        Only the ``preprocessor``, ``encoder`` and ``aux_ctc.decoder`` sections
        are read, so a trimmed config (as stored in a .pt bundle) is enough.
        """
        pcfg = cfg["preprocessor"]
        preprocessor = MelSpectrogramPreprocessor(
            sample_rate=pcfg.get("sample_rate", 16000),
            window_size=pcfg.get("window_size", 0.025),
            window_stride=pcfg.get("window_stride", 0.01),
            window=pcfg.get("window", "hann"),
            features=pcfg.get("features", 80),
            n_fft=pcfg.get("n_fft", None),
            preemph=pcfg.get("preemph", 0.97),
            log=pcfg.get("log", True),
            log_zero_guard_type=pcfg.get("log_zero_guard_type", "add"),
            log_zero_guard_value=pcfg.get("log_zero_guard_value", 2**-24),
            mag_power=pcfg.get("mag_power", 2.0),
            normalize=pcfg.get("normalize", "per_feature"),
            pad_to=pcfg.get("pad_to", 16),
            pad_value=pcfg.get("pad_value", 0.0),
            mel_norm=pcfg.get("mel_norm", "slaney"),
        )

        ecfg = cfg["encoder"]
        encoder = ConformerEncoder(
            feat_in=ecfg["feat_in"],
            n_layers=ecfg["n_layers"],
            d_model=ecfg["d_model"],
            subsampling_factor=ecfg.get("subsampling_factor", 8),
            subsampling_conv_channels=ecfg.get("subsampling_conv_channels", -1),
            ff_expansion_factor=ecfg.get("ff_expansion_factor", 4),
            n_heads=ecfg.get("n_heads", 8),
            conv_kernel_size=ecfg.get("conv_kernel_size", 9),
            conv_norm_type=ecfg.get("conv_norm_type", "batch_norm"),
            xscaling=ecfg.get("xscaling", True),
            pos_emb_max_len=ecfg.get("pos_emb_max_len", 5000),
            att_context_size=tuple(ecfg.get("att_context_size", [-1, -1])),
        )

        # CTC head dims come from aux_ctc.decoder
        dcfg = cfg["aux_ctc"]["decoder"]
        vocab = dcfg.get("vocabulary", None)
        num_classes = dcfg.get("num_classes", -1)
        if num_classes is None or num_classes <= 0:
            num_classes = len(vocab)
        ctc_decoder = ConvASRDecoder(feat_in=dcfg["feat_in"], num_classes=num_classes)
        return preprocessor, encoder, ctc_decoder

    @classmethod
    def _from_parts(cls, cfg, state_dict, tokenizer):
        """Assemble a model from a config, a state_dict and a tokenizer."""
        preprocessor, encoder, ctc_decoder = cls._build_modules(cfg)
        model = cls(preprocessor, encoder, ctc_decoder, tokenizer)

        # keep only encoder + ctc_decoder weights; ignore rnnt decoder/joint
        wanted = {k: v for k, v in state_dict.items() if k.startswith("encoder.") or k.startswith("ctc_decoder.")}
        missing, _unexpected = model.load_state_dict(wanted, strict=False)
        # preprocessor buffers (window/fb) are recomputed, not loaded -> expected missing
        real_missing = [m for m in missing if not m.startswith("preprocessor.")]
        if real_missing:
            raise RuntimeError(f"Missing weights (architecture mismatch): {real_missing[:10]} ...")
        model.eval()
        return model

    # -- construction from an extracted .nemo dir -------------------------- #
    @classmethod
    def from_extracted(cls, cfg, weights_path, tokenizer_model_path, map_location="cpu"):
        import sentencepiece as spm

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(tokenizer_model_path)

        state = torch.load(weights_path, map_location=map_location, weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        return cls._from_parts(cfg, state, tokenizer)

    @classmethod
    def from_nemo(cls, nemo_path, map_location="cpu"):
        tmp = tempfile.mkdtemp(prefix="nemo_fa_")
        unpack_nemo(nemo_path, tmp)
        cfg, weights_path, tok_path = _read_nemo_parts(tmp)
        return cls.from_extracted(cfg, weights_path, tok_path, map_location=map_location)

    # -- plain-PyTorch bundle (no NeMo, no tar) ---------------------------- #
    @classmethod
    def from_pretrained(cls, bundle_path, map_location="cpu"):
        """Load from a ``.pt`` bundle produced by :func:`convert_nemo_to_pt`.

        The bundle is a single torch-saved dict holding the trimmed config, the
        encoder + CTC-head ``state_dict``, and the serialized SentencePiece
        tokenizer — so loading needs neither a tar unpack nor NeMo.
        """
        import sentencepiece as spm

        bundle = torch.load(bundle_path, map_location=map_location, weights_only=False)
        if bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(
                f"{bundle_path} is not a FastConformer-CTC bundle "
                f"(expected format={BUNDLE_FORMAT!r}, got {bundle.get('format')!r})"
            )
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.LoadFromSerializedProto(bundle["tokenizer_proto"])
        return cls._from_parts(bundle["config"], bundle["state_dict"], tokenizer)

    # -- inference --------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, waveforms, lengths):
        feats, feat_len = self.preprocessor(waveforms, lengths)
        enc, enc_len = self.encoder(feats, feat_len)
        log_probs = self.ctc_decoder(enc)  # (B, T, V+1)
        return log_probs, enc_len

    def _greedy_decode_ids(self, log_probs, enc_len):
        preds = log_probs.argmax(dim=-1)  # (B, T)
        results = []
        for b in range(preds.size(0)):
            seq = preds[b, : enc_len[b]].tolist()
            collapsed = []
            prev = None
            for p in seq:
                if p != prev and p != self.blank_id:
                    collapsed.append(p)
                prev = p
            results.append(collapsed)
        return results

    def _plan_batches(self, durations, batch_size, max_batch_seconds):
        """Group example indices into batches.

        With ``max_batch_seconds`` set, indices are sorted by duration (longest
        first, so an over-budget run fails fast) and each batch is capped both by
        ``batch_size`` and by ``n * max_len_seconds**2 <= max_batch_seconds**2``.
        That quadratic budget bounds peak memory — which scales with
        batch * length**2 in the relative-position attention — to roughly that of
        one ``max_batch_seconds``-long clip, so long clips fall into small (even
        size-1) batches while short clips still pack up to ``batch_size``.
        A single clip longer than ``max_batch_seconds`` still gets its own batch.

        With ``max_batch_seconds`` None, falls back to fixed sequential batches in
        input order.
        """
        n = len(durations)
        if max_batch_seconds is None:
            return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]

        order = sorted(range(n), key=lambda i: durations[i], reverse=True)
        budget = float(max_batch_seconds) ** 2
        batches = []
        current, current_max = [], 0.0
        for i in order:
            new_max = max(current_max, durations[i])
            over_size = len(current) + 1 > batch_size
            over_budget = (len(current) + 1) * new_max**2 > budget
            if current and (over_size or over_budget):
                batches.append(current)
                current, current_max = [], 0.0
                new_max = durations[i]
            current.append(i)
            current_max = new_max
        if current:
            batches.append(current)
        return batches

    @torch.no_grad()
    def transcribe(
        self,
        audio_paths,
        batch_size=4,
        device=None,
        target_sr=16000,
        progress=False,
        max_batch_seconds=None,
    ):
        """audio_paths: list of wav file paths. Returns list of transcript strings.

        Set ``progress=True`` for a tqdm progress bar over the batches (falls
        back to no bar if tqdm is not installed).

        Set ``max_batch_seconds`` to enable duration-aware dynamic batching: clips
        are sorted by length and batched under a memory budget so a few long clips
        cannot blow up GPU memory (see :meth:`_plan_batches`). Output order always
        matches the input order regardless of internal batching.
        """
        import soundfile as sf

        device = device or next(self.parameters()).device
        self.to(device)

        if max_batch_seconds is not None:
            durations = [sf.info(p).duration for p in audio_paths]
        else:
            durations = [0.0] * len(audio_paths)
        batches = self._plan_batches(durations, batch_size, max_batch_seconds)

        if progress:
            try:
                from tqdm.auto import tqdm

                batches = tqdm(batches, unit="batch", desc="transcribe")
            except ImportError:
                pass

        texts = [None] * len(audio_paths)
        for batch_indices in batches:
            waves = []
            for idx in batch_indices:
                wav, sr = sf.read(audio_paths[idx], dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != target_sr:
                    import torchaudio

                    wav = torchaudio.functional.resample(
                        torch.from_numpy(wav), sr, target_sr
                    ).numpy()
                waves.append(torch.from_numpy(wav))
            lengths = torch.tensor([w.numel() for w in waves], dtype=torch.long)
            maxlen = int(lengths.max())
            padded = torch.zeros(len(waves), maxlen, dtype=torch.float32)
            for j, w in enumerate(waves):
                padded[j, : w.numel()] = w
            padded, lengths = padded.to(device), lengths.to(device)

            log_probs, enc_len = self.forward(padded, lengths)
            id_seqs = self._greedy_decode_ids(log_probs, enc_len.cpu())
            for idx, ids in zip(batch_indices, id_seqs):
                texts[idx] = self.tokenizer.DecodeIds(ids)
        return texts
