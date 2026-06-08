"""Mel-spectrogram feature extraction, faithful port of NeMo's
AudioToMelSpectrogramPreprocessor / FilterbankFeatures (inference path only).

Reference: nemo/collections/asr/parts/preprocessing/features.py
"""

import math

import librosa
import torch
import torch.nn as nn

CONSTANT = 1e-5


def normalize_batch(x, seq_len, normalize_type):
    """Per-feature mean/std normalization over the valid (unpadded) time steps.

    Mirrors nemo ...preprocessing/features.py::normalize_batch for 'per_feature'.
    """
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            if seq_len[i] > 0:
                x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
                x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    else:
        raise NotImplementedError(f"normalize_type={normalize_type} not supported in this standalone port")


class MelSpectrogramPreprocessor(nn.Module):
    """Standalone equivalent of nemo's AudioToMelSpectrogramPreprocessor.

    Only the (eval/inference) forward path is implemented: no dithering, no
    spec-augment, no narrow-band augmentation. All numeric behaviour
    (preemphasis, centered STFT, slaney mel bank, log guard, per-feature norm,
    pad-to multiple) matches NeMo so that pretrained weights produce identical
    encoder inputs.
    """

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        features=80,
        n_fft=None,
        preemph=0.97,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        mag_power=2.0,
        normalize="per_feature",
        pad_to=16,
        pad_value=0.0,
        mel_norm="slaney",
        dither=1e-5,  # ignored at inference
    ):
        super().__init__()
        self.win_length = int(round(window_size * sample_rate))
        self.hop_length = int(round(window_stride * sample_rate))
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.preemph = preemph
        self.mag_power = mag_power
        self.log = log
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self.normalize = normalize
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.nfilt = features

        highfreq = highfreq or sample_rate / 2

        window_fn = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            None: None,
        }[window]
        win = window_fn(self.win_length, periodic=False) if window_fn is not None else None
        self.register_buffer("window", win, persistent=False)

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=features, fmin=lowfreq, fmax=highfreq, norm=mel_norm
            ),
            dtype=torch.float,
        )
        self.register_buffer("fb", filterbanks, persistent=False)

    def get_seq_len(self, seq_len):
        # Matches NeMo exactly (note: no +1 — NeMo treats the final STFT frame
        # as padding). pad_amount cancels n_fft for even n_fft (center=True).
        pad_amount = self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype=torch.long)

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(f"Invalid log_zero_guard_value: {self.log_zero_guard_value}")
        return self.log_zero_guard_value

    @torch.no_grad()
    def forward(self, x, seq_len):
        """x: (B, T_samples) float waveform; seq_len: (B,) valid sample counts.

        Returns (features (B, nfilt, T_frames), out_lengths (B,)).
        """
        out_len = self.get_seq_len(seq_len)

        # preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # centered STFT (matches torch.stft center=True, pad_mode='constant')
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window.to(dtype=torch.float, device=x.device),
            pad_mode="constant",
            return_complex=True,
        )

        # magnitude -> power
        x = torch.sqrt(x.real.pow(2) + x.imag.pow(2))
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # mel projection
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError(self.log_zero_guard_type)

        # per-feature normalization over valid frames
        if self.normalize:
            x, _, _ = normalize_batch(x, out_len, self.normalize)

        # zero out padded frames
        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) >= out_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1), self.pad_value)

        # pad time to a multiple of pad_to
        if self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, self.pad_to - pad_amt), value=self.pad_value)

        return x, out_len
