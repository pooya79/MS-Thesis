"""Frozen FRCRN and DNSMOS adapters for reproducible bridge input preparation.

FRCRN uses the authors' ClearVoice model class and published HF checkpoint.
DNSMOS P.835 uses Microsoft's non-personalized model/calibration (SIG and BAK
only); no P.808 network is needed. See the script guide for source links.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import torch

FRCRN_REVISION = "3766e6a64b0d8cb58f08d913d617bf129f11ed53"
DNSMOS_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dnsmos_segments(wave: np.ndarray) -> list[np.ndarray]:
    """Match dnsmos_local.py: double short clips, 9.01 s windows, 1 s hop."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError("DNSMOS requires a nonempty finite mono waveform")
    while len(wave) < 144160:
        wave = np.concatenate((wave, wave))
    count = int(np.floor(len(wave) / 16000) - 9.01) + 1
    return [wave[i * 16000:i * 16000 + 144160] for i in range(count)]


class DNSMOS:
    def __init__(self, model_root: Path):
        import onnxruntime as ort
        model_root.mkdir(parents=True, exist_ok=True)
        path = model_root / "sig_bak_ovr.onnx"
        if not path.is_file():
            url = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
            temporary = path.with_suffix(".download")
            with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            temporary.replace(path)
        if sha256(path) != DNSMOS_SHA256:
            raise ValueError("DNSMOS model differs from the verified P.835 checkpoint")
        self.identity = f"DNSMOS-P835-nonpersonalized:sha256:{sha256(path)}"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

    def __call__(self, wave: np.ndarray) -> dict[str, float]:
        values = []
        for segment in dnsmos_segments(wave):
            prediction = self.session.run(None, {self.session.get_inputs()[0].name: segment[None]})[0]
            sig, bak, _ = np.asarray(prediction).reshape(-1, 3)[0]
            values.append((np.polyval([-0.08397278, 1.22083953, 0.0052439], sig),
                           np.polyval([-0.13166888, 1.60915514, -0.39604546], bak)))
        scores = np.mean(values, axis=0)
        if not np.isfinite(scores).all():
            raise ValueError("DNSMOS returned nonfinite scores")
        # Preserve raw calibrated predictions, while bounding the supervision to
        # the paper's MOS range. Any clipping is visible in the manifest.
        return {"dnsmos_sig_raw": float(scores[0]), "dnsmos_bak_raw": float(scores[1]),
                "dnsmos_sig": float(np.clip(scores[0], 1, 5)), "dnsmos_bak": float(np.clip(scores[1], 1, 5))}


class FRCRN:
    def __init__(self, model_root: Path, device: str):
        from huggingface_hub import hf_hub_download
        from clearvoice.models.frcrn_se.frcrn import FRCRN_SE_16K
        version = importlib.metadata.version("clearvoice")
        if version != "0.1.2":
            raise ValueError("run `uv sync` to install the locked clearvoice==0.1.2")
        model_root.mkdir(parents=True, exist_ok=True)
        record = model_root / "checkpoint.json"
        repo = "alibabasglab/FRCRN_SE_16K"
        if record.is_file():
            metadata = json.loads(record.read_text())
            checkpoint_path = Path(metadata["path"])
        else:
            revision = FRCRN_REVISION
            pointer = hf_hub_download(repo, "last_best_checkpoint", revision=revision)
            filename = Path(pointer).read_text().strip()
            checkpoint_path = Path(hf_hub_download(repo, filename, revision=revision))
            metadata = {"repo": repo, "revision": revision, "path": str(checkpoint_path),
                        "sha256": sha256(checkpoint_path)}
            record.write_text(json.dumps(metadata, indent=2))
        if sha256(checkpoint_path) != metadata["sha256"]:
            raise ValueError("FRCRN checkpoint hash changed")
        args = SimpleNamespace(win_len=640, win_inc=320, fft_len=640, win_type="hanning")
        self.model = FRCRN_SE_16K(args).model
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.identity = f"{repo}@{metadata['revision']}:sha256:{metadata['sha256']}:clearvoice-{version}:direct-v1"

    @torch.inference_mode()
    def __call__(self, wave: np.ndarray) -> np.ndarray:
        # Match the authors' unnormalized array-input padding recipe. All task
        # clips are <=30 s, so their >120 s segmentation branch is unnecessary.
        length = len(wave)
        window, stride = 16000, 12000
        target = length
        if length < window:
            target = window
        elif length < window + stride:
            target = window + stride
        elif (length - window) % stride:
            target += length - ((length - window) // stride) * stride
        padded = np.pad(wave, (0, target - length)).astype(np.float32)
        result = self.model.inference_batch(torch.from_numpy(padded)[None].to(self.device))
        result = result.detach().cpu().numpy().reshape(-1)
        if len(result) < length or not np.isfinite(result).all():
            raise ValueError("FRCRN returned invalid/short audio")
        return result[:length]  # remove only our right padding, never shift audio
