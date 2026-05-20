from __future__ import annotations

import hashlib


def stable_seed(global_seed: int, split: str, clip_id: str, variant_index: int) -> int:
    """Return a deterministic 32-bit seed for one generated pair."""
    key = f"{global_seed}|{split}|{clip_id}|{variant_index}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**32)
