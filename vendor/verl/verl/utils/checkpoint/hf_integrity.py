# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Preserve legacy fixed RoPE buffers and reject incomplete HF exports."""

import json
from contextlib import contextmanager
from pathlib import Path

from safetensors import safe_open


@contextmanager
def preserve_rotary_buffers(bridge):
    """Supply only legacy, immutable inv_freq buffers omitted by Megatron.

    Moonlight's original shard index includes these buffers. mbridge waits for
    every indexed key before writing a shard, but Megatron computes RoPE itself
    and does not export them. Never substitute missing trainable parameters.
    """
    io = getattr(bridge, "safetensor_io", None)
    index = getattr(io, "index", None) or {}
    buffers = {key: filename for key, filename in index.items() if key.endswith(".rotary_emb.inv_freq")}
    if not buffers:
        yield
        return
    original = bridge.export_weights
    had_override = "export_weights" in vars(bridge)

    def export_with_buffers(*args, **kwargs):
        seen = set()
        for key, tensor in original(*args, **kwargs):
            seen.add(key)
            yield key, tensor
        for key, filename in buffers.items():
            if key not in seen:
                with safe_open(str(Path(io.hf_dir) / filename), framework="pt", device="cpu") as source:
                    yield key, source.get_tensor(key)

    bridge.export_weights = export_with_buffers
    try:
        yield
    finally:
        if had_override:
            bridge.export_weights = original
        else:
            del bridge.export_weights


def validate_hf_checkpoint(directory):
    """Check shard headers/tensor keys before a checkpoint is acknowledged.

    Header inspection avoids loading the full model into memory. safetensors
    also rejects truncated payloads or invalid data offsets when opening files.
    """
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text()).get("weight_map", {})
        if not index:
            raise RuntimeError(f"Empty HF weight index: {index_path}")
        shards = {}
        for key, filename in index.items():
            path = directory / filename
            if path.resolve().parent != directory.resolve():
                raise RuntimeError(f"HF shard must be inside checkpoint directory: {filename}")
            shards.setdefault(filename, set()).add(key)
    else:
        shards = {"model.safetensors": set()}
    for filename, expected in shards.items():
        path = directory / filename
        if not path.is_file():
            raise RuntimeError(f"Incomplete HF checkpoint; missing shard: {path}")
        try:
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                actual = set(shard.keys())
                missing = expected - actual
                if not actual or missing:
                    raise RuntimeError(f"Incomplete HF shard {path}; missing tensors: {sorted(missing)[:8]}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Unreadable HF checkpoint shard: {path}") from exc
