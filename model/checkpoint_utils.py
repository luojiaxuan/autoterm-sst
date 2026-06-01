"""Checkpoint loading helpers for trusted local demo artifacts."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch


def load_local_checkpoint(path: str, map_location: Any = "cpu") -> Any:
    """Load a trusted local checkpoint, including legacy metadata objects."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@contextlib.contextmanager
def legacy_torch_load_defaults() -> Iterator[None]:
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_fairseq_checkpoint_to_cpu(path: str, *args, **kwargs) -> Any:
    """Call Fairseq's loader with PyTorch 2.6+ legacy checkpoint semantics."""
    import fairseq

    with legacy_torch_load_defaults():
        return fairseq.checkpoint_utils.load_checkpoint_to_cpu(path, *args, **kwargs)
