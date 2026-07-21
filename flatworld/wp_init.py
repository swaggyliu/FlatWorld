"""Warp device initialization helpers."""

from __future__ import annotations

import os

import warp as wp

_INITIALIZED = False


def init_warp(device: str | None = None, prefer_cuda: bool = True) -> str:
    """Initialize Warp once per process and select a device.

    Args:
        device: Explicit device string, e.g. ``"cpu"`` or ``"cuda:0"``.
        prefer_cuda: If no device given, prefer CUDA when available.

    Returns:
        Selected device string.
    """
    global _INITIALIZED
    if not _INITIALIZED:
        wp.init()
        _INITIALIZED = True

    if device is None:
        env = os.environ.get("FLATWORLD_DEVICE", "").strip()
        if env:
            device = env
        elif prefer_cuda and wp.is_cuda_available():
            device = "cuda:0"
        else:
            device = "cpu"

    wp.set_device(device)
    return device


def ensure_warp() -> None:
    """No-op-safe init for library code that may run before tests call init_warp."""
    if not _INITIALIZED:
        init_warp()
