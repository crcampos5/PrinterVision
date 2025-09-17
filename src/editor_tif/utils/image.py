"""Image manipulation helpers."""

from __future__ import annotations

import numpy as np


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize arbitrary arrays into the 0-255 uint8 range."""
    arr = np.asarray(array, dtype=np.float32)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - min_val) / (max_val - min_val)
    return (scaled * 255.0).clip(0, 255).astype(np.uint8)
