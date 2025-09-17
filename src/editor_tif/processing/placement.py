"""Placement helpers for positioning the TIF mosaic."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

Centroid = Tuple[float, float]


def place_tile_on_centroids(canvas: np.ndarray, tile: np.ndarray, centroids: Sequence[Centroid]) -> np.ndarray:
    """Overlay the tile image centered on each centroid over a canvas."""
    result = canvas.copy()
    tile_h, tile_w = tile.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    for x, y in centroids:
        cx = int(round(x - tile_w / 2))
        cy = int(round(y - tile_h / 2))
        x0 = max(cx, 0)
        y0 = max(cy, 0)
        x1 = min(cx + tile_w, canvas_w)
        y1 = min(cy + tile_h, canvas_h)
        if x0 >= x1 or y0 >= y1:
            continue
        tile_slice = tile[y0 - cy : y1 - cy, x0 - cx : x1 - cx]
        region = result[y0:y1, x0:x1]
        if result.ndim == 2:
            mask = tile_slice > 0
            region[mask] = tile_slice[mask]
        else:
            mask = tile_slice > 0
            region[mask] = tile_slice[mask]
        result[y0:y1, x0:x1] = region
    return result
