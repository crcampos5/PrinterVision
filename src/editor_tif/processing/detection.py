"""Object detection utilities for reference images."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Centroid = Tuple[float, float]


def detect_centroids(image: np.ndarray, min_area: int = 50) -> Tuple[np.ndarray, List[Centroid]]:
    """Threshold the image and return centroids of detected components."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = np.count_nonzero(thresh)
    black = thresh.size - white
    if white > black:
        thresh = cv2.bitwise_not(thresh)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    detected: List[Centroid] = []
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] < min_area:
            continue
        detected.append((centroids[idx][0], centroids[idx][1]))
    return thresh, detected


def draw_centroids_overlay(image: np.ndarray, centroids: Sequence[Centroid]) -> np.ndarray:
    """Return a BGR preview that marks each centroid."""
    if image.ndim == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        color = image.copy()
    for index, (x, y) in enumerate(centroids, start=1):
        center = (int(round(x)), int(round(y)))
        cv2.circle(color, center, 12, (0, 255, 0), 2)
        cv2.putText(
            color,
            str(index),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return color
