import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtGui")

from PySide6.QtGui import QImage, qRgba

from src.editor_tif.infrastructure.qt_image import qimage_to_numpy


def test_qimage_to_numpy_handles_memoryview_bits():
    width, height = 3, 2
    image = QImage(width, height, QImage.Format_RGBA8888)
    image.fill(qRgba(10, 20, 30, 40))

    result = qimage_to_numpy(image)

    assert result.shape == (height, width, 4)
    assert result.dtype == np.uint8
    # Ensure the color matches the fill value to confirm data integrity.
    expected_pixel = np.array([10, 20, 30, 40], dtype=np.uint8)
    assert np.all(result[0, 0] == expected_pixel)
