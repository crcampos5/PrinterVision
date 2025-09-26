import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import List

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtCore")
pytest.importorskip("PySide6.QtGui")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication, QGraphicsScene
from PySide6.QtCore import QPointF

from editor_tif.domain.models.template import Placement
from editor_tif.domain.services.placement import apply_placement_to_item
from editor_tif.presentation.views.scene_items import ImageItem, Layer


class _DummyDocument:
    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self._layer_seq: int = 0

    def get_tile_dimensions_mm(self):  # pragma: no cover - helper for ImageItem
        return None


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_item(mm_to_scene: float = 1.0) -> ImageItem:
    doc = _DummyDocument()
    pixels = np.ones((4, 4, 4), dtype=np.uint8) * 255
    layer = Layer(id=1, path=None, pixels=pixels)
    doc.layers.append(layer)
    return ImageItem(layer, document=doc, mm_to_scene=mm_to_scene)


def test_apply_placement_updates_layer_and_commits(qt_app):
    item = _build_item(mm_to_scene=2.0)
    emissions: list = []
    item.events.committed.connect(lambda payload: emissions.append(payload))

    placement = Placement(
        tx=40.0,
        ty=60.0,
        rotation_deg=30.0,
        scale_x=1.25,
        scale_y=1.20,
        piv_x=0.0,
        piv_y=0.0,
    )

    apply_placement_to_item(item, placement)

    assert item.layer.x == pytest.approx(20.0)
    assert item.layer.y == pytest.approx(30.0)
    assert item.layer.rotation == pytest.approx(30.0)
    assert item.layer.scale == pytest.approx((1.25 + 1.20) / 2.0)

    pos = item.pos()
    assert pos.x() == pytest.approx(placement.tx)
    assert pos.y() == pytest.approx(placement.ty)
    assert emissions and emissions[-1] is item


def test_item_change_keeps_layer_in_sync(qt_app):
    item = _build_item(mm_to_scene=1.5)
    scene = QGraphicsScene()
    scene.addItem(item)

    item.setPos(QPointF(30.0, -15.0))
    assert item.layer.x == pytest.approx(30.0 / 1.5)
    assert item.layer.y == pytest.approx(-15.0 / 1.5)

    item.setRotation(45.0)
    assert item.layer.rotation == pytest.approx(item.rotation())

    item.setScale(1.3)
    assert item.layer.scale == pytest.approx(item.scale())
