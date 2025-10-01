import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import List

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtCore")
pytest.importorskip("PySide6.QtGui")
pytest.importorskip("PySide6.QtWidgets")
cv2 = pytest.importorskip("cv2")

from PySide6.QtWidgets import QApplication, QGraphicsScene
from PySide6.QtCore import QPointF

from editor_tif.domain.models.template import (
    ContourSignature,
    Placement,
    PlacementRule,
    Template,
)
from editor_tif.domain.services.placement import (
    apply_placement_to_item,
    get_item_min_area_rect,
    get_signature_box_vertices,
    placement_from_template,
)
from editor_tif.presentation.views.scene_items import ImageItem, Layer, ContourItem
from editor_tif.infrastructure.contour_detector import ContourDetector


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

    rect = get_item_min_area_rect(item, min_area=1.0)
    assert rect is not None
    (piv_x, piv_y), _, _ = rect

    placement = Placement(
        tx=40.0,
        ty=60.0,
        rotation_deg=30.0,
        scale_x=1.25,
        scale_y=1.20,
        piv_x=piv_x,
        piv_y=piv_y,
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


def test_placement_from_template_aligns_principal_axis():
    base_signature = ContourSignature(
        cx=0.0,
        cy=0.0,
        width=30.0,
        height=10.0,
        angle_deg=0.0,
        principal_axis=(1.0, 0.0),
    )
    rule = PlacementRule(offset_norm=(1.0, 0.5), rotation_offset_deg=30.0)
    template = Template(
        item_source_id="dummy",
        item_original_size=(10.0, 5.0),
        base_contour=base_signature,
        rule=rule,
    )

    target_signature = ContourSignature(
        cx=100.0,
        cy=200.0,
        width=40.0,
        height=20.0,
        angle_deg=0.0,
        principal_axis=(-1.0, 0.0),
    )

    placement = placement_from_template(template, target_signature)

    expected_rotation = (target_signature.angle_deg + rule.rotation_offset_deg) % 360.0
    assert placement.rotation_deg == pytest.approx(expected_rotation)

    expected_tx = target_signature.cx + (target_signature.width * 0.5)
    assert placement.tx == pytest.approx(expected_tx)
    assert placement.ty == pytest.approx(target_signature.cy)


def _sort_vertices(vertices):
    arr = np.array(vertices, dtype=np.float32)
    if arr.shape[0] == 0:
        return arr
    order = np.lexsort((arr[:, 0], arr[:, 1]))
    return arr[order]


def test_min_rect_vertices_flow_detector_to_placement(qt_app):
    image = np.zeros((120, 120), dtype=np.uint8)
    rect = ((60.0, 60.0), (50.0, 30.0), 25.0)
    rect_pts = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, rect_pts, 255)

    detector = ContourDetector(min_area=20.0)
    contours, _ = detector.detect(image)
    assert contours, "No se detectaron contornos"
    contour = contours[0]

    assert contour.min_rect_vertices is not None
    assert len(contour.min_rect_vertices) >= 4

    expected_sorted = _sort_vertices(cv2.boxPoints(rect))
    detected_sorted = _sort_vertices(contour.min_rect_vertices)
    assert detected_sorted.shape == expected_sorted.shape
    assert np.allclose(detected_sorted, expected_sorted, atol=1.5)

    signature = ContourSignature(
        cx=float(contour.cx),
        cy=float(contour.cy),
        width=float(contour.width),
        height=float(contour.height),
        angle_deg=float(contour.angle_deg),
        min_rect_vertices=[(float(x), float(y)) for x, y in contour.min_rect_vertices],
    )

    dst_vertices = get_signature_box_vertices(signature)
    dst_sorted = _sort_vertices(dst_vertices)
    assert np.allclose(dst_sorted, expected_sorted, atol=1.5)

    item = ContourItem()
    item.set_from_signature(signature)
    poly = item.polygon()
    assert poly.size() >= 4
    poly_vertices = [(poly[i].x(), poly[i].y()) for i in range(min(4, poly.size()))]
    poly_sorted = _sort_vertices(poly_vertices)
    assert np.allclose(poly_sorted, expected_sorted, atol=1.5)
