# src/editor_tif/views/scene_items.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem

from editor_tif.utils.qt import numpy_to_qpixmap


@dataclass
class Layer:
    id: int
    path: Optional[Path]
    pixels: np.ndarray            # imagen numpy (dtype/canales intactos)
    photometric: Optional[str] = None
    cmyk_order: Optional[Tuple[int,int,int,int]] = None
    alpha_index: Optional[int] = None
    icc_profile: Optional[bytes] = None
    ink_names: Optional[list[str]] = None
    # Transformaciones de escena:
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0         # grados
    scale: float = 1.0
    opacity: float = 1.0


class ImageItem(QGraphicsPixmapItem):
    """Item gráfico que representa un Layer y permite manipularlo en la escena."""

    def __init__(self, layer: Layer):
        # Pixmap de preview (RGB8) usando las hints de photometric/CMYK para pantalla
        pix = numpy_to_qpixmap(layer.pixels,
                               photometric_hint=layer.photometric,
                               cmyk_order=layer.cmyk_order,
                               alpha_index=layer.alpha_index)
        super().__init__(pix)
        self.layer = layer

        self.setTransformationMode(Qt.SmoothTransformation)
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable
            | QGraphicsPixmapItem.ItemIsSelectable
            | QGraphicsPixmapItem.ItemSendsGeometryChanges
            | QGraphicsPixmapItem.ItemIsFocusable
        )
        self.setOpacity(self.layer.opacity)
        self.setTransformOriginPoint(self.boundingRect().center())
        self.sync_from_layer()

    # Sincroniza item -> capa al mover con el mouse
    def itemChange(self, change, value):  # noqa: N802 (Qt)
        if change == QGraphicsPixmapItem.ItemPositionHasChanged and self.isSelected():
            p: QPointF = value
            self.layer.x, self.layer.y = float(p.x()), float(p.y())
        if change == QGraphicsPixmapItem.ItemSelectedHasChanged:
            # refrescar origen por si se reescala
            self.setTransformOriginPoint(self.boundingRect().center())
        return super().itemChange(change, value)

    # Tecla rápida: rotación fina
    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta().y()
            step = 2.0 if delta > 0 else -2.0
            self.layer.rotation += step
            self.sync_from_layer()
            event.accept()
            return
        super().wheelEvent(event)

    # Aplicar la info del layer al item
    def sync_from_layer(self) -> None:
        self.setPos(self.layer.x, self.layer.y)
        self.setOpacity(self.layer.opacity)
        t = QTransform()
        t.scale(self.layer.scale, self.layer.scale)
        self.setTransform(t)
        self.setRotation(self.layer.rotation)
