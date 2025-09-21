from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QPointF, QObject, Signal, QRectF
from PySide6.QtGui import QTransform, QPen, QBrush
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsEllipseItem

from editor_tif.infrastructure.qt_image import numpy_to_qpixmap
from editor_tif.infrastructure.tif_io import load_image_data  # para from_source_id


@dataclass
class Layer:
    id: int
    path: Optional[Path]
    pixels: np.ndarray
    photometric: Optional[str] = None
    cmyk_order: Optional[Tuple[int, int, int, int]] = None
    alpha_index: Optional[int] = None
    icc_profile: Optional[bytes] = None
    ink_names: Optional[list[str]] = None
    # Transformaciones en UNIDADES FÍSICAS (mm):
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    opacity: float = 1.0
    # Dimensiones físicas propias (si aplica):
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    # Para copiar/pegar
    source_id: Optional[str] = None


class _ItemEvents(QObject):
    committed = Signal(object)  # se emite cuando confirmas una interacción


class ImageItem(QGraphicsPixmapItem):
    """
    Item gráfico en escena "milímetros".
    - Rotación con Shift + rueda (no afecta el zoom del viewer).
    """

    def __init__(self, layer: Layer, document, mm_to_scene: float = 1.0):
        pix = numpy_to_qpixmap(
            layer.pixels,
            photometric_hint=layer.photometric,
            cmyk_order=layer.cmyk_order,
            alpha_index=layer.alpha_index,
        )
        super().__init__(pix)
        self.layer = layer
        self.document = document
        self.mm_to_scene = float(mm_to_scene)
        self.source_id = layer.source_id or (str(layer.path) if layer.path else "")
        self.events = _ItemEvents()
        self.setTransformationMode(Qt.SmoothTransformation)
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable
            | QGraphicsPixmapItem.ItemIsSelectable
            | QGraphicsPixmapItem.ItemSendsGeometryChanges
            | QGraphicsPixmapItem.ItemIsFocusable
        )
        self.setOpacity(self.layer.opacity)
        self.setTransformOriginPoint(self.boundingRect().center())
        br = self.boundingRect()
        self.setOffset(-br.width() / 2.0, -br.height() / 2.0)
        self.sync_from_layer()

    # --- reconstrucción desde identificador (para pegar) ---
    @classmethod
    def from_source_id(cls, source_id: str, document, mm_to_scene: float = 1.0):
        path = Path(source_id)
        data = load_image_data(path)
        if data is None:
            raise ValueError(f"No se pudo cargar imagen desde {path}")
        document._layer_seq += 1
        layer = Layer(
            id=document._layer_seq,
            path=path,
            pixels=data.pixels,
            photometric=data.photometric,
            cmyk_order=data.cmyk_order,
            alpha_index=data.alpha_index,
            width_mm=data.width_mm,
            height_mm=data.height_mm,
            icc_profile=getattr(data, "icc_profile", None),
            ink_names=getattr(data, "ink_names", None),
            source_id=str(path),
        )
        document.layers.append(layer)
        return cls(layer, document=document, mm_to_scene=mm_to_scene)

    # === Helpers ===
    def _layer_mpp(self) -> Optional[Tuple[float, float]]:
        if self.layer.width_mm is not None and self.layer.height_mm is not None:
            w_mm, h_mm = self.layer.width_mm, self.layer.height_mm
        else:
            dims = self.document.get_tile_dimensions_mm()
            if dims is None:
                return None
            w_mm, h_mm = float(dims[0]), float(dims[1])

        ph, pw = self.layer.pixels.shape[:2]
        if pw == 0 or ph == 0:
            return None
        return (w_mm / float(pw), h_mm / float(ph))  # (mpp_x, mpp_y)

    # === Sincronización layer->item ===
    def sync_from_layer(self) -> None:
        self.setPos(self.layer.x * self.mm_to_scene, self.layer.y * self.mm_to_scene)
        self.setOpacity(self.layer.opacity)
        t = QTransform()
        rot = float(getattr(self.layer, "rotation", 0.0))
        if rot:
            t.rotate(rot)
        mpp = self._layer_mpp()
        if mpp is not None:
            sx_base = mpp[0] * self.mm_to_scene
            sy_base = mpp[1] * self.mm_to_scene
        else:
            sx_base = sy_base = 1.0
        s_user = float(getattr(self.layer, "scale", 1.0))
        t.scale(sx_base * s_user, sy_base * s_user)
        self.setTransform(t, False)

    # === Sincronización item->layer (drag en escena) ===
    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsPixmapItem.ItemPositionHasChanged:
            p: QPointF = self.pos()
            self.layer.x = float(p.x()) / self.mm_to_scene
            self.layer.y = float(p.y()) / self.mm_to_scene
        return super().itemChange(change, value)

    # === Rotación con Shift + rueda (pasos de 2°) ===
    def wheelEvent(self, event):  # noqa: N802
        # Rotar solo si está presionado Shift (el zoom ya quedó Ctrl+rueda en el viewer)
        if event.modifiers() & Qt.ShiftModifier:
            d = event.delta()  # QGraphicsSceneWheelEvent: entero tipo "ticks" (±120 por notch)
            step = 2.0 if d > 0 else -2.0
            self.layer.rotation += step
            self.sync_from_layer()
            # Notificamos commit para que quede en el undo/redo como un paso
            self.events.committed.emit(self)
            event.accept()
            return

        # Sin Shift, deja pasar el evento (para scroll normal del view)
        super().wheelEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        p = self.pos()
        self.layer.x = float(p.x()) / self.mm_to_scene
        self.layer.y = float(p.y()) / self.mm_to_scene
        self.events.committed.emit(self)


class CentroidItem(QGraphicsEllipseItem):
    """Marca de centroide. El (0,0) local es el centro."""
    def __init__(self, radius_px: float = 6.0, parent=None):
        super().__init__(parent)
        r = float(radius_px)
        self.setRect(QRectF(-r, -r, 2*r, 2*r))
        self.setZValue(-5)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, False)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setPen(QPen(Qt.green, 1))
        self.setBrush(QBrush(Qt.transparent))

    def center_scene_pos(self):
        return self.scenePos()
