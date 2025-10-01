from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math

import numpy as np
from PySide6.QtCore import Qt, QPointF, QObject, Signal, QRectF
from PySide6.QtGui import QTransform, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsPolygonItem

from editor_tif.infrastructure.qt_image import numpy_to_qpixmap
from editor_tif.infrastructure.tif_io import load_image_data  # usado en from_source_id
from editor_tif.domain.models.template import ContourSignature


# =========================
#   DATA: LAYER
# =========================
@dataclass
class Layer:
    """Capa de imagen con metadata y transformaciones en mm."""
    id: int
    path: Optional[Path]
    pixels: np.ndarray
    photometric: Optional[str] = None
    cmyk_order: Optional[Tuple[int, int, int, int]] = None
    alpha_index: Optional[int] = None
    icc_profile: Optional[bytes] = None
    ink_names: Optional[list[str]] = None

    # Transformaciones físicas (mm)
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    opacity: float = 1.0

    transform_matrix: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None

    # Dimensiones físicas (si aplica)
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None

    # Identificador lógico (para copiar/pegar)
    source_id: Optional[str] = None

    # Marcadores internos
    is_template_overlay: bool = False


# =========================
#   EVENTS
# =========================
class _ItemEvents(QObject):
    committed = Signal(object)  # se emite cuando confirmas una interacción


# =========================
#   IMAGE ITEM
# =========================
class ImageItem(QGraphicsPixmapItem):
    """
    Item gráfico en la escena.
    - Representa un Layer (con unidades físicas mm).
    - Rotación con Shift+rueda.
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

        # Configuración inicial
        self.setTransformationMode(Qt.SmoothTransformation)
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable
            | QGraphicsPixmapItem.ItemIsSelectable
            | QGraphicsPixmapItem.ItemSendsGeometryChanges
            | QGraphicsPixmapItem.ItemIsFocusable
        )
        self.setOpacity(self.layer.opacity)
        self.setTransformOriginPoint(self.boundingRect().center())

        # Centrar el pixmap respecto al origen
        br = self.boundingRect()
        self.setOffset(-br.width() / 2.0, -br.height() / 2.0)

        self.sync_from_layer()

    # --- reconstrucción desde source_id ---
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

    # --- resolución física ---
    def _layer_mpp(self) -> Optional[Tuple[float, float]]:
        """Devuelve milímetros por pixel (x,y)."""
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
        return (w_mm / float(pw), h_mm / float(ph))

    # --- sync layer -> item ---
    def sync_from_layer(self) -> None:
        matrix = getattr(self.layer, "transform_matrix", None)

        if matrix is not None:
            try:
                a, b, tx = matrix[0]
                c, d, ty = matrix[1]
                t = QTransform(float(a), float(c), 0.0, float(b), float(d), 0.0, float(tx), float(ty), 1.0)
                self.setTransformOriginPoint(QPointF(0.0, 0.0))
                self.setPos(0.0, 0.0)
                self.setTransform(t, False)
            except (TypeError, ValueError, IndexError):
                try:
                    self.layer.transform_matrix = None
                except AttributeError:
                    pass
                matrix = None

        if matrix is None:
            self.setTransformOriginPoint(self.boundingRect().center())
            self.setPos(self.layer.x * self.mm_to_scene, self.layer.y * self.mm_to_scene)
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

        self.setOpacity(self.layer.opacity)

    # --- sync item -> layer (cuando se mueve en escena) ---
    def itemChange(self, change, value):
        if change == QGraphicsPixmapItem.ItemPositionHasChanged:
            p: QPointF = self.pos()
            self.layer.x = float(p.x()) / self.mm_to_scene
            self.layer.y = float(p.y()) / self.mm_to_scene
        elif change == QGraphicsPixmapItem.ItemRotationHasChanged:
            self.layer.rotation = float(self.rotation())
        elif change == QGraphicsPixmapItem.ItemScaleHasChanged:
            self.layer.scale = float(self.scale())
        return super().itemChange(change, value)

    # --- rotación con Shift+rueda ---
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            d = event.delta()  # ±120 por notch
            step = 2.0 if d > 0 else -2.0
            self.layer.rotation += step
            self.sync_from_layer()
            self.events.committed.emit(self)
            event.accept()
            return
        super().wheelEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        p = self.pos()
        self.layer.x = float(p.x()) / self.mm_to_scene
        self.layer.y = float(p.y()) / self.mm_to_scene
        self.events.committed.emit(self)


# =========================
#   CENTROID ITEM
# =========================
class CentroidItem(QGraphicsEllipseItem):
    """
    Marca gráfica de un centroide. (0,0) local = centro.

    Además de marcar el centro, este item puede almacenar una firma geométrica
    mínima del contorno asociado (ancho/alto del bbox y ángulo), para poder
    generar directamente un ContourSignature.
    """
    def __init__(
        self,
        radius_px: float = 6.0,
        *,
        bbox_width: float = 0.0,
        bbox_height: float = 0.0,
        angle_deg: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        r = float(radius_px)
        self.setRect(QRectF(-r, -r, 2*r, 2*r))
        self.setZValue(-5)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, False)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setPen(QPen(Qt.green, 1))
        self.setBrush(QBrush(Qt.transparent))

        # Firma geométrica mínima del contorno asociado
        self._bbox_w: float = float(bbox_width)
        self._bbox_h: float = float(bbox_height)
        self._angle_deg: float = float(angle_deg)
        self._principal_axis: Optional[Tuple[float, float]] = None

    # ---- Helpers geométricos ----
    def center_scene_pos(self) -> QPointF:
        """Devuelve la posición del centroide en coordenadas de escena."""
        return self.scenePos()

    def set_signature(
        self,
        *,
        bbox_width: float,
        bbox_height: float,
        angle_deg: float,
        principal_axis: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Actualiza ancho/alto del bbox y ángulo del contorno asociado."""
        self._bbox_w = float(bbox_width)
        self._bbox_h = float(bbox_height)
        self._angle_deg = float(angle_deg)
        if principal_axis is None:
            self._principal_axis = None
        else:
            self._principal_axis = (float(principal_axis[0]), float(principal_axis[1]))

    def get_signature(self) -> Tuple[float, float, float, Optional[Tuple[float, float]]]:
        """Obtiene (bbox_width, bbox_height, angle_deg, principal_axis)."""
        return self._bbox_w, self._bbox_h, self._angle_deg, self._principal_axis

    # ---- Export directo a dominio ----
    def to_contour_signature(self) -> ContourSignature:
        """
        Crea un ContourSignature (agnóstico de UI) desde este item.
        - cx, cy: tomados de la posición en escena
        - width, height, angle_deg: desde su firma local (_bbox_w/_bbox_h/_angle_deg)
        """
        pos = self.center_scene_pos()
        half_w = self._bbox_w * 0.5
        half_h = self._bbox_h * 0.5
        angle_rad = math.radians(self._angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        local_vertices = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        rect_vertices = [
            (
                float(pos.x()) + cos_a * vx - sin_a * vy,
                float(pos.y()) + sin_a * vx + cos_a * vy,
            )
            for vx, vy in local_vertices
        ]
        return ContourSignature(
            cx=float(pos.x()),
            cy=float(pos.y()),
            width=self._bbox_w,
            height=self._bbox_h,
            angle_deg=self._angle_deg,
            principal_axis=self._principal_axis,
            min_rect_vertices=rect_vertices,
        )


class ContourItem(QGraphicsPolygonItem):
    """
    Contorno seleccionable. Se construye desde una ContourSignature (cx,cy,w,h,angle).
    Usamos un bbox rotado (rombo/rectángulo) como representación mínima.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setZValue(-1)  # por encima del fondo, por debajo de los ImageItem
        self.setFlags(
            QGraphicsPolygonItem.ItemIsSelectable
            | QGraphicsPolygonItem.ItemIsFocusable
        )
        self.setPen(QPen(Qt.red, 0.5, Qt.SolidLine))
        self.setBrush(QBrush(Qt.transparent))
        self._sig: Optional[ContourSignature] = None

    def set_from_signature(self, sig: ContourSignature):
        """Si sig.polygon existe (>=3 pts), dibuja ese polígono. 
        Si no, construye un rect centrado (w×h), lo rota y lo traslada."""
        self._sig = sig

        # 1) Intentar dibujar el polígono real
        pts = getattr(sig, "polygon", None)
        if pts and len(pts) >= 3:
            # pts puede venir como [(x,y), ...] o [QPointF,...]
            qpts = []
            for p in pts:
                if hasattr(p, "x"):  # QPointF
                    qpts.append(QPointF(float(p.x()), float(p.y())))
                else:                # tupla/lista (x,y)
                    qpts.append(QPointF(float(p[0]), float(p[1])))
            self.setPolygon(QPolygonF(qpts))
            return

        # 2) Intentar dibujar usando los vértices del rectángulo mínimo
        rect_vertices = getattr(sig, "min_rect_vertices", None)
        if rect_vertices and len(rect_vertices) >= 4:
            qpts = []
            for p in rect_vertices[:4]:
                if hasattr(p, "x") and hasattr(p, "y"):
                    qpts.append(QPointF(float(p.x()), float(p.y())))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    qpts.append(QPointF(float(p[0]), float(p[1])))
            if qpts:
                poly = QPolygonF(qpts)
                if poly.size() >= 4:
                    self.setPolygon(poly)
                    return

        # 3) Fallback: rectángulo por firma (w,h,angle,cx,cy)
        w = float(max(sig.width, 1e-6))
        h = float(max(sig.height, 1e-6))
        hw, hh = w * 0.5, h * 0.5

        poly_local = QPolygonF([
            QPointF(-hw, -hh), QPointF(+hw, -hh),
            QPointF(+hw, +hh), QPointF(-hw, +hh)
        ])

        t = QTransform()
        t.rotate(float(sig.angle_deg))
        poly_rot = t.map(poly_local)

        poly_rot_trans = QPolygonF([QPointF(p.x() + float(sig.cx), p.y() + float(sig.cy)) for p in poly_rot])
        self.setPolygon(poly_rot_trans)


    def to_contour_signature(self) -> ContourSignature:
        if self._sig is None:
            # Si no hay firma almacenada, la reconstruimos aproximando desde el polygon actual
            poly = self.polygon()
            if poly.isEmpty():
                return ContourSignature(0, 0, 0, 0, 0)
            # bbox axis-aligned
            br = poly.boundingRect()
            cx, cy = br.center().x(), br.center().y()
            rect_vertices = [
                (br.left(), br.top()),
                (br.right(), br.top()),
                (br.right(), br.bottom()),
                (br.left(), br.bottom()),
            ]
            return ContourSignature(cx, cy, br.width(), br.height(), 0.0, min_rect_vertices=rect_vertices)
        return self._sig
