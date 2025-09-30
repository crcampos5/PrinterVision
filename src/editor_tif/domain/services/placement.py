# src/editor_tif/domain/services/placement.py
from __future__ import annotations

from typing import Sequence, Tuple, List, Callable, Optional
import math

from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene
from PySide6.QtCore import QPointF

# Vista/UI
from editor_tif.presentation.views.scene_items import CentroidItem, ImageItem, Layer

# Dominio
from editor_tif.domain.models.template import (
    Template,
    ContourSignature,
    PlacementRule,
    Placement,
)

# -------------------------
# Tipos auxiliares
# -------------------------
Centroid = Tuple[float, float]


# =========================================================
# Colocación desde PLANTILLA (sin escalado, centro como ancla)
# =========================================================
def _clamp01_pair(v) -> Tuple[float, float]:
    """Devuelve (x,y) clamp en [0,1], con fallback (0.5,0.5) si algo viene mal."""
    try:
        x = max(0.0, min(1.0, float(v[0])))
        y = max(0.0, min(1.0, float(v[1])))
        return x, y
    except Exception:
        return 0.5, 0.5


def placement_from_template(
    template: Template,
    target: ContourSignature,
) -> Placement:
    """
    Calcula Placement para reproducir la RELACIÓN RELATIVA guardada en la plantilla:
      - Sin escalado (sx = sy = 1).
      - El centro del ítem se coloca en: (cx_dest, cy_dest) + R(angle_dest) * delta_local_dest
        donde delta_local_dest se obtiene de offset_norm y tamaño del bbox destino.
      - El desplazamiento se orienta con angle_dest únicamente, conservando el anclaje relativo.
      - Rotación final del ítem: angle_dest + rotation_offset_deg.

    Asume que template.rule.offset_norm y template.rule.rotation_offset_deg
    fueron medidos al CREAR la plantilla respecto al contorno base.
    """
    rule: PlacementRule = template.rule

    # 1) Sin escalado
    sx = sy = 1.0

    # 2) Offset normalizado (en marco del bbox del contorno base) -> clamp y usar como relación
    off_xn, off_yn = _clamp01_pair(getattr(rule, "offset_norm", (0.5, 0.5)))

    base_angle_deg = float(target.angle_deg)

    # 4) Rotación final
    rot = (base_angle_deg + float(getattr(rule, "rotation_offset_deg", 0.0)))
    angle_rad = math.radians(base_angle_deg)
    # 5) Offset local destino (marco del bbox destino, sin rotar)
    #    (0.5,0.5) significa el centro del bbox -> delta (0,0)
    dx_local = (off_xn - 0.5) * target.width
    dy_local = (off_yn - 0.5) * target.height

    # 6) Convertir delta_local al marco de ESCENA usando la orientación del contorno
    dx_scene = math.cos(angle_rad) * dx_local - math.sin(angle_rad) * dy_local
    dy_scene = math.sin(angle_rad) * dx_local + math.cos(angle_rad) * dy_local

    # 7) Posición final del CENTRO del ítem
    tx = target.cx + dx_scene
    ty = target.cy + dy_scene

    return Placement(tx=tx, ty=ty, rotation_deg=rot, scale_x=sx, scale_y=sy, piv_x=target.cx, piv_y=target.cy)


# =========================================================
# Aplicar Placement a un ImageItem (centro en tx,ty)
# =========================================================
def apply_placement_to_item(
    item: ImageItem,
    placement: Placement,
) -> None:
    """
    Aplica Placement a un ImageItem respetando el pivote del contorno detectado.
    - El pivote en escena (placement.piv_x, placement.piv_y) se mapea al marco local
      del ítem y se usa como origen de transformación.
    - El offset local se ajusta para que setPos(tx, ty) sitúe dicho pivote sobre el
      centroide objetivo.
    - Sin escalado (1.0).
    """
    br = item.boundingRect()

    pivot_scene = QPointF(float(getattr(placement, "piv_x", placement.tx)),
                          float(getattr(placement, "piv_y", placement.ty)))

    current_offset = item.offset()

    try:
        pivot_local = item.mapFromScene(pivot_scene)
        pivot_relative = pivot_local - current_offset
        if not (math.isfinite(pivot_relative.x()) and math.isfinite(pivot_relative.y())):
            raise ValueError("invalid pivot coordinates")
    except Exception:
        center_local = br.center()
        pivot_relative = center_local - current_offset

    # Ajustar el origen local del pixmap para que el pivote quede en (0,0)
    item.setOffset(-pivot_relative.x(), -pivot_relative.y())

    # Con el pivote en (0,0), usarlo como origen de transformaciones
    item.setTransformOriginPoint(QPointF(0.0, 0.0))

    # Sincronizar layer <-> item para que doc.save utilice la pose real
    layer = getattr(item, "layer", None)
    mm_to_scene = float(getattr(item, "mm_to_scene", 1.0) or 1.0)

    if layer is not None:
        layer.x = float(placement.tx) / mm_to_scene
        layer.y = float(placement.ty) / mm_to_scene
        layer.rotation = float(placement.rotation_deg)

        sx = float(placement.scale_x)
        sy = float(placement.scale_y)
        if math.isfinite(sx) and math.isfinite(sy):
            if math.isclose(sx, sy, rel_tol=1e-6, abs_tol=1e-6):
                layer.scale = sx
            else:
                layer.scale = (sx + sy) * 0.5
        elif math.isfinite(sx):
            layer.scale = sx
        elif math.isfinite(sy):
            layer.scale = sy

    # Reflejar el estado del layer en el item
    if hasattr(item, "sync_from_layer") and callable(item.sync_from_layer):
        item.sync_from_layer()
    else:
        item.setRotation(placement.rotation_deg)
        item.setScale(float(placement.scale_x))
        item.setPos(QPointF(placement.tx, placement.ty))

    events = getattr(item, "events", None)
    committed = getattr(events, "committed", None)
    emit = getattr(committed, "emit", None)
    if callable(emit):
        emit(item)


# =========================================================
# Utilidades de clonado por centroides (flujo legacy)
# =========================================================

def clone_item_to_centroids(
    scene: QGraphicsScene,
    source_item: ImageItem,
    document,
    mm_to_scene: float,
    bind_callback: Optional[Callable[[ImageItem], None]] = None,
) -> List[ImageItem]:
    """
    Clona un ImageItem en cada CentroidItem de la escena, generando un Layer por clon.
    Conserva metadata CMYK/ICC y estados; posiciona por centroide (centro).
    """
    centroids = [it for it in scene.items() if isinstance(it, CentroidItem)]
    if not centroids:
        return []

    src_layer = source_item.layer
    created: List[ImageItem] = []

    for c in centroids:
        # 1) nuevo Layer (independiente)
        document._layer_seq += 1
        layer = Layer(
            id=document._layer_seq,
            path=src_layer.path,
            pixels=src_layer.pixels,  # compartir píxeles: transformación es por item
            photometric=src_layer.photometric,
            cmyk_order=src_layer.cmyk_order,
            alpha_index=src_layer.alpha_index,
            icc_profile=src_layer.icc_profile,
            ink_names=src_layer.ink_names,
            width_mm=src_layer.width_mm,
            height_mm=src_layer.height_mm,
            x=src_layer.x,
            y=src_layer.y,
            rotation=src_layer.rotation,
            scale=src_layer.scale,
            opacity=src_layer.opacity,
        )
        document.layers.append(layer)

        # 2) nuevo ImageItem ligado a ese Layer
        item = ImageItem(layer, document=document, mm_to_scene=mm_to_scene)

        # Pivot y offset para posicionar por el centro
        br = item.boundingRect()
        item.setTransformOriginPoint(br.center())
        item.setOffset(-br.width() / 2.0, -br.height() / 2.0)

        # 3) posicionar en el centroide (centro geométrico)
        item.setPos(c.center_scene_pos())

        # Sincronizar layer.x/y (en mm)
        layer.x = item.pos().x() / mm_to_scene
        layer.y = item.pos().y() / mm_to_scene

        scene.addItem(item)
        if bind_callback:
            bind_callback(item)  # para SelectionHandler.bind_item

        created.append(item)

    return created


# =========================================================
# OpenCV/NumPy (mosaico por centroides)
# =========================================================
def place_tile_on_centroids(canvas, tile, centroids: Sequence[Centroid]):
    """
    Superpone la imagen `tile` centrada en cada centroid sobre `canvas`.
    Evita salir de límites y copia solo píxeles no-cero del tile.
    """
    import numpy as np

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
        mask = tile_slice > 0
        region[mask] = tile_slice[mask]
        result[y0:y1, x0:x1] = region

    return result


# =========================================================
# Aplicar plantilla sobre todos los centroides (adaptador)
# =========================================================
def apply_template_over_scene_centroids(
    scene: QGraphicsScene,
    template: Template,
    source_item: ImageItem,
    document,
    mm_to_scene: float,
    get_contour_signature: Callable[[CentroidItem], ContourSignature],
    bind_callback: Optional[Callable[[ImageItem], None]] = None,
) -> List[ImageItem]:
    """
    Clona y coloca `source_item` sobre cada CentroidItem usando la `template`.
    Necesita un adaptador `get_contour_signature` que traduzca un CentroidItem
    a un ContourSignature (cx, cy, width, height, angle_deg).
    """
    centroids = [it for it in scene.items() if isinstance(it, CentroidItem)]
    if not centroids:
        return []

    src_layer = source_item.layer
    created: List[ImageItem] = []

    for c in centroids:
        target_sig = get_contour_signature(c)
        placement = placement_from_template(template, target_sig)

        # 1) nuevo Layer
        document._layer_seq += 1
        layer = Layer(
            id=document._layer_seq,
            path=src_layer.path,
            pixels=src_layer.pixels,
            photometric=src_layer.photometric,
            cmyk_order=src_layer.cmyk_order,
            alpha_index=src_layer.alpha_index,
            icc_profile=src_layer.icc_profile,
            ink_names=src_layer.ink_names,
            width_mm=src_layer.width_mm,
            height_mm=src_layer.height_mm,
            x=src_layer.x,
            y=src_layer.y,
            rotation=src_layer.rotation,
            scale=src_layer.scale,
            opacity=src_layer.opacity,
        )
        document.layers.append(layer)

        # 2) nuevo ImageItem
        item = ImageItem(layer, document=document, mm_to_scene=mm_to_scene)
        apply_placement_to_item(item, placement)

        # Sincronizar layer.x/y (mm)
        layer.x = item.pos().x() / mm_to_scene
        layer.y = item.pos().y() / mm_to_scene

        scene.addItem(item)
        if bind_callback:
            bind_callback(item)

        created.append(item)

    return created
