# src/editor_tif/domain/services/placement.py
from __future__ import annotations

from typing import Sequence, Tuple, List, Callable, Optional
from dataclasses import replace

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
    FitMode,
    Placement,
)

# -------------------------
# Tipos auxiliares
# -------------------------
Centroid = Tuple[float, float]


# -------------------------
# CÁLCULO DE PLACEMENT
# -------------------------
def _compute_scales(
    item_w: float,
    item_h: float,
    bbox_w: float,
    bbox_h: float,
    fit_mode: FitMode,
    keep_aspect_ratio: bool,
) -> Tuple[float, float]:
    """Devuelve (scale_x, scale_y) para llevar el item al bbox según el fit_mode."""
    if item_w <= 0 or item_h <= 0:
        return 1.0, 1.0

    sx_w = bbox_w / item_w
    sy_h = bbox_h / item_h

    if fit_mode == FitMode.NONE:
        sx, sy = 1.0, 1.0

    elif fit_mode == FitMode.FIT_WIDTH:
        sx = sx_w
        sy = sx if keep_aspect_ratio else 1.0

    elif fit_mode == FitMode.FIT_HEIGHT:
        sy = sy_h
        sx = sy if keep_aspect_ratio else 1.0

    elif fit_mode == FitMode.FIT_SHORT:
        s = min(sx_w, sy_h)
        sx = sy = s

    elif fit_mode == FitMode.FIT_LONG:
        s = max(sx_w, sy_h)
        sx = sy = s

    elif fit_mode == FitMode.STRETCH:
        sx = sx_w
        sy = sy_h

    else:
        sx, sy = 1.0, 1.0

    if keep_aspect_ratio and fit_mode in (FitMode.FIT_WIDTH, FitMode.FIT_HEIGHT):
        # ya sincronizado arriba
        pass

    return float(sx), float(sy)


def _anchor_offset_local(item_w: float, item_h: float, anchor_norm: Tuple[float, float]) -> Tuple[float, float]:
    """
    Devuelve el offset (en coords locales del item) para que el punto anchor_norm quede en el origen.
    Ej: anchor=(0.5,0.5) -> centra el item.
    """
    ax, ay = anchor_norm
    return -(ax * item_w), -(ay * item_h)


def _offset_in_bbox(rule: PlacementRule, bbox_w: float, bbox_h: float) -> Tuple[float, float]:
    """
    Desplazamiento normalizado (0..1) dentro del bbox del contorno, antes de rotar.
    (0.5,0.5) centra el anclaje en el centro del bbox.
    """
    ox = rule.offset_norm[0] * bbox_w
    oy = rule.offset_norm[1] * bbox_h
    return ox, oy


def placement_from_template(
    template: Template,
    target: ContourSignature,
) -> Placement:
    """
    Calcula la Placement final (tx, ty, rot, sx, sy) de un item de tamaño original
    template.item_original_size sobre un contorno destino (target).

    Convenciones:
    - angle_deg del contorno es antihoraria.
    - rotation_offset_deg se suma a la orientación del contorno.
    - offset_norm se aplica en el marco del bbox del contorno (antes de rotar).
    - anchor_norm define el punto del item que se alinea al destino.
    """
    iw, ih = template.item_original_size
    rule = template.rule

    # 1) escalas
    sx, sy = _compute_scales(iw, ih, target.width, target.height, rule.fit_mode, rule.keep_aspect_ratio)

    # 2) tamaño efectivo post-escala
    eff_w = iw * sx
    eff_h = ih * sy

    # 3) offset local por anclaje (mueve el item para que anchor quede en el (0,0) local)
    off_local_x, off_local_y = _anchor_offset_local(eff_w, eff_h, rule.anchor_norm)

    # 4) offset dentro del bbox del contorno, en marco del contorno (sin rotación)
    bbox_off_x, bbox_off_y = _offset_in_bbox(rule, target.width, target.height)

    # 5) rotación final
    rot = (target.angle_deg + rule.rotation_offset_deg) % 360.0
    rot_rad = math.radians(rot)

    # 6) posición final:
    #    - partimos del centro del contorno
    #    - aplicamos offset del bbox (en el marco del contorno)
    #    - y aplicamos el offset local del anclaje (ya escalado) y lo rotamos
    # Centro del bbox (cx, cy)
    cx, cy = target.cx, target.cy

    # Offset del bbox en marco del contorno (x horizontal del bbox, y vertical del bbox)
    # Rotamos ese offset por el ángulo del contorno para pasarlo a escena:
    rx = math.cos(rot_rad) * (bbox_off_x - target.width * 0.5) - math.sin(rot_rad) * (bbox_off_y - target.height * 0.5)
    ry = math.sin(rot_rad) * (bbox_off_x - target.width * 0.5) + math.cos(rot_rad) * (bbox_off_y - target.height * 0.5)

    # Offset local del anclaje del item se aplica tal cual, porque setOffset() lo compensa al dibujar
    # Aquí lo convertimos a una traslación en escena rotando el vector (off_local_x, off_local_y):
    off_rot_x = math.cos(rot_rad) * off_local_x - math.sin(rot_rad) * off_local_y
    off_rot_y = math.sin(rot_rad) * off_local_x + math.cos(rot_rad) * off_local_y

    tx = cx + rx + off_rot_x
    ty = cy + ry + off_rot_y

    return Placement(tx=tx, ty=ty, rotation_deg=rot, scale_x=sx, scale_y=sy)


# -------------------------
# APLICAR A ITEMS / ESCENA
# -------------------------
def apply_placement_to_item(
    item: ImageItem,
    placement: Placement,
) -> None:
    """
    Aplica la transformación calculada a un ImageItem.
    - Usa el centro del boundingRect como pivot de rotación.
    - Usa setOffset() para compensar el anclaje.
    """
    br = item.boundingRect()
    # setTransformOriginPoint -> pivot en el centro visual del item (post-offset)
    item.setTransformOriginPoint(br.center())

    # Al usar setOffset(), movemos el “lienzo local” del item. Como Placement ya incluye
    # la compensación del anclaje, dejamos el offset en (0,0) y confiamos en tx/ty.
    # Si deseas forzar que el anclaje quede exactamente en el centro del QPixmap, podrías
    # calcular aquí un offset distinto. En este flujo, no es necesario.
    item.setOffset(0.0, 0.0)

    # Rotación
    item.setRotation(placement.rotation_deg)
    # Escalas (en ImageItem, suele implementarse con una propiedad scale o matriz)
    item.setScale(max(placement.scale_x, placement.scale_y))  # si solo soportas escala uniforme

    # Posición final
    item.setPos(QPointF(placement.tx, placement.ty))


def _clone_from_item(src: QGraphicsItem) -> QGraphicsItem:
    """
    Clona un QGraphicsItem. Idealmente, ImageItem provee .clone().
    Fallback para QGraphicsPixmapItem.
    """
    if hasattr(src, "clone") and callable(src.clone):
        return src.clone()

    if isinstance(src, QGraphicsPixmapItem):
        c = QGraphicsPixmapItem(src.pixmap())
        c.setTransformationMode(src.transformationMode())
        return c

    raise TypeError("Implementa .clone() en tu ImageItem o maneja este tipo aquí.")


def clone_item_to_centroids(
    scene: QGraphicsScene,
    source_item: ImageItem,
    document,
    mm_to_scene: float,
    bind_callback: Optional[Callable[[ImageItem], None]] = None,
) -> List[ImageItem]:
    """
    Clona un ImageItem en cada CentroidItem de la escena, generando un Layer por clon.
    Conserva metadata CMYK/ICC y estados; posiciona por centroide.
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

        # Alinear pivot al centro visual
        br = item.boundingRect()
        item.setTransformOriginPoint(br.center())

        # 3) posicionar en el centroide (centro geométrico)
        item.setOffset(-br.width() / 2.0, -br.height() / 2.0)
        item.setPos(c.center_scene_pos())

        # Sincronizar layer.x/y (en mm)
        layer.x = item.pos().x() / mm_to_scene
        layer.y = item.pos().y() / mm_to_scene

        scene.addItem(item)
        if bind_callback:
            bind_callback(item)  # para SelectionHandler.bind_item

        created.append(item)

    return created


# -------------------------
# MOSAICO (OPENCV/NUMPY)
# -------------------------
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


# -------------------------
# APLICAR TEMPLATE A TODOS LOS CONTORNOS (vía CentroidItem)
# -------------------------
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
