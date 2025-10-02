# src/editor_tif/domain/services/placement.py
from __future__ import annotations

from typing import Sequence, Tuple, List, Callable, Optional, Iterable
import math

import cv2
import numpy as np

from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene
from PySide6.QtCore import QPointF
from PySide6.QtGui import QTransform

# Vista/UI
from editor_tif.presentation.views.scene_items import CentroidItem, ImageItem, Layer

# Infraestructura
from editor_tif.infrastructure.qt_image import qimage_to_numpy

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


def _order_box_points(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = np.array(list(points), dtype=np.float32)
    if pts.shape != (4, 2):
        return [(float(p[0]), float(p[1])) for p in pts]

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = [None] * 4
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    return [(float(p[0]), float(p[1])) for p in ordered]


def _matrix_to_tuple(matrix: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
    )


def _matrix_to_qtransform(matrix: np.ndarray) -> QTransform:
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    print("matrix: ", matrix)
    return QTransform(float(a), float(c), 0.0, float(b), float(d), 0.0, float(tx), float(ty), 1.0)


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


def get_item_min_area_rect_local_vertices(
    item: ImageItem,
    *,
    threshold: int = 127,
    min_area: float = 100.0,
) -> Optional[dict]:
    """Calcula los vértices del minAreaRect del contenido del item en coords locales.

    Devuelve un diccionario serializable con ``vertices`` (orden TL, TR, BR, BL),
    ``center`` (en coords locales), ``size`` y ``angle``. Si no se puede estimar,
    retorna ``None``.
    """

    pixmap = item.pixmap()
    if pixmap.isNull():
        return None

    image = pixmap.toImage()
    array = qimage_to_numpy(image)

    if array.ndim == 3:
        channels = array.shape[2]
        if channels >= 4:
            gray = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
        elif channels == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            gray = array[..., 0]
    else:
        gray = array.astype(np.uint8)

    gray = np.ascontiguousarray(gray)

    if threshold is None or threshold < 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < float(min_area):
        return None

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    ordered = _order_box_points(box)

    offset = item.offset()
    off_x = float(offset.x())
    off_y = float(offset.y())

    vertices_local = [(x + off_x, y + off_y) for x, y in ordered]
    center_local = (float(rect[0][0]) + off_x, float(rect[0][1]) + off_y)
    size = (float(rect[1][0]), float(rect[1][1]))
    angle = float(rect[2])

    return {
        "vertices": [[float(x), float(y)] for x, y in vertices_local],
        "center": [float(center_local[0]), float(center_local[1])],
        "size": [float(size[0]), float(size[1])],
        "angle": angle,
    }


def get_signature_box_vertices(signature: ContourSignature) -> List[Tuple[float, float]]:
    """Devuelve los vértices ordenados (TL,TR,BR,BL) del rect destino en escena."""

    rect_vertices = getattr(signature, "min_rect_vertices", None)
    box = None
    if rect_vertices:
        try:
            pts = np.array(rect_vertices, dtype=np.float32)
            if pts.shape[0] >= 4:
                box = pts[:4]
        except (TypeError, ValueError):
            box = None

    if box is None and signature.polygon:
        try:
            pts = np.array(signature.polygon, dtype=np.float32)
            if pts.ndim == 2:
                pts = pts.reshape((-1, 1, 2))
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
        except (cv2.error, ValueError, TypeError):
            box = None

    if box is None:
        rect = ((float(signature.cx), float(signature.cy)), (float(signature.width), float(signature.height)), float(signature.angle_deg))
        box = cv2.boxPoints(rect)

    return _order_box_points(box)


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

    rect_meta = template.meta.get("item_min_area_rect") if isinstance(template.meta, dict) else None
    src_vertices: Optional[List[Tuple[float, float]]] = None
    src_vertices_scene: Optional[List[Tuple[float, float]]] = None
    if isinstance(rect_meta, dict):
        verts = rect_meta.get("vertices")
        if isinstance(verts, list) and len(verts) >= 4:
            try:
                src_vertices = [
                    (float(v[0]), float(v[1]))
                    for v in verts
                ][:4]
            except (TypeError, ValueError, IndexError):
                src_vertices = None
        scene_verts = rect_meta.get("scene_vertices")
        if isinstance(scene_verts, list) and len(scene_verts) >= 4:
            try:
                src_vertices_scene = [
                    (float(v[0]), float(v[1]))
                    for v in scene_verts
                ][:4]
            except (TypeError, ValueError, IndexError):
                src_vertices_scene = None
    if src_vertices_scene is None and isinstance(template.meta, dict):
        scene_meta = template.meta.get("item_min_area_rect_scene")
        if isinstance(scene_meta, list) and len(scene_meta) >= 4:
            try:
                src_vertices_scene = [
                    (float(v[0]), float(v[1]))
                    for v in scene_meta
                ][:4]
            except (TypeError, ValueError, IndexError):
                src_vertices_scene = None

    if src_vertices is None:
        # Fallback: usar bbox centrado en el origen local
        w = float(template.item_original_size[0])
        h = float(template.item_original_size[1])
        half_w = w / 2.0
        half_h = h / 2.0
        src_vertices = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]

    dst_vertices = get_signature_box_vertices(target)
    dst_vertices_np = np.array(dst_vertices, dtype=np.float32) if dst_vertices else np.zeros((0, 2), dtype=np.float32)

    off_xn, off_yn = _clamp01_pair(getattr(rule, "offset_norm", (0.5, 0.5)))
    base_angle_deg = float(target.angle_deg)
    angle_rad = math.radians(base_angle_deg)
    dx_local = (off_xn - 0.5) * target.width
    dy_local = (off_yn - 0.5) * target.height
    dx_scene = math.cos(angle_rad) * dx_local - math.sin(angle_rad) * dy_local
    dy_scene = math.sin(angle_rad) * dx_local + math.cos(angle_rad) * dy_local
    dest_center = np.array([float(target.cx + dx_scene), float(target.cy + dy_scene)], dtype=np.float32)

    if dst_vertices_np.size == 0:
        dst_vertices_np = np.array([
            dest_center + [-0.5, -0.5],
            dest_center + [0.5, -0.5],
            dest_center + [0.5, 0.5],
            dest_center + [-0.5, 0.5],
        ], dtype=np.float32)

    center_current = dst_vertices_np.mean(axis=0)
    dst_vertices_np += (dest_center - center_current)

    rotation_offset_deg = float(getattr(rule, "rotation_offset_deg", 0.0))
    if rotation_offset_deg:
        rot_rad = math.radians(rotation_offset_deg)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        dst_vertices_np = ((dst_vertices_np - dest_center) @ rot_mat.T) + dest_center

    dst_vertices = [tuple(map(float, pt)) for pt in dst_vertices_np]

    matrix = None
    src_pts = None
    if src_vertices_scene and len(src_vertices_scene) >= 3:
        src_pts = np.array(src_vertices_scene[:3], dtype=np.float32)
    elif len(src_vertices) >= 3:
        src_pts = np.array(src_vertices[:3], dtype=np.float32)

    if src_pts is not None and len(dst_vertices) >= 3:
        dst_pts = np.array(dst_vertices[:3], dtype=np.float32)
        try:
            matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        except cv2.error:
            matrix = None

    tx = float(dest_center[0])
    ty = float(dest_center[1])
    rot = float(base_angle_deg + rotation_offset_deg)
    sx = sy = 1.0

    if matrix is None:
        # Fallback al comportamiento previo
        tx = float(dest_center[0])
        ty = float(dest_center[1])
    else:
        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])
        rot = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        sx = math.hypot(matrix[0, 0], matrix[1, 0])
        sy = math.hypot(matrix[0, 1], matrix[1, 1])

    matrix_tuple = _matrix_to_tuple(matrix) if matrix is not None else None

    return Placement(
        tx=tx,
        ty=ty,
        rotation_deg=rot,
        scale_x=sx,
        scale_y=sy,
        piv_x=float(dest_center[0]),
        piv_y=float(dest_center[1]),
        matrix=matrix_tuple,
    )


# =========================================================
# Utilidades geométricas de items
# =========================================================

def get_item_min_area_rect(
    item: ImageItem,
    threshold: int = 127,
    min_area: float = 100.0,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """Devuelve el rectángulo de área mínima del contenido del item en coordenadas de escena.

    Analiza el pixmap del item para estimar su orientación y centro geométrico.
    Si no se detectan contornos significativos (área < ``min_area``) devuelve ``None``.
    """

    pixmap = item.pixmap()
    if pixmap.isNull():
        return None

    image = pixmap.toImage()
    array = qimage_to_numpy(image)

    if array.ndim == 3:
        channels = array.shape[2]
        if channels >= 4:
            gray = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
        elif channels == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            gray = array[..., 0]
    else:
        gray = array.astype(np.uint8)

    gray = np.ascontiguousarray(gray)

    if threshold is None or threshold < 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < float(min_area):
        return None

    rect = cv2.minAreaRect(largest)
    center_local = QPointF(float(rect[0][0]), float(rect[0][1]))
    box = cv2.boxPoints(rect)

    scene_points = [item.mapToScene(QPointF(float(x), float(y))) for x, y in box]
    scene_center = item.mapToScene(center_local)

    def _dist(p1: QPointF, p2: QPointF) -> float:
        return math.hypot(p2.x() - p1.x(), p2.y() - p1.y())

    if len(scene_points) != 4:
        return (scene_center.x(), scene_center.y()), (0.0, 0.0), 0.0

    edge_lengths = [_dist(scene_points[i], scene_points[(i + 1) % 4]) for i in range(4)]
    if not edge_lengths:
        return (scene_center.x(), scene_center.y()), (0.0, 0.0), 0.0

    width_scene = 0.5 * (edge_lengths[0] + edge_lengths[2]) if len(edge_lengths) >= 3 else edge_lengths[0]
    height_scene = 0.5 * (edge_lengths[1] + edge_lengths[3]) if len(edge_lengths) >= 4 else edge_lengths[0]

    edge_vec_x = scene_points[1].x() - scene_points[0].x()
    edge_vec_y = scene_points[1].y() - scene_points[0].y()
    angle_scene = math.degrees(math.atan2(edge_vec_y, edge_vec_x))

    return (
        (scene_center.x(), scene_center.y()),
        (float(width_scene), float(height_scene)),
        float(angle_scene),
    )


# =========================================================
# Aplicar Placement a un ImageItem (centro en tx,ty)
# =========================================================
def apply_placement_to_item(
    item: ImageItem,
    placement: Placement,
) -> None:
    """Aplica el Placement calculado al item y sincroniza el Layer asociado."""

    layer = getattr(item, "layer", None)
    mm_to_scene = float(getattr(item, "mm_to_scene", 1.0) or 1.0)

    matrix_data = getattr(placement, "matrix", None)

    if matrix_data is not None:
        print("la matrix existe")
        matrix_np = np.array(matrix_data, dtype=np.float64)
        tx = float(matrix_np[0, 2])
        ty = float(matrix_np[1, 2])
        if layer is not None:
            print("el layer existe")
            layer.transform_matrix = _matrix_to_tuple(matrix_np)
            layer.x = tx / mm_to_scene
            layer.y = ty / mm_to_scene
            layer.rotation = float(placement.rotation_deg)
            print("placement_scale: ", placement.scale_x, placement.scale_y)
            #sx = float(placement.scale_x)
            #sy = float(placement.scale_y)
            sx = 1.0
            sy = 1.0
            if math.isfinite(sx) and math.isfinite(sy):
                if math.isclose(sx, sy, rel_tol=1e-6, abs_tol=1e-6):
                    layer.scale = sx
                else:
                    layer.scale = (sx + sy) * 0.5
            elif math.isfinite(sx):
                layer.scale = sx
            elif math.isfinite(sy):
                layer.scale = sy
        if hasattr(item, "sync_from_layer") and callable(item.sync_from_layer) and layer is not None:
            item.sync_from_layer()
        else:
            item.setTransformOriginPoint(QPointF(0.0, 0.0))
            item.setPos(0.0, 0.0)
            item.setTransform(_matrix_to_qtransform(matrix_np), False)
    else:
        if layer is not None:
            layer.transform_matrix = None
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

        if hasattr(item, "sync_from_layer") and callable(item.sync_from_layer) and layer is not None:
            item.sync_from_layer()
        else:
            item.setTransformOriginPoint(QPointF(0.0, 0.0))
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

        scene.addItem(item)
        if bind_callback:
            bind_callback(item)

        created.append(item)

    return created
