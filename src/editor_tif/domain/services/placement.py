"""Placement helpers for positioning the TIF mosaic."""

from __future__ import annotations

from typing import Sequence, Tuple, List, Callable, Optional
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene
from editor_tif.presentation.views.scene_items import CentroidItem, ImageItem, Layer

import numpy as np

Centroid = Tuple[float, float]


def place_tile_on_centroids(canvas: np.ndarray, tile: np.ndarray, centroids: Sequence[Centroid]) -> np.ndarray:
    """Overlay the tile image centered on each centroid over a canvas."""
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
        if result.ndim == 2:
            mask = tile_slice > 0
            region[mask] = tile_slice[mask]
        else:
            mask = tile_slice > 0
            region[mask] = tile_slice[mask]
        result[y0:y1, x0:x1] = region
    return result

def _clone_from_item(src: QGraphicsItem) -> QGraphicsItem:
    # Ideal: tu ImageItem exponga .clone()
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
    """Clona un ImageItem en cada CentroidItem de la escena creando un Layer por clon."""
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
            pixels=src_layer.pixels,                 # ok compartir píxeles; lo editable son las transformaciones
            photometric=src_layer.photometric,
            cmyk_order=src_layer.cmyk_order,
            alpha_index=src_layer.alpha_index,
            icc_profile=src_layer.icc_profile,
            ink_names=src_layer.ink_names,
            width_mm=src_layer.width_mm,
            height_mm=src_layer.height_mm,
            # Copia transform del source; la posición la definimos por el centroide:
            x=src_layer.x,
            y=src_layer.y,
            rotation=src_layer.rotation,
            scale=src_layer.scale,
            opacity=src_layer.opacity,
        )
        document.layers.append(layer)

        # 2) nuevo ImageItem ligado a ese Layer
        item = ImageItem(layer, document=document, mm_to_scene=mm_to_scene)

        # Alinear por centro (por si en algún sitio no quedó aplicado el offset)
        br = item.boundingRect()
        item.setTransformOriginPoint(br.center())
        item.setOffset(-br.width() / 2.0, -br.height() / 2.0)

        # 3) posicionar en el centroide
        item.setPos(c.center_scene_pos())
        # sincronizar layer.x/y en mm desde la posición (por si el setPos cambió algo):
        layer.x = item.pos().x() / mm_to_scene
        layer.y = item.pos().y() / mm_to_scene

        scene.addItem(item)
        if bind_callback:
            bind_callback(item)  # para SelectionHandler.bind_item

        created.append(item)

    return created