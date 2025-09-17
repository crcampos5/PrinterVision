"""Image document model handling the current image state."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dataclasses import dataclass
from editor_tif.processing.detection import Centroid, detect_centroids, draw_centroids_overlay
from editor_tif.processing.placement import place_tile_on_centroids
from editor_tif.utils.io import ImageData, load_image_data, save_image_tif
from editor_tif.views.scene_items import Layer


class ImageDocument:
    """Encapsulate image data along with metadata and derived products."""

    def __init__(self, min_area: int = 50, workspace_width_mm: float = 480.0, workspace_height_mm: float = 600.0) -> None:

        self.layers: list[Layer] = []
        self._layer_seq: int = 0

        self.min_area = min_area
        self.workspace_width_mm = workspace_width_mm
        self.workspace_height_mm = workspace_height_mm
        self.tile_photometric: Optional[str] = None
        self.tile_cmyk_order: Optional[Tuple[int,int,int,int]] = None
        self.tile_alpha_index: Optional[int] = None
        self.tile_icc_profile: Optional[bytes] = None
        self.tile_ink_names: Optional[list[str]] = None

        # Reference image and detection outputs.
        self.reference_path: Optional[Path] = None
        self.reference_image: Optional[np.ndarray] = None
        self.reference_overlay: Optional[np.ndarray] = None
        self.centroids: List[Centroid] = []

        # Mosaic tile data and physical dimensions.
        self.tile_path: Optional[Path] = None
        self.tile_image: Optional[np.ndarray] = None  # stored as HxWxC (C=1 or 3)
        self.tile_mm_width: Optional[float] = None
        self.tile_mm_height: Optional[float] = None

        # Resulting canvas composed only of mosaics.
        self.output_image: Optional[np.ndarray] = None  # canvas same size as reference

        # Cached conversion factors between pixels and millimeters.
        self.mm_per_pixel_x: Optional[float] = None
        self.mm_per_pixel_y: Optional[float] = None

    @property
    def has_reference(self) -> bool:
        return self.reference_image is not None

    @property
    def has_output(self) -> bool:
        return self.output_image is not None

    def load_reference(self, path: Path) -> bool:
        """Load a reference image and detect centroids."""
        data = load_image_data(path)
        if data is None:
            return False
        image = data.pixels
        _, centroids = detect_centroids(image, self.min_area)
        if not centroids:
            return False
        self.reference_path = path
        self.reference_image = image
        self.centroids = centroids
        self.reference_overlay = draw_centroids_overlay(image, centroids)
        self.tile_path = None
        self.tile_image = None  # stored as HxWxC (C=1 or 3)
        self.tile_mm_width = None
        self.tile_mm_height = None
        self.output_image = None  # canvas same size as reference
        self._recompute_mm_per_pixel()
        return True

    def load_tile(self, path: Path) -> bool:
        """Load a TIF tile and build the placement result respecting physical size."""
        if self.reference_image is None or not self.centroids:
            return False
        data = load_image_data(path)
        if data is None:
            return False
        self.tile_path = path
        self.tile_image = data.pixels
        self.tile_mm_width = data.width_mm
        self.tile_mm_height = data.height_mm
        self.tile_photometric = data.photometric
        self.tile_cmyk_order = data.cmyk_order
        self.tile_alpha_index = data.alpha_index
        self.tile_icc_profile = getattr(data, "icc_profile", None)
        self.tile_ink_names = getattr(data, "ink_names", None)
        # Fall back to pixel-derived size if metadata is missing.
        if self.tile_mm_width is None and self.mm_per_pixel_x is not None:
            self.tile_mm_width = self.tile_image.shape[1] * self.mm_per_pixel_x
        if self.tile_mm_height is None and self.mm_per_pixel_y is not None:
            self.tile_mm_height = self.tile_image.shape[0] * self.mm_per_pixel_y
        return self._generate_output()

    def rebuild_output(self) -> bool:
        """Recalculate the output using current settings."""
        return self._generate_output()

    def update_workspace(self, width_mm: float, height_mm: float) -> None:
        """Update workspace dimensions and regenerate output if needed."""
        self.workspace_width_mm = width_mm
        self.workspace_height_mm = height_mm
        self._recompute_mm_per_pixel()
        if self.tile_image is not None:
            self._generate_output()

    def get_reference_preview(self) -> Optional[np.ndarray]:
        """Return the reference overlay for display."""
        return self.reference_overlay

    def get_output_preview(self) -> Optional[np.ndarray]:
        """Return the generated output image."""
        return self.output_image

    def get_mm_per_pixel(self) -> Optional[Tuple[float, float]]:
        if self.mm_per_pixel_x is None or self.mm_per_pixel_y is None:
            return None
        return self.mm_per_pixel_x, self.mm_per_pixel_y

    def get_tile_dimensions_mm(self) -> Optional[Tuple[float, float]]:
        if self.tile_mm_width is None or self.tile_mm_height is None:
            return None
        return self.tile_mm_width, self.tile_mm_height

    def save_output(self, path: Path) -> bool:
        if self.layers:
            img = self.rasterize_layers()
        else:
            if self.output_image is None:
                return False
            img = self.output_image

        # DPI finales (tile fijo)
        mmpp_x, mmpp_y = self._target_mm_per_px()
        if mmpp_x is None: mmpp_x = self.mm_per_pixel_x
        if mmpp_y is None: mmpp_y = self.mm_per_pixel_y
        dpi_x = (25.4 / mmpp_x) if mmpp_x else None
        dpi_y = (25.4 / mmpp_y) if mmpp_y else None


        # Photometric según canales
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            photometric = "minisblack"
        elif img.ndim == 3 and img.shape[2] == 3:
            photometric = "rgb"
        elif img.ndim == 3 and img.shape[2] >= 4:
            photometric = "separated"  # CMYK (+ posibles spots)
        else:
            photometric = None

        # Metadatos heredados del tile
        icc = getattr(self, "tile_icc_profile", None)
        ink_names = getattr(self, "tile_ink_names", None)

        # Decidir si hay ALFA o si el 5º canal es SPOT:
        extrasamples = None
        number_of_inks = None
        inkset = None  # 1 = CMYK

        channels = img.shape[2] if (img.ndim == 3) else 1
        if photometric == "separated":
            # Si el tile traía alfa y sigue estando al final -> marcar ExtraSamples=ALPHA
            if self.tile_alpha_index is not None and channels == (self.tile_alpha_index + 1):
                extrasamples = [2]  # 2 = Unassociated Alpha
                # Si hay nombres de tintas y además alfa, no cuentes el alfa como 'ink'
                if ink_names:
                    ink_names = [n for i, n in enumerate(ink_names) if i != self.tile_alpha_index]
            else:
                # No hay alfa: si hay nombres de tintas (p.ej. CMYK+Spot), declara el número de tintas
                if ink_names and len(ink_names) == channels:
                    number_of_inks = channels
                    inkset = 1  # CMYK base

        return save_image_tif(
            path,
            img,
            photometric=photometric,
            dpi_x=dpi_x,
            dpi_y=dpi_y,
            icc_profile=icc,
            ink_names=ink_names,
            extrasamples=extrasamples,
            number_of_inks=number_of_inks,
            inkset=inkset,
        )




    def _recompute_mm_per_pixel(self) -> None:
        if self.reference_image is None:
            self.mm_per_pixel_x = None
            self.mm_per_pixel_y = None
            return
        height_px, width_px = self.reference_image.shape[:2]
        self.mm_per_pixel_x = self.workspace_width_mm / width_px if width_px else None
        self.mm_per_pixel_y = self.workspace_height_mm / height_px if height_px else None

    def _scaled_tile(self) -> Optional[np.ndarray]:
        """En modo tile fijo, devuelve el tile sin redimensionar."""
        if self.tile_image is None:
            return None
        # Aquí podrías adaptar a ≤4 canales si hace falta para OpenCV,
        # pero NO cambies tamaño.
        return self.tile_image
    
    def _target_mm_per_px(self) -> Tuple[Optional[float], Optional[float]]:
        """
        mm/px objetivo para el canvas derivado del TILE (tile fijo).
        Si el TIF trae dimensiones físicas, úsalo; de lo contrario, cae al mm/px de la referencia.
        """
        if self.tile_image is None:
            return None, None
        h, w = self.tile_image.shape[:2]
        mmpp_x = (self.tile_mm_width / float(w)) if (self.tile_mm_width and w) else self.mm_per_pixel_x
        mmpp_y = (self.tile_mm_height / float(h)) if (self.tile_mm_height and h) else self.mm_per_pixel_y
        return mmpp_x, mmpp_y


    def _blank_canvas(self) -> np.ndarray:
        """
        Lienzo en blanco cuyo tamaño en píxeles representa el workspace físico
        (480×600 mm por defecto) a la resolución (mm/px) derivada del TILE,
        sin re-muestrear el tile.
        """
        if self.tile_image is None:
            raise RuntimeError("Tile image required")

        mmpp_x, mmpp_y = self._target_mm_per_px()
        if mmpp_x is None or mmpp_y is None:
            raise RuntimeError("Cannot determine target mm/px for canvas")

        width_px  = max(1, int(round(self.workspace_width_mm  / mmpp_x)))
        height_px = max(1, int(round(self.workspace_height_mm / mmpp_y)))

        # Heredar dtype y canales del TILE (evita conversiones posteriores)
        dtype = self.tile_image.dtype
        channels = self.tile_image.shape[2] if self.tile_image.ndim == 3 else 1

        is_int = np.issubdtype(dtype, np.integer)
        maxv = np.iinfo(dtype).max if is_int else 1.0

        # Blanco lógico: CMYK (separated/cmyk) = 0; RGB/GRAY = max
        tile_ph = (self.tile_photometric or "").lower()
        white_val = 0 if tile_ph in ("separated", "cmyk") else maxv

        if channels == 1:
            return np.full((height_px, width_px), white_val, dtype=dtype)
        return np.full((height_px, width_px, channels), white_val, dtype=dtype)

    

    def _generate_output(self) -> bool:
        """
        Genera el compuesto en modo 'tile fijo':
        - Canvas a la resolución física del tile.
        - Re-escala centroides desde píxeles de referencia a píxeles del canvas.
        - NO re-muestrea el tile (se usa tal cual).
        """
        if self.reference_image is None or self.tile_image is None or not self.centroids:
            return False
        if self.mm_per_pixel_x is None or self.mm_per_pixel_y is None:
            return False

        mmpp_x_target, mmpp_y_target = self._target_mm_per_px()
        if mmpp_x_target is None or mmpp_y_target is None:
            return False

        # Escala de traslado: px_ref -> px_canvas
        sx = self.mm_per_pixel_x / mmpp_x_target
        sy = self.mm_per_pixel_y / mmpp_y_target

        # Centroides re-escalados (place_tile_on_centroids acepta floats/ints)
        scaled_centroids = [(x * sx, y * sy) for (x, y) in self.centroids]

        canvas = self._blank_canvas()
        tile = self.tile_image  # tal cual, sin resize

        self.output_image = place_tile_on_centroids(canvas, tile, scaled_centroids)
        return True
    
    def add_layer_from_tile(self) -> Layer | None:
        """Crea un Layer desde el TIF cargado (tile_image) y lo devuelve."""
        if self.tile_image is None:
            return None
        self._layer_seq += 1
        layer = Layer(
            id=self._layer_seq,
            path=self.tile_path,
            pixels=self.tile_image,
            photometric=self.tile_photometric,
            cmyk_order=self.tile_cmyk_order,
            alpha_index=self.tile_alpha_index,
            icc_profile=self.tile_icc_profile,
            ink_names=self.tile_ink_names,
            x=0.0, y=0.0, rotation=0.0, scale=1.0, opacity=1.0,
        )
        self.layers.append(layer)
        return layer

    def add_layer_from_jpg(self, img: np.ndarray, path: Path | None = None) -> Layer:
        self._layer_seq += 1
        layer = Layer(id=self._layer_seq, path=path, pixels=img, x=0.0, y=0.0, rotation=0.0, scale=1.0, opacity=1.0)
        self.layers.append(layer)
        return layer

    def clear_layers(self) -> None:
        self.layers.clear()

    def rasterize_layers(self) -> np.ndarray:
        """Compone todas las capas del documento en un canvas blanco (usa photometric del TIF si hay)."""
        # base: canvas del tamaño del workspace (en px) calculado con _target_mm_per_px()
        if self.tile_image is None and self.reference_image is None:
            raise RuntimeError("No hay referencia ni tile para deducir resolución")
        mmpp_x, mmpp_y = self._target_mm_per_px()
        if mmpp_x is None or mmpp_y is None:
            # cae a mm/px de la referencia
            mmpp_x, mmpp_y = self.mm_per_pixel_x, self.mm_per_pixel_y
        if mmpp_x is None or mmpp_y is None:
            raise RuntimeError("No se pudo determinar mm/px")

        width_px  = max(1, int(round(self.workspace_width_mm  / mmpp_x)))
        height_px = max(1, int(round(self.workspace_height_mm / mmpp_y)))

        # Determinar dtype/photometric base; si hay algún layer CMYK usa blanco=0
        dtype = self.layers[0].pixels.dtype if self.layers else np.uint8
        channels = self.layers[0].pixels.shape[2] if (self.layers and self.layers[0].pixels.ndim == 3) else 3
        ph = (self.tile_photometric or "").lower()
        white = 0 if ph in ("separated","cmyk") else (np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0)

        canvas = np.full((height_px, width_px, channels), white, dtype=dtype) if channels > 1 else np.full((height_px, width_px), white, dtype=dtype)

        # Pintar cada layer aplicando transform (versión simple: rotación/escala nearest + sobreescritura por >0)
        for layer in self.layers:
            img = layer.pixels
            # aplicar escala/rotación con OpenCV (mantén dtype/canales)
            Mscale = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), layer.rotation, layer.scale)
            warped = cv2.warpAffine(img, Mscale, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST, borderValue=white)
            # ubicación
            x0 = int(round(layer.x)); y0 = int(round(layer.y))
            x1 = min(x0 + warped.shape[1], canvas.shape[1]); y1 = min(y0 + warped.shape[0], canvas.shape[0])
            if x1 <= 0 or y1 <= 0: continue
            xs = max(0, -x0); ys = max(0, -y0)
            x0 = max(0, x0); y0 = max(0, y0)
            tile_slice = warped[ys:y1-y0+ys, xs:x1-x0+xs]

            # mezcla simple: copia de valores >0 (igual que placement.py)
            region = canvas[y0:y1, x0:x1]
            mask = tile_slice > 0
            region[mask] = tile_slice[mask]
            canvas[y0:y1, x0:x1] = region

        self.output_image = canvas
        return canvas


