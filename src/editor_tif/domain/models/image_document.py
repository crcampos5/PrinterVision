"""Image document model: reference (px), physical scale (mm/px), detections (px), tile and output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Detecciones en píxeles (Reference Pixel Space)
from editor_tif.domain.models.contours import Contour, Centroid
# Servicio de detección (OpenCV)
from editor_tif.infrastructure.contour_detector import ContourDetector
# Colocación rápida por centroides en raster
from editor_tif.domain.services.placement import place_tile_on_centroids
# I/O de imágenes con metadatos
from editor_tif.infrastructure.tif_io import load_image_data, save_image_tif
# Capas para el viewer (edición)
from editor_tif.presentation.views.scene_items import Layer


# ------------------------- helpers internos -------------------------
def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    a = img.astype(np.float32)
    a = a - a.min()
    m = a.max() if a.max() else 1.0
    return (a / m * 255.0).clip(0, 255).astype(np.uint8)


def _centroid_xy(c: Centroid | Tuple[float, float]) -> Tuple[float, float]:
    if hasattr(c, "x") and hasattr(c, "y"):
        return float(c.x), float(c.y)
    return float(c[0]), float(c[1])


# ============================= DOCUMENTO =============================
@dataclass
class ImageDocument:
    """
    Fuente de verdad:
      - Referencia original (np.ndarray) y mm/px del workspace.
      - Detecciones SOLO en píxeles de la referencia (centroids_px, contours_px).
      - Tile (TIF) con metadatos físicos (mm).
      - Capas editables y/o salida raster compuesta.
    """

    # Workspace físico (mm)
    workspace_width_mm: float = 480.0
    workspace_height_mm: float = 600.0

    # Umbral de detección
    min_area: float = 50.0

    # ---------------- Estado de edición / capas ----------------
    layers: list[Layer] = field(default_factory=list)
    _layer_seq: int = 0

    # ---------------- Referencia / detecciones -----------------
    reference_path: Optional[Path] = None
    _reference_np: Optional[np.ndarray] = None
    reference_overlay: Optional[np.ndarray] = None  # solo para preview

    # Detecciones en píxeles (Reference Pixel Space)
    contours_px: List[Contour] = field(default_factory=list)
    centroids_px: List[Centroid] = field(default_factory=list)

    # ---------------- Tile y metadatos físicos -----------------
    tile_path: Optional[Path] = None
    tile_image: Optional[np.ndarray] = None
    tile_mm_width: Optional[float] = None
    tile_mm_height: Optional[float] = None

    tile_photometric: Optional[str] = None
    tile_cmyk_order: Optional[Tuple[int, int, int, int]] = None
    tile_alpha_index: Optional[int] = None
    tile_icc_profile: Optional[bytes] = None
    tile_ink_names: Optional[list[str]] = None

    # ---------------- Salida compuesta -------------------------
    output_image: Optional[np.ndarray] = None

    # ---------------- Escala física de la referencia -----------
    mm_per_pixel_x: Optional[float] = None
    mm_per_pixel_y: Optional[float] = None

    # ============================ PROPIEDADES ============================
    @property
    def has_reference(self) -> bool:
        return self._reference_np is not None

    @property
    def has_output(self) -> bool:
        return self.output_image is not None

    # ======================== CARGA / DETECCIÓN ==========================
    def load_reference(self, path: Path, *, detector: ContourDetector | None = None) -> bool:
        """
        Carga la imagen de referencia (JPG/PNG/TIF) y (opcionalmente) corre
        detección de contornos/centroides. Detecciones se guardan en PX.
        """
        data = load_image_data(path)
        if data is None or data.pixels is None or data.pixels.size == 0:
            return False

        self.reference_path = path
        self._reference_np = data.pixels
        self._recompute_mm_per_pixel()

        # Si se provee o construye detector → popular detecciones en px
        if detector is None:
            detector = ContourDetector(min_area=self.min_area)

        if self._reference_np is not None:
            try:
                contours, centroids = detector.detect(self._reference_np)
            except Exception:
                contours, centroids = [], []
            self.contours_px = list(contours)
            self.centroids_px = list(centroids)
            self.reference_overlay = self._build_overlay(self._reference_np, self.contours_px, self.centroids_px)
        else:
            self.contours_px = []
            self.centroids_px = []
            self.reference_overlay = None

        # Reset de tile / salida (nueva referencia invalida anteriores)
        self.tile_path = None
        self.tile_image = None
        self.tile_mm_width = None
        self.tile_mm_height = None
        self.tile_photometric = None
        self.tile_cmyk_order = None
        self.tile_alpha_index = None
        self.tile_icc_profile = None
        self.tile_ink_names = None
        self.output_image = None

        return True

    def detect_with(self, detector: ContourDetector) -> bool:
        """Permite refrescar detecciones en PX con un detector externo."""
        if self._reference_np is None:
            return False
        try:
            contours, centroids = detector.detect(self._reference_np)
        except Exception:
            return False

        self.contours_px = list(contours or [])
        self.centroids_px = list(centroids or [])
        self.reference_overlay = self._build_overlay(self._reference_np, self.contours_px, self.centroids_px)
        return bool(self.contours_px or self.centroids_px)


    def update_workspace(self, width_mm: float, height_mm: float) -> None:
        """Actualiza dimensiones físicas del workspace y regenera si procede."""
        self.workspace_width_mm = float(width_mm)
        self.workspace_height_mm = float(height_mm)
        self._recompute_mm_per_pixel()
        if self.tile_image is not None:
            self._generate_output()

    # =============================== GETTERS ================================
    def get_reference_preview(self) -> Optional[np.ndarray]:
        """Devuelve la referencia con overlay (si existe). Para UI; no posiciona."""
        return self.reference_overlay

    def get_output_preview(self) -> Optional[np.ndarray]:
        return self.output_image

    def get_reference_image_np(self) -> Optional[np.ndarray]:
        return self._reference_np

    def get_mm_per_pixel(self) -> Optional[Tuple[float, float]]:
        if self.mm_per_pixel_x is None or self.mm_per_pixel_y is None:
            return None
        return self.mm_per_pixel_x, self.mm_per_pixel_y

    def get_tile_dimensions_mm(self) -> Optional[Tuple[float, float]]:
        if self.tile_mm_width is None or self.tile_mm_height is None:
            return None
        return self.tile_mm_width, self.tile_mm_height

    # =============================== GUARDADO ===============================
    def save_output(self, path: Path) -> bool:
        """
        Guarda el resultado (rasterizando capas si existen) con metadatos del tile.
        """
        if self.layers:
            img = self.rasterize_layers()
        else:
            if self.output_image is None:
                return False
            img = self.output_image

        # DPI finales basados en mm/px objetivo (tile fijo)
        mmpp_x, mmpp_y = self._target_mm_per_px()
        if mmpp_x is None:
            mmpp_x = self.mm_per_pixel_x
        if mmpp_y is None:
            mmpp_y = self.mm_per_pixel_y
        dpi_x = (25.4 / mmpp_x) if mmpp_x else None
        dpi_y = (25.4 / mmpp_y) if mmpp_y else None

       

        # Photometric por canales
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            photometric = "minisblack"
        elif img.ndim == 3 and img.shape[2] == 3:
            photometric = "rgb"
        elif img.ndim == 3 and img.shape[2] >= 4:
            photometric = "separated"  # CMYK (+ spots)
        else:
            photometric = None

        icc = getattr(self, "tile_icc_profile", None)
        ink_names = getattr(self, "tile_ink_names", None)

        # ExtraSamples/InkSet
        extrasamples = None
        number_of_inks = None
        inkset = None  # 1 = CMYK

        channels = img.shape[2] if (img.ndim == 3) else 1
        if photometric == "separated":
            if self.tile_alpha_index is not None and channels == (self.tile_alpha_index + 1):
                extrasamples = [2]  # 2 = Unassociated Alpha
                if ink_names:
                    ink_names = [n for i, n in enumerate(ink_names) if i != self.tile_alpha_index]
            else:
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

    # =============================== INTERNOS ===============================
    def _recompute_mm_per_pixel(self) -> None:
        """mm/px de la referencia en función del workspace actual."""
        if self._reference_np is None:
            self.mm_per_pixel_x = None
            self.mm_per_pixel_y = None
            return
        h_px, w_px = self._reference_np.shape[:2]
        self.mm_per_pixel_x = (self.workspace_width_mm / w_px) if w_px else None
        self.mm_per_pixel_y = (self.workspace_height_mm / h_px) if h_px else None

    def _target_mm_per_px(self) -> Tuple[Optional[float], Optional[float]]:
        """
        mm/px objetivo del canvas (tile fijo). Usa dimensiones físicas del tile si existen,
        o cae a la escala de la referencia.
        """
        if self.tile_image is None:
            print("es none")
            return None, None
        h, w = self.tile_image.shape[:2]
        mmpp_x = (self.tile_mm_width / float(w)) if (self.tile_mm_width and w) else self.mm_per_pixel_x
        mmpp_y = (self.tile_mm_height / float(h)) if (self.tile_mm_height and h) else self.mm_per_pixel_y
        return mmpp_x, mmpp_y

    def _blank_canvas(self) -> np.ndarray:
        """Canvas blanco del tamaño físico del workspace a la resolución objetivo."""
        if self.tile_image is None:
            raise RuntimeError("Tile image required")

        mmpp_x, mmpp_y = self._target_mm_per_px()
        if mmpp_x is None or mmpp_y is None:
            raise RuntimeError("Cannot determine target mm/px for canvas")

        width_px = max(1, int(round(self.workspace_width_mm / mmpp_x)))
        height_px = max(1, int(round(self.workspace_height_mm / mmpp_y)))

        dtype = self.tile_image.dtype
        channels = self.tile_image.shape[2] if self.tile_image.ndim == 3 else 1

        is_int = np.issubdtype(dtype, np.integer)
        maxv = np.iinfo(dtype).max if is_int else 1.0

        tile_ph = (self.tile_photometric or "").lower()
        #white_val = 0 if tile_ph in ("separated", "cmyk") else maxv
        white_val = maxv

        if channels == 1:
            return np.full((height_px, width_px), white_val, dtype=dtype)
        return np.full((height_px, width_px, channels), white_val, dtype=dtype)

    def _generate_output(self) -> bool:
        """
        Compuesto rápido (tile fijo):
          - Canvas a la resolución física objetivo (tile).
          - Traslada centroides PX_ref → PX_canvas usando mm/px.
          - Usa el tile tal cual (sin resize).
        """
        if self._reference_np is None or self.tile_image is None or not self.centroids_px:
            return False
        if self.mm_per_pixel_x is None or self.mm_per_pixel_y is None:
            return False

        mmpp_x_target, mmpp_y_target = self._target_mm_per_px()
        if mmpp_x_target is None or mmpp_y_target is None:
            return False

        # Factor px_ref → px_canvas
        sx = self.mm_per_pixel_x / mmpp_x_target
        sy = self.mm_per_pixel_y / mmpp_y_target

        # Centroides reescalados (enteros para raster)
        scaled_centroids: list[Tuple[int, int]] = []
        for c in self.centroids_px:
            x, y = _centroid_xy(c)
            scaled_centroids.append((int(round(x * sx)), int(round(y * sy))))

        canvas = self._blank_canvas()
        tile = self.tile_image  # sin remuestrear
        self.output_image = place_tile_on_centroids(canvas, tile, scaled_centroids)
        return True

    # ============================== CAPAS (VIEWER) ============================
    def add_layer_from_tile(self) -> Layer | None:
        """Crea una Layer desde el TIF cargado (tile_image) y la devuelve."""
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
            width_mm=self.tile_mm_width,
            height_mm=self.tile_mm_height,
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
        """
        Compone capas sobre canvas blanco (white=0), centrando cada layer en (layer.x, layer.y).
        - Independiente de tile_*.
        - Respeta width_mm/height_mm -> escala física.
        - Warp affine expandido (sin recortes por rotación/escala).
        - Alpha opcional como máscara (>0); sin alpha: copia donde ≠ white.
        """
        if self._reference_np is None and not self.layers:
            raise RuntimeError("No hay referencia ni capas para deducir resolución")

        # mm/px del canvas desde la referencia/workspace
        mmpp_x, mmpp_y = self.mm_per_pixel_x, self.mm_per_pixel_y
        if mmpp_x is None or mmpp_y is None:
            raise RuntimeError("No se pudo determinar mm/px desde la referencia/workspace")

        width_px  = max(1, int(round(self.workspace_width_mm  / mmpp_x)))
        height_px = max(1, int(round(self.workspace_height_mm / mmpp_y)))

        # Canvas: dtype/canales de la primera capa
        base = self.layers[0].pixels
        dtype = base.dtype
        channels = (base.shape[2] if base.ndim == 3 else 1)

        white = 0  # en tu pipeline, "blanco" es 0
        canvas = (np.full((height_px, width_px, channels), white, dtype=dtype)
                if channels > 1 else np.full((height_px, width_px), white, dtype=dtype))

        for layer in self.layers:
            img = layer.pixels
            # Normaliza a (H,W,C)
            if img.ndim == 2:
                img = img[..., np.newaxis]

            # Ajusta canales para compatibilidad con canvas (permitimos 1 canal extra como alpha)
            ch = img.shape[2]
            if ch < channels:
                img = np.concatenate([img, np.repeat(img[..., -1:], channels - ch, axis=2)], axis=2)
                ch = channels
            if ch > channels + 1:
                img = img[..., :channels + 1]
                ch = img.shape[2]

            h, w = img.shape[:2]

            # Escala física: mm -> px del canvas
            s_phys = 1.0
            if self.mm_per_pixel_x and self.mm_per_pixel_y:
                if getattr(layer, "width_mm", None):
                    target_w_px = float(layer.width_mm) / float(mmpp_x)
                    s_phys = target_w_px / float(w)
                elif getattr(layer, "height_mm", None):
                    target_h_px = float(layer.height_mm) / float(mmpp_y)
                    s_phys = target_h_px / float(h)

            eff_scale = float(layer.scale) * float(s_phys)

            # Rotación + escala alrededor del centro, con expansión de tamaño para no recortar
            cx, cy = (w / 2.0, h / 2.0)
            print("rotacion: ", layer.rotation)
            M = cv2.getRotationMatrix2D((cx, cy), layer.rotation, eff_scale)

            # Calcular tamaño expandido
            cos = abs(M[0, 0]); sin = abs(M[0, 1])
            new_w = int(round((h * sin) + (w * cos)))
            new_h = int(round((h * cos) + (w * sin)))

            # Ajustar traslación para centrar en el nuevo lienzo
            M[0, 2] += (new_w / 2.0) - cx
            M[1, 2] += (new_h / 2.0) - cy

            # borderValue: escalar si hay >4 canales (limitación OpenCV)
            if ch <= 4:
                border_val = (white,) * ch
            else:
                border_val = white

            warped = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=border_val)

            # Colocación centrada en (layer.x, layer.y)
            x_px = layer.x / mmpp_x
            y_px = layer.y / mmpp_y
            x0 = int(round(x_px - new_w / 2.0))
            y0 = int(round(y_px - new_h / 2.0))

            # Recortes contra canvas
            if x0 >= canvas.shape[1] or y0 >= canvas.shape[0] or (x0 + new_w <= 0) or (y0 + new_h <= 0):
                continue

            xs = max(0, -x0); ys = max(0, -y0)
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(x0 + new_w - xs, canvas.shape[1])
            y1 = min(y0 + new_h - ys, canvas.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue

            tile_slice = warped[ys:ys + (y1 - y0), xs:xs + (x1 - x0)]
            region = canvas[y0:y1, x0:x1]

            # Máscara: alpha extra si canales = canvas+1, si no ≠ white
            has_extra_alpha = (tile_slice.ndim == 3 and tile_slice.shape[2] == (channels + 1))
            if has_extra_alpha:
                alpha = tile_slice[..., -1]
                color = tile_slice[..., :channels]
                mask = (alpha > 0)
                region[mask] = color[mask]
            else:
                if tile_slice.ndim == 3:
                    # Ajuste de canales por seguridad
                    if tile_slice.shape[2] > channels:
                        tile_slice = tile_slice[..., :channels]
                    elif tile_slice.shape[2] < channels:
                        tile_slice = np.concatenate(
                            [tile_slice, np.repeat(tile_slice[..., -1:], channels - tile_slice.shape[2], axis=2)], axis=2
                        )
                    mask = np.any(tile_slice != white, axis=2)
                    region[mask] = tile_slice[mask]
                else:
                    mask = (tile_slice != white)
                    region[mask] = tile_slice[mask]

            canvas[y0:y1, x0:x1] = region

        self.output_image = canvas
        return canvas



    # ========================= OVERLAY (opcional/UI) =========================
    def _build_overlay(
        self,
        ref: np.ndarray,
        contours: List[Contour],
        centroids: List[Centroid],
    ) -> np.ndarray:
        """
        Dibuja centroides (verde) y contornos (rojo) sobre la referencia para preview.
        Esto NO se usa para posicionar ítems en escena.
        """
        base = _ensure_uint8(ref)
        if base.ndim == 2:
            vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        elif base.shape[2] >= 3:
            vis = base[..., :3].copy()
        else:
            vis = base.copy()

        return vis
