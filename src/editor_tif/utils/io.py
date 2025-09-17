"""File-system helpers for image handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile

from .image import normalize_to_uint8


@dataclass
class ImageData:
    """Container for pixel data and basic physical metadata."""

    pixels: np.ndarray
    dpi_x: Optional[float] = None
    dpi_y: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    photometric: Optional[str] = None          # 'rgb', 'minisblack', 'separated', etc.
    cmyk_order: Optional[tuple[int,int,int,int]] = None  # índices (C,M,Y,K) dentro de pixels
    alpha_index: Optional[int] = None          # índice de alfa si existe (extrasample)
    icc_profile: Optional[bytes] = None          # <--- NUEVO
    ink_names: Optional[list[str]] = None


def _rational_to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            num, den = value
            den = float(den)
            if den == 0:
                return None
            return float(num) / den
        return float(value)
    except Exception:
        return None


def _apply_resolution_unit(resolution: Optional[float], unit_code: Optional[int]) -> Optional[float]:
    """Convert raw resolution to dots-per-inch taking the unit into account."""
    if resolution is None or resolution <= 0:
        return None
    if unit_code == 2:  # inch
        return resolution
    if unit_code == 3:  # centimeter
        return resolution * 2.54
    if unit_code == 1:  # no units
        return None
    return None


def _compute_size_mm(shape: tuple[int, ...], dpi_x: Optional[float], dpi_y: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    height_px = shape[0]
    width_px = shape[1]
    width_mm = (width_px / dpi_x * 25.4) if dpi_x else None
    height_mm = (height_px / dpi_y * 25.4) if dpi_y else None
    return width_mm, height_mm


def load_image_data(path: Path) -> Optional[ImageData]:
    """Carga una imagen devolviendo pixeles y metadatos sin alterar dtype ni canales."""
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    dpi_x: Optional[float] = None
    dpi_y: Optional[float] = None

    try:
        if suffix == ".tif":
            # Leer el primer page del TIF sin modificar dtype/canales
            with tifffile.TiffFile(str(path)) as tif:
                page = tif.pages[0]
                image = page.asarray()  # <- sin normalizar ni reducir canales
                tags = page.tags
                # photometric (p.ej. 'SEPARATED' para CMYK)
                photometric = (page.photometric.name.lower() if page.photometric else None)
                # posibles extrasamples (alfa no asociado, etc.)
                extras = list(page.extrasamples) if page.extrasamples is not None else []
                alpha_index = None
                # ICC profile (tag 34675)
                icc = None
                icc_tag = tags.get("ICCProfile")
                if icc_tag is not None:
                    icc = icc_tag.value  # bytes
                # Ink names/orden de tintas (para separar CMYK y spot)
                inknames_tag = tags.get("InkNames")
                cmyk_order = None
                if photometric == "separated":
                    # Intentar derivar orden real de canales a partir de InkNames
                    # InkNames viene como bytes separados por \x00 en TIFF
                    names = None
                    if inknames_tag is not None:
                        raw = inknames_tag.value
                        if isinstance(raw, (bytes, bytearray)):
                            names = [n for n in raw.decode("latin1").split("\x00") if n]
                        elif isinstance(raw, str):
                            names = [n for n in raw.split("\x00") if n]
                    # Buscar C,M,Y,K en el orden correcto
                    if names:
                        def idx_of(tgt):
                            for i, n in enumerate(names):
                                nn = n.strip().lower()
                                if tgt in nn:  # 'cyan','magenta','yellow','black/black (K)'
                                    return i
                            return None
                        iC, iM, iY = idx_of("cyan"), idx_of("magenta"), idx_of("yellow")
                        # 'black' o 'key'
                        iK = idx_of("black")
                        if iK is None:
                            iK = idx_of("key")
                        if None not in (iC, iM, iY, iK):
                            cmyk_order = (iC, iM, iY, iK)
                    # ExtraSamples (2 = unassociated alpha, 1 = associated)
                    # TIFF no fija la posición exacta; asumimos alfa es el ÚLTIMO canal si extrasamples presente
                    if extras:
                        alpha_index = image.shape[2] - 1
                # Extraer resolución y convertir a DPI (igual que antes)
                unit_tag = tags.get("ResolutionUnit")
                unit_code = int(unit_tag.value) if unit_tag else None
                x_res_tag = tags.get("XResolution")
                y_res_tag = tags.get("YResolution")

                dpi_x = _apply_resolution_unit(
                    _rational_to_float(x_res_tag.value if x_res_tag else None),
                    unit_code,
                )
                dpi_y = _apply_resolution_unit(
                    _rational_to_float(y_res_tag.value if y_res_tag else None),
                    unit_code,
                )
        else:
            # Para otros formatos, cargar sin cambios; OpenCV entrega BGR en 8-bit típicamente.
            # IMREAD_UNCHANGED respeta canales/dtype del archivo.
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None and image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

    if image is None:
        return None

    array = np.asarray(image)

    # Si viene en 2D (gris), opcionalmente añade eje de canal para coherencia (H, W, 1).
    # Esto NO cambia la información ni el dtype.
    if array.ndim == 2:
        array = array[..., np.newaxis]

    # IMPORTANTE: No normalizar a uint8 ni recortar canales aquí.
    # Eso debe hacerse sólo para la PREVISUALIZACIÓN (viewer), no en el buffer maestro.

    width_mm, height_mm = _compute_size_mm(array.shape[:2], dpi_x, dpi_y)

    return ImageData(
        pixels=array,       # <- dtype y canales intactos
        dpi_x=dpi_x,
        dpi_y=dpi_y,
        width_mm=width_mm,
        height_mm=height_mm,
        photometric=locals().get("photometric"),
        cmyk_order=locals().get("cmyk_order"),
        alpha_index=locals().get("alpha_index"),
        icc_profile=locals().get("icc"),
        ink_names=locals().get("ink_names"),
    )



def save_image_tif(
    path: Path,
    image: np.ndarray,
    photometric: str | None = None,
    dpi_x: float | None = None,
    dpi_y: float | None = None,
    icc_profile: bytes | None = None,
    ink_names: list[str] | None = None,
    extrasamples: list[int] | None = None,
    number_of_inks: int | None = None,
    inkset: int | None = None,  # 1 = CMYK
) -> bool:
    try:
        kws = {
            "append": False,     # <- NO anexar páginas
            "bigtiff": False,    # opcional
            "imagej": False,     # opcional
        }
        if photometric:
            kws["photometric"] = photometric  # 'rgb' | 'minisblack' | 'separated'
        if dpi_x and dpi_y:
            kws["resolution"] = (float(dpi_x), float(dpi_y))
            kws["resolutionunit"] = "INCH"
        if icc_profile:
            kws["iccprofile"] = icc_profile

        extratags = []
        # InkNames (id 333), NumberOfInks (id 334), InkSet (id 332)
        if ink_names:
            payload = ("\x00".join(ink_names) + "\x00").encode("latin1")
            extratags.append((333, "B", len(payload), payload, True))
        if number_of_inks is not None:
            extratags.append((334, "H", 1, number_of_inks, True))
        if inkset is not None:
            extratags.append((332, "H", 1, inkset, True))
        # ExtraSamples (id 338) SOLO si realmente es alfa
        if extrasamples:
            extratags.append((338, "H", len(extrasamples), extrasamples, True))
        if extratags:
            kws["extratags"] = extratags

        tifffile.imwrite(str(path), image, **kws)
        return True
    except Exception:
        return False

