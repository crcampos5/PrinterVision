# qt.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from PySide6.QtGui import QImage, QPixmap

def _cmyk_to_rgb8_from_order(a: np.ndarray, order: Tuple[int,int,int,int]) -> np.ndarray:
    """CMYK uint8 en orden arbitrario -> RGB uint8 (solo preview)."""
    C, M, Y, K = [a[..., i].astype(np.float32) / 255.0 for i in order]
    R = (1.0 - np.minimum(1.0, C + K))
    G = (1.0 - np.minimum(1.0, M + K))
    B = (1.0 - np.minimum(1.0, Y + K))
    return (np.stack([R, G, B], axis=-1) * 255.0).astype(np.uint8)

def _to_preview_rgb8(arr: np.ndarray, photometric_hint: Optional[str] = None,
                     cmyk_order: Optional[Tuple[int,int,int,int]] = None,
                     alpha_index: Optional[int] = None) -> np.ndarray:
    """Normaliza cualquier imagen a RGB uint8 SOLO para previsualización."""
    if arr is None:
        return None
    a = arr

    # 16-bit -> 8-bit lineal (sin gamma extra); otros dtypes -> clamp a 8-bit
    if a.dtype == np.uint16:
        a = (a / 257).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)

    # Asegurar eje de canales
    if a.ndim == 2:
        a = a[..., np.newaxis]

    if a.ndim != 3:
        g = a.astype(np.uint8).squeeze()
        return np.stack([g, g, g], axis=-1)

    ch = a.shape[2]

    # ¿Hay alfa?
    has_alpha = alpha_index is not None and 0 <= alpha_index < ch
    # Caso típico RGBA "puro"
    if (photometric_hint or "").lower() == "rgba" and ch >= 4:
        # Devuelve RGBA tal cual (asegura contigüidad más adelante)
        return a[..., :4]

    if ch == 1:
        g = a[..., 0]
        return np.stack([g, g, g], axis=-1)

    if ch == 3 and not has_alpha:
        # RGB directo
        return a

    # 4+ canales: CMYK (+extras) u otros
    if (photometric_hint or "").lower() in ("separated", "cmyk") and ch >= 4:
        # CMYK -> RGB
        if cmyk_order is None:
            cmyk_order = (0, 1, 2, 3)
        rgb = _cmyk_to_rgb8_from_order(a, cmyk_order)
        if has_alpha:
            alpha = a[..., alpha_index]
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
            return rgba
        return rgb

    # Otras combinaciones con alfa (p.ej. RGB+A etiquetado como None)
    if has_alpha:
        alpha = a[..., alpha_index]
        # Si hay >=3 canales antes del alfa, tómalo como RGB; si no, duplica gris
        if ch >= 4:
            rgb = a[..., :3]
        elif ch == 2:
            g = a[..., 0]
            rgb = np.stack([g, g, g], axis=-1)
        else:
            # Fallback
            rgb = a[..., :3]
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        return rgba

    # Fallback general: primeros 3 canales
    return a[..., :3]

def numpy_to_qimage(array: np.ndarray) -> QImage:
    """Convierte un numpy YA normalizado a RGB8/GRAY8/RGBA8 en QImage."""
    array = np.ascontiguousarray(array)  # asegurar memoria contigua
    if array.ndim == 2:
        h, w = array.shape
        bpl = array.strides[0]
        return QImage(array.data, w, h, bpl, QImage.Format_Grayscale8).copy()

    if array.ndim == 3:
        h, w, ch = array.shape
        bpl = array.strides[0]
        if ch == 3:
            return QImage(array.data, w, h, bpl, QImage.Format_RGB888).copy()
        if ch == 4:
            return QImage(array.data, w, h, bpl, QImage.Format_RGBA8888).copy()

    raise ValueError("Unsupported array shape for QImage conversion")

def numpy_to_qpixmap(array: np.ndarray,
                     photometric_hint: Optional[str] = None,
                     cmyk_order: Optional[Tuple[int,int,int,int]] = None,
                     alpha_index: Optional[int] = None) -> QPixmap:
    preview = _to_preview_rgb8(array,
                               photometric_hint=photometric_hint,
                               cmyk_order=cmyk_order,
                               alpha_index=alpha_index)
    return QPixmap.fromImage(numpy_to_qimage(preview))

def qimage_to_numpy(image: QImage) -> np.ndarray:
    """QImage -> numpy (RGBA8)."""
    image = image.convertToFormat(QImage.Format_RGBA8888)
    w, h = image.width(), image.height()
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    arr = np.array(ptr, dtype=np.uint8).reshape(h, w, 4)
    return arr.copy()
