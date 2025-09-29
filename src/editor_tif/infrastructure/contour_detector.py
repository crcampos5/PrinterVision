# src/editor_tif/infrastructure/contour_detector.py
from __future__ import annotations
from typing import List, Tuple, Optional
import math
import cv2
import numpy as np

from editor_tif.domain.models.contours import Contour, Centroid

class ContourDetector:
    """
    Detector de contornos sincronizado con la lógica de connected-components:
    - Los CENTROIDES se obtienen exactamente igual que en `detection.detect_centroids`:
      RGB->GRAY, blur 5x5, Otsu, posible inversión, connectedComponentsWithStats.
    - Los CONTORNOS se extraen por componente (label) para que bbox/ángulo correspondan
      al mismo conjunto de píxeles que define el centroide.
    Salida en píxeles de la referencia.
    """

    def __init__(
        self,
        *,
        min_area: float = 50.0,
        approx_epsilon_factor: float = 0.01,
    ) -> None:
        self.min_area = float(min_area)
        self.approx_epsilon_factor = float(approx_epsilon_factor)

    # ---------- helpers internos ----------
    @staticmethod
    def _to_gray_like_detection(image: np.ndarray) -> np.ndarray:
        """Imita exactamente la conversión usada por detection.py (RGB->GRAY)."""
        if image.ndim == 3:
            # detection.py usa COLOR_RGB2GRAY
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    @staticmethod
    def _binarize_like_detection(gray: np.ndarray) -> np.ndarray:
        """Otsu + inversión condicional (igual que detection.py)."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white = int(np.count_nonzero(thresh))
        black = thresh.size - white
        if white > black:
            thresh = cv2.bitwise_not(thresh)
        return thresh

    def detect(self, image: np.ndarray) -> Tuple[List[Contour], List[Centroid]]:
        if image is None or image.size == 0:
            return [], []

        # === 1) Pipeline idéntico al de detection.py para centroides ===
        gray = self._to_gray_like_detection(image)
        thresh = self._binarize_like_detection(gray)

        # Etiquetado de componentes conectados
        num_labels, labels, stats, centroids_arr = cv2.connectedComponentsWithStats(thresh)

        # Recolecta centroides filtrados por área (igual que detection.py)
        centroids: List[Centroid] = []
        valid_labels: List[int] = []
        for idx in range(1, num_labels):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            cx = float(centroids_arr[idx][0])
            cy = float(centroids_arr[idx][1])
            centroids.append(Centroid(x=cx, y=cy))
            valid_labels.append(idx)

        # Si no hay componentes válidos, salimos temprano
        if not valid_labels:
            return [], []

        # === 2) Contornos por componente (coherentes con los centroides anteriores) ===
        contours: List[Contour] = []
        for idx in valid_labels:
            # Máscara binaria del label actual
            mask = (labels == idx).astype(np.uint8) * 255

            # Busca contornos en LA MÁSCARA del componente
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            # En teoría hay 1 contorno grande; por seguridad toma el de mayor área
            cnt = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(cnt))
            if area < self.min_area:
                continue

            # BBox rotado del componente
            rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
            (cx, cy), (w, h), angle = rect

            # Normalización del ángulo del rectángulo (fallback)
            angle_norm = float(angle)
            if w < h:
                angle_norm += 90.0
                w, h = h, w
            #print("angle_norm: ", angle_norm)
            # Polígono aproximado (opcional)
            eps = self.approx_epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            poly = [(float(p[0][0]), float(p[0][1])) for p in approx]

            principal_axis: Optional[Tuple[float, float]] = None
            angle_from_pca: Optional[float] = None
            if cnt.shape[0] >= 2:
                pts = cnt.reshape(-1, 2).astype(np.float32)
                if pts.shape[0] >= 2:
                    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
                    if eigenvectors is not None and eigenvectors.size >= 2:
                        axis = eigenvectors[0].astype(np.float64)
                        if mean is not None:
                            mean_vec = mean[0].astype(np.float64)
                        else:
                            mean_vec = np.array([cx, cy], dtype=np.float64)
                        pts64 = pts.astype(np.float64)
                        centered = pts64 - mean_vec
                        projections = centered @ axis
                        if projections.size > 0:
                            idx = int(np.argmax(np.abs(projections)))
                            if projections[idx] < 0:
                                axis = -axis
                            norm = float(np.linalg.norm(axis))
                            if norm > 1e-12:
                                axis /= norm
                                axis_x = float(axis[0])
                                axis_y = float(axis[1])
                                principal_axis = (axis_x, axis_y)
                                angle_from_pca = math.degrees(math.atan2(axis_y, axis_x))
                                #print("angle_from_pca: ", angle_from_pca)
                                #if angle_from_pca < -90.0:
                                #    angle_from_pca += 180.0
                                #elif angle_from_pca >= 90.0:
                                #    angle_from_pca -= 180.0

            angle_final = angle_from_pca if angle_from_pca is not None else angle_norm
            #print("angle_final: ", angle_final)
            contours.append(
                Contour(
                    cx=float(cx),
                    cy=float(cy),
                    width=float(w),
                    height=float(h),
                    angle_deg=float(angle_from_pca),
                    polygon=poly if len(poly) >= 3 else None,
                    principal_axis=principal_axis,
                )
            )

        return contours, centroids
