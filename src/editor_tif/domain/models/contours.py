from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Centroid:
    x: float  # px en la imagen de referencia
    y: float  # px en la imagen de referencia

@dataclass
class Contour:
    cx: float         # centroide X (px)
    cy: float         # centroide Y (px)
    width: float      # ancho del bbox (px)
    height: float     # alto del bbox (px)
    angle_deg: float  # ángulo principal (grados, antihorario)
    polygon: Optional[List[Tuple[float, float]]] = None  # opcional: lista de puntos (px)
    box_vertices: Optional[List[Tuple[float, float]]] = None  # vertices del bbox rotado (px)
    principal_axis: Optional[Tuple[float, float]] = None  # vector unitario del eje mayor (px)
