# src/editor_tif/domain/models/template.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import uuid
import json


class FitMode(str, Enum):
    """
    Estrategia de escalado del ítem respecto al contorno destino.
    - FIT_SHORT: ajusta usando el eje corto del bounding box del contorno.
    - FIT_LONG: ajusta usando el eje largo del bounding box del contorno.
    - FIT_WIDTH: ajusta usando el ancho del bounding box del contorno.
    - FIT_HEIGHT: ajusta usando el alto del bounding box del contorno.
    - STRETCH: estira de forma no uniforme para llenar (scale_x != scale_y).
    - NONE: no escala (útil si ya trabajas a escala física).
    """
    FIT_SHORT = "fit_short"
    FIT_LONG = "fit_long"
    FIT_WIDTH = "fit_width"
    FIT_HEIGHT = "fit_height"
    STRETCH = "stretch"
    NONE = "none"


@dataclass(frozen=True)
class ContourSignature:
    """
    Firma geométrica mínima y agnóstica de UI de un contorno.
    Todas las unidades en coordenadas de escena (tu sistema de unidades),
    y ángulos en grados (convención antihoraria).
    """
    cx: float           # centroide X
    cy: float           # centroide Y
    width: float        # ancho del bounding box del contorno
    height: float       # alto del bounding box del contorno
    angle_deg: float    # orientación principal del contorno (ej. eje mayor)
    polygon: Optional[List[Tuple[float, float]]] = None
    principal_axis: Optional[Tuple[float, float]] = None  # vector unitario del eje mayor (orientado)
    min_rect_vertices: Optional[List[Tuple[float, float]]] = None  # vértices del minAreaRect (coordenadas de escena)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        polygon = data.get("polygon")
        if polygon is not None:
            norm_poly: List[List[float]] = []
            for p in polygon:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    norm_poly.append([float(p[0]), float(p[1])])
                elif hasattr(p, "x") and hasattr(p, "y"):
                    norm_poly.append([float(p.x()), float(p.y())])
            data["polygon"] = norm_poly if norm_poly else None
        principal_axis = data.get("principal_axis")
        if principal_axis is not None:
            try:
                data["principal_axis"] = [float(principal_axis[0]), float(principal_axis[1])]
            except (TypeError, ValueError, IndexError):
                data["principal_axis"] = None
        min_rect_vertices = data.get("min_rect_vertices")
        if min_rect_vertices is not None:
            norm_rect: List[List[float]] = []
            for p in min_rect_vertices:
                if hasattr(p, "x") and hasattr(p, "y"):
                    norm_rect.append([float(p.x()), float(p.y())])
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    norm_rect.append([float(p[0]), float(p[1])])
            data["min_rect_vertices"] = norm_rect if norm_rect else None
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ContourSignature":
        parsed: Dict[str, Any] = dict(data)
        polygon = parsed.get("polygon")
        if polygon is not None:
            norm_poly = []
            for p in polygon:
                try:
                    norm_poly.append(tuple(float(coord) for coord in p[:2]))
                except (TypeError, ValueError):
                    continue
            parsed["polygon"] = norm_poly if norm_poly else None
        principal_axis = parsed.get("principal_axis")
        if principal_axis is not None:
            try:
                parsed["principal_axis"] = (
                    float(principal_axis[0]),
                    float(principal_axis[1]),
                )
            except (TypeError, ValueError, IndexError):
                parsed["principal_axis"] = None
        rect_vertices = parsed.get("min_rect_vertices")
        if rect_vertices is not None:
            norm_rect_vertices = []
            for p in rect_vertices:
                try:
                    norm_rect_vertices.append(tuple(float(coord) for coord in p[:2]))
                except (TypeError, ValueError):
                    continue
            parsed["min_rect_vertices"] = norm_rect_vertices if norm_rect_vertices else None
        return ContourSignature(**parsed)


@dataclass(frozen=True)
class PlacementRule:
    """
    Reglas relativas ítem ↔ contorno capturadas al crear la plantilla.

    - anchor_norm: punto de anclaje del ítem en coordenadas normalizadas
      de su caja (0..1, 0..1). (0.5, 0.5) -> centro del ítem.
    - offset_norm: desplazamiento normalizado dentro del bbox del contorno,
      expresado en el marco local del contorno antes de rotar (0..1, 0..1).
      (0.5, 0.5) -> centro del bbox del contorno.
    - rotation_offset_deg: desfase angular fijo a sumar a la orientación del contorno.
    - fit_mode: estrategia de escalado (ver FitMode).
    - keep_aspect_ratio: si True, fuerza escala uniforme (x=y) cuando aplique.
    """
    anchor_norm: Tuple[float, float] = (0.5, 0.5)
    offset_norm: Tuple[float, float] = (0.5, 0.5)
    rotation_offset_deg: float = 0.0
    fit_mode: FitMode = FitMode.FIT_SHORT
    keep_aspect_ratio: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fit_mode"] = self.fit_mode.value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PlacementRule":
        data = dict(data)
        data["fit_mode"] = FitMode(data.get("fit_mode", FitMode.FIT_SHORT))
        return PlacementRule(**data)


@dataclass(frozen=True)
class Placement:
    """
    Resultado de una colocación (transformación final).
    Separa el cálculo del modelo: útil si decides que services/placement.py
    haga la matemática y devuelva este valor.
    - tx, ty: posición final del punto de anclaje del ítem en escena.
    - rotation_deg: rotación total del ítem.
    - scale_x, scale_y: escalas a aplicar al ítem (uniformes o no).
    """
    tx: float
    ty: float
    rotation_deg: float
    scale_x: float
    scale_y: float
    piv_x: float
    piv_y: float
    matrix: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Placement":
        matrix = data.get("matrix")
        if matrix is not None:
            try:
                matrix = (
                    (float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])),
                    (float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])),
                )
            except (TypeError, ValueError, IndexError):
                matrix = None
        return Placement(
            tx=float(data["tx"]),
            ty=float(data["ty"]),
            rotation_deg=float(data["rotation_deg"]),
            scale_x=float(data["scale_x"]),
            scale_y=float(data["scale_y"]),
            piv_x=float(data["piv_x"]),
            piv_y=float(data["piv_y"]),
            matrix=matrix,
        )


@dataclass
class Template:
    """
    Entidad de dominio que describe cómo replicar un ítem (p. ej., un TIFF)
    sobre contornos similares, conservando la relación geométrica que se
    estableció al crear la plantilla.

    Campos clave:
    - id: UUID único de la plantilla.
    - item_source_id: identificador lógico del recurso del ítem.
      Puede ser una ruta, un id interno, o una clave de repositorio.
    - item_original_size: (w, h) tamaño base del ítem en unidades de escena.
      Se usa como referencia para el escalado relativo.
    - base_contour: firma del contorno con el que se creó la plantilla.
      (sirve para auditoría, depuración o cálculos relativos avanzados).
    - rule: reglas relativas de anclaje/offset/escala/rotación.
    - meta: diccionario opcional para anotar cualquier información adicional.
    """
    item_source_id: str
    item_original_size: Tuple[float, float]
    base_contour: ContourSignature
    rule: PlacementRule = field(default_factory=PlacementRule)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: Dict[str, Any] = field(default_factory=dict)

    # -------------------------
    # SERIALIZACIÓN
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "item_source_id": self.item_source_id,
            "item_original_size": list(self.item_original_size),
            "base_contour": self.base_contour.to_dict(),
            "rule": self.rule.to_dict(),
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Template":
        return Template(
            id=data.get("id", str(uuid.uuid4())),
            item_source_id=data["item_source_id"],
            item_original_size=tuple(data["item_original_size"]),
            base_contour=ContourSignature.from_dict(data["base_contour"]),
            rule=PlacementRule.from_dict(data["rule"]),
            meta=data.get("meta", {}),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(s: str) -> "Template":
        return Template.from_dict(json.loads(s))
