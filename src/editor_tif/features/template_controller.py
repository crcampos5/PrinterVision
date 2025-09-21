from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

from PySide6.QtWidgets import QGraphicsScene
from PySide6.QtCore import QPointF
from PySide6.QtGui import QTransform, QPolygonF

from editor_tif.presentation.views.scene_items import ImageItem, CentroidItem, ContourItem
from editor_tif.domain.models.template import (
    Template,
    ContourSignature,
    PlacementRule,
)
from editor_tif.domain.services.placement import (
    placement_from_template,
    apply_placement_to_item,
)

TargetItem = Union[CentroidItem, ContourItem]


@dataclass
class TemplateRecord:
    template: Template
    name: str = "unnamed"


class TemplateController:
    """
    Administra la creación y aplicación de plantillas:

      - create_template_from_selection:
          requiere 1 ImageItem + (1 CentroidItem o 1 ContourItem).
      - apply_template_to_all_centroids / apply_template_to_selection:
          ahora soportan CentroidItem o ContourItem como destinos.

    Nota: el tamaño original del ítem se computa en unidades de escena (mm_to_scene)
    para que placement sea coherente con lo que ves en el viewer.
    """

    def __init__(
        self,
        scene: QGraphicsScene,
        document,
        *,
        mm_to_scene: float,
        get_selection: Callable[[], Iterable[object]],
        bind_callback: Optional[Callable[[ImageItem], None]] = None,
    ):
        self.scene = scene
        self.document = document
        self.mm_to_scene = float(mm_to_scene)
        self.get_selection = get_selection
        self.bind_callback = bind_callback

        # Repositorio simple en memoria
        self._templates: list[TemplateRecord] = []

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _item_size_scene_units(self, item: ImageItem) -> Tuple[float, float]:
        """
        Devuelve el tamaño "original" del ítem en unidades de escena.
        Prioriza dimensiones físicas del Layer (mm) -> convierte a escena.
        Si no hay mm, usa el boundingRect local convertido por mm_to_scene.
        """
        layer = item.layer
        if layer.width_mm is not None and layer.height_mm is not None:
            w_mm = float(layer.width_mm)
            h_mm = float(layer.height_mm)
            return w_mm * self.mm_to_scene, h_mm * self.mm_to_scene

        # Fallback: boundingRect del pixmap en px -> aproxima 1 px ~ 1 mm_visual * mm_to_scene
        br = item.boundingRect()
        return br.width() * self.mm_to_scene, br.height() * self.mm_to_scene

    @staticmethod
    def _polygon_center(poly: QPolygonF) -> QPointF:
        if poly.isEmpty():
            return QPointF(0.0, 0.0)
        sx = sy = 0.0
        for i in range(poly.count()):
            p = poly.at(i)
            sx += p.x()
            sy += p.y()
        n = float(poly.count())
        return QPointF(sx / n, sy / n)

    def as_contour_signature(self, obj: TargetItem) -> Optional[ContourSignature]:
        """
        Obtiene la ContourSignature para CentroidItem o ContourItem.
        - CentroidItem: usa to_contour_signature() (posición ya en escena).
        - ContourItem: intenta su firma interna (_sig) si existe; si no,
                       estima a partir de su polígono y transform.
        """
        # CentroidItem tiene export directo
        if isinstance(obj, CentroidItem):
            try:
                return obj.to_contour_signature()
            except Exception:
                return None

        # ContourItem: preferir firma interna si existe
        if isinstance(obj, ContourItem):
            sig = getattr(obj, "_sig", None)
            if isinstance(sig, ContourSignature):
                return sig

            # Estimar a partir de su polígono en escena:
            try:
                poly: QPolygonF = obj.mapToScene(obj.polygon())
                if poly.isEmpty():
                    return None

                # Centro por promedio de vértices
                c = self._polygon_center(poly)

                # Ancho/alto: bbox axis-aligned en escena
                rx = [poly.at(i).x() for i in range(poly.count())]
                ry = [poly.at(i).y() for i in range(poly.count())]
                w = max(rx) - min(rx)
                h = max(ry) - min(ry)

                # Ángulo: si hay transform, extraer componente de rotación aproximada
                # (obj.transform() devuelve QTransform local → mapear un vector unitario)
                T: QTransform = obj.sceneTransform()
                p0 = T.map(QPointF(0.0, 0.0))
                px = T.map(QPointF(1.0, 0.0))
                vx, vy = (px.x() - p0.x()), (px.y() - p0.y())
                import math
                angle_deg = math.degrees(math.atan2(vy, vx))

                return ContourSignature(cx=float(c.x()), cy=float(c.y()), width=float(w), height=float(h), angle_deg=float(angle_deg))
            except Exception:
                return None

        return None

    def _ensure_source_item(self, template: Template, source_item: Optional[ImageItem]) -> ImageItem:
        if source_item is not None:
            return source_item
        # Reconstruir desde item_source_id
        return ImageItem.from_source_id(template.item_source_id, self.document, mm_to_scene=self.mm_to_scene)

    # ---------------------------------------------------------------------
    # Crear plantilla
    # ---------------------------------------------------------------------
    def create_template(
        self,
        *,
        source_item: ImageItem,
        base_signature: ContourSignature,
        rule: Optional[PlacementRule] = None,
        name: str = "unnamed",
    ) -> Template:
        """
        Crea una Template a partir de un ImageItem (base) y la firma geométrica
        del contorno con el que se "casó" el ítem.
        """
        iw_scene, ih_scene = self._item_size_scene_units(source_item)
        tpl = Template(
            item_source_id=source_item.source_id,
            item_original_size=(iw_scene, ih_scene),
            base_contour=base_signature,
            rule=rule or PlacementRule(),  # anchor (0.5,0.5), offset (0.5,0.5), fit_short (por defecto)
            meta={"created_from": "controller.create_template"},
        )
        self._templates.append(TemplateRecord(template=tpl, name=name))
        return tpl

    def create_template_from_selection(
        self,
        *,
        name: str = "unnamed",
        rule: Optional[PlacementRule] = None,
    ) -> Template:
        """
        Espera que el usuario haya seleccionado:
          - 1 ImageItem (el TIF base)
          - 1 marcador de posición: CentroidItem o ContourItem
        """
        sel = list(self.get_selection())
        img = next((s for s in sel if isinstance(s, ImageItem)), None)
        tgt = next((s for s in sel if isinstance(s, (CentroidItem, ContourItem))), None)

        if img is None or tgt is None:
            raise ValueError("Selecciona un ImageItem y un marcador (CentroidItem o ContourItem) para crear la plantilla.")

        sig = self.as_contour_signature(tgt)
        if sig is None:
            raise ValueError("No se pudo obtener la firma geométrica del marcador seleccionado.")

        return self.create_template(source_item=img, base_signature=sig, rule=rule, name=name)

    # ---------------------------------------------------------------------
    # Aplicar plantilla
    # ---------------------------------------------------------------------
    def _apply_to_targets(self, template: Template, targets: Iterable[TargetItem], *, source_item: Optional[ImageItem]) -> List[ImageItem]:
        src = self._ensure_source_item(template, source_item)
        created: List[ImageItem] = []
        for t in targets:
            sig = self.as_contour_signature(t)
            if sig is None:
                continue
            placement = placement_from_template(template, sig)

            # Crear clon consistente y aplicar placement
            self.document._layer_seq += 1
            item = ImageItem.from_source_id(template.item_source_id, self.document, mm_to_scene=self.mm_to_scene)
            apply_placement_to_item(item, placement)

            self.scene.addItem(item)
            if self.bind_callback:
                self.bind_callback(item)
            created.append(item)
        return created

    def apply_template_to_all_centroids(
        self,
        template: Template,
        *,
        source_item: Optional[ImageItem] = None,
    ) -> List[ImageItem]:
        """
        Aplica la plantilla clonando el ítem base sobre TODOS los marcadores:
        tanto CentroidItem como ContourItem presentes en la escena.
        (Se mantiene el nombre público para compatibilidad hacia arriba).
        """
        targets: List[TargetItem] = []
        try:
            for it in self.scene.items():
                if isinstance(it, (CentroidItem, ContourItem)):
                    targets.append(it)
        except Exception:
            pass
        return self._apply_to_targets(template, targets, source_item=source_item)

    def apply_template_to_selection(
        self,
        template: Template,
        *,
        source_item: Optional[ImageItem] = None,
        centroids: Optional[Iterable[CentroidItem]] = None,  # compat: argumento antiguo
    ) -> List[ImageItem]:
        """
        Aplica la plantilla a la selección actual. Acepta CentroidItem y ContourItem.
        Si 'centroids' se pasa (API antigua), se respeta, pero se recomienda usar la selección.
        """
        if centroids is not None:
            targets = list(centroids)  # compat con firma previa
        else:
            sel = list(self.get_selection())
            targets = [s for s in sel if isinstance(s, (CentroidItem, ContourItem))]
            if not targets:
                return []
        return self._apply_to_targets(template, targets, source_item=source_item)

    # ---------------------------------------------------------------------
    # Gestión de plantillas en memoria
    # ---------------------------------------------------------------------
    def find_template(self, name: str) -> Optional[Template]:
        rec = next((r for r in self._templates if r.name == name), None)
        return rec.template if rec else None

    def list_templates(self) -> List[str]:
        return [r.name for r in self._templates]
