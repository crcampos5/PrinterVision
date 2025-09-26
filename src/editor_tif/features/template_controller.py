from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, List, Optional, Tuple, Union
import math

from PySide6.QtWidgets import QGraphicsScene
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPixmap,
    QPolygonF,
    QStandardItem,
    QTransform,
)

# Vista / Items
from editor_tif.presentation.views.scene_items import ImageItem, CentroidItem, ContourItem

# Dominio (modelos y colocación)
from editor_tif.domain.models.template import (
    Template,
    ContourSignature,
    PlacementRule,
    FitMode,
)
from editor_tif.domain.services.placement import (
    placement_from_template,
    apply_placement_to_item,
)

# Grupo visual (overlay de plantilla)
from editor_tif.presentation.views.template_items import TemplateGroupItem

TargetItem = Union[CentroidItem, ContourItem]


def _signature_direction_angle_deg(sig: ContourSignature) -> Optional[float]:
    axis = getattr(sig, "principal_axis", None)
    if not axis:
        return None
    try:
        vx = float(axis[0])
        vy = float(axis[1])
    except (TypeError, ValueError):
        return None
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return None
    ang = math.degrees(math.atan2(vy, vx))
    return ang % 360.0


@dataclass
class TemplateRecord:
    """Registro de plantilla 'lógica' (modelo) para clonación automática."""
    template: Template
    name: str = "unnamed"


@dataclass
class GroupRecord:
    """Registro visual para el panel (grupo rígido en escena)."""
    group: TemplateGroupItem
    name: str


class TemplateController:
    """
    Controlador de plantillas:
      - Crea grupos rígidos (ImageItem + ContourItem) desde la selección.
      - Registra miniaturas/nombre/visibilidad en el panel.
      - Mantiene la plantilla lógica (Template) para clonaciones posteriores.

    Requisitos de selección al crear:
      - 1 ImageItem
      - 1 ContourItem (o CentroidItem si allow_centroid=True)
    """

    def __init__(
        self,
        scene: QGraphicsScene,
        document,
        *,
        mm_to_scene: float,
        get_selection: Callable[[], Iterable[object]],
        # QStandardItemModel (3 columnas: thumb, nombre, visible)
        panel_model=None,
        bind_callback: Optional[Callable[[ImageItem], None]] = None,
    ):
        self.scene = scene
        self.document = document
        self.mm_to_scene = float(mm_to_scene)
        self.get_selection = get_selection
        self.bind_callback = bind_callback

        self.panel_model = panel_model

        # Repos
        self._templates: list[TemplateRecord] = []
        self._groups: list[GroupRecord] = []

        # Paleta contornos de grupo
        self._palette = [
            QColor("#FF6B6B"),
            QColor("#4D96FF"),
            QColor("#6BCB77"),
            QColor("#FFD93D"),
            QColor("#A66CFF"),
        ]
        self._color_idx = 0

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _set_layer_overlay_flag(self, layer, value: bool) -> None:
        if layer is None:
            return
        try:
            layer.is_template_overlay = value
        except AttributeError:
            pass

    def _next_color(self) -> QColor:
        c = self._palette[self._color_idx % len(self._palette)]
        self._color_idx += 1
        return c

    def _item_size_scene_units(self, item: ImageItem) -> Tuple[float, float]:
        """
        Devuelve el tamaño "original" del ítem en unidades de escena.
        Prioriza dimensiones físicas del Layer (mm) -> convierte a escena.
        """
        layer = item.layer
        if layer.width_mm is not None and layer.height_mm is not None:
            w_mm = float(layer.width_mm)
            h_mm = float(layer.height_mm)
            return w_mm * self.mm_to_scene, h_mm * self.mm_to_scene

        # Fallback: boundingRect local * mm_to_scene (aprox)
        br = item.boundingRect()
        return br.width() * self.mm_to_scene, br.height() * self.mm_to_scene

    def _item_center_in_scene(self, item: ImageItem) -> QPointF:
        """Centro exacto del item en COORDENADAS DE ESCENA (respeta offset/rotación/escala)."""
        return item.mapToScene(item.boundingRect().center())

    def as_contour_signature(self, obj: TargetItem) -> Optional[ContourSignature]:
        """
        Obtiene la ContourSignature para CentroidItem o ContourItem.
        - CentroidItem: usa to_contour_signature().
        - ContourItem: usa su firma interna (_sig) si existe; si no, la estima del polígono/transform.
        """
        if isinstance(obj, CentroidItem):
            try:
                return obj.to_contour_signature()
            except Exception:
                return None

        if isinstance(obj, ContourItem):
            sig = getattr(obj, "_sig", None)
            if isinstance(sig, ContourSignature):
                print("siempre tiene")
                return sig

        return None

    def _ensure_source_item(self, template: Template, source_item: Optional[ImageItem]) -> ImageItem:
        if source_item is not None:
            return source_item
        return ImageItem.from_source_id(template.item_source_id, self.document, mm_to_scene=self.mm_to_scene)

    # ---------------------------------------------------------------------
    # Medición de relación relativa (offset_norm y rotation_offset) al CREAR
    # ---------------------------------------------------------------------

    def _measure_rule_from_item_and_signature(
        self,
        item: ImageItem,
        sig: ContourSignature,
        base_rule: Optional[PlacementRule] = None,
    ) -> PlacementRule:
        """
        Mide la relación relativa Item↔Contorno para construir una PlacementRule inmutable.

        - offset_norm: posición del CENTRO del ítem dentro del bbox del contorno,
        medida en el MARCO LOCAL del contorno (antes de rotar) y normalizada [0..1].
        (0.5, 0.5) -> centro del bbox del contorno.
        - rotation_offset_deg: rotación relativa = rot_item - angle_contour.
        - anchor_norm: fijo en (0.5, 0.5) para anclar el centro del ítem.
        - fit_mode: NONE (sin escalado).
        """
        # Centro del ítem en COORDENADAS DE ESCENA (respeta offset/rot/escala del item)
        ci_scene: QPointF = self._item_center_in_scene(item)

        # Centro del contorno en escena
        cx, cy = float(sig.cx), float(sig.cy)

        # Delta en escena
        dx_scene = ci_scene.x() - cx
        dy_scene = ci_scene.y() - cy

        # Pasar delta al marco LOCAL DEL CONTORNO (rotando por -angle)
        direction_angle_deg = _signature_direction_angle_deg(sig)
        angle_for_offsets_deg = direction_angle_deg if direction_angle_deg is not None else float(sig.angle_deg)
        ang_rad = math.radians(angle_for_offsets_deg)
        cos_a = math.cos(ang_rad)
        sin_a = math.sin(ang_rad)
        # Rot(-ang) * [dx, dy]
        dx_loc = cos_a * dx_scene + sin_a * dy_scene
        dy_loc = -sin_a * dx_scene + cos_a * dy_scene

        # Normalizar respecto al bbox del contorno: (0.5,0.5) es el centro del bbox
        w = max(float(sig.width), 1e-6)
        h = max(float(sig.height), 1e-6)
        off_xn = 0.5 + (dx_loc / w)
        off_yn = 0.5 + (dy_loc / h)

        # Clamp [0,1] (robustez)
        off_xn = 0.0 if off_xn < 0.0 else (1.0 if off_xn > 1.0 else off_xn)
        off_yn = 0.0 if off_yn < 0.0 else (1.0 if off_yn > 1.0 else off_yn)

        # Rotación relativa (módulo 360)
        contour_angle_deg = direction_angle_deg if direction_angle_deg is not None else float(sig.angle_deg)
        rot_off = (float(item.rotation()) - contour_angle_deg) % 360.0

        # Construir nueva regla (PlacementRule es frozen → no mutar, crear copia)
        if base_rule is None:
            new_rule = PlacementRule(
                anchor_norm=(0.5, 0.5),
                offset_norm=(off_xn, off_yn),
                rotation_offset_deg=rot_off,
                fit_mode=FitMode.NONE,
                keep_aspect_ratio=False,
            )
        else:
            new_rule = replace(
                base_rule,
                anchor_norm=(0.5, 0.5),
                offset_norm=(off_xn, off_yn),
                rotation_offset_deg=rot_off,
                fit_mode=FitMode.NONE,
                keep_aspect_ratio=False,
            )

        return new_rule

    # ---------------------------------------------------------------------
    # Crear plantilla lógica (domain) — útil para clonaciones automáticas
    # ---------------------------------------------------------------------

    def create_template(
            self,
            *,
            source_item: ImageItem,
            base_signature: ContourSignature,
            rule: Optional[PlacementRule] = None,
            name: str = "unnamed", ) -> Template:
        # Medir relación relativa y consolidar la regla (sin escalado)
        measured_rule = self._measure_rule_from_item_and_signature(
            source_item, base_signature, base_rule=rule)

        # Tamaño original del ítem en unidades de escena (sólo informativo)
        iw_scene, ih_scene = self._item_size_scene_units(source_item)

        tpl = Template(
            item_source_id=source_item.source_id,
            item_original_size=(iw_scene, ih_scene),
            base_contour=base_signature,
            rule=measured_rule,
            meta={"created_from": "controller.create_template"},
        )
        self._templates.append(TemplateRecord(template=tpl, name=name))
        return tpl

    # ---------------------------------------------------------------------
    # Crear grupo rígido desde la selección + registrar en panel
    # ---------------------------------------------------------------------

    def create_group_from_selection(
        self,
        *,
        name: str = "unnamed",
        allow_centroid: bool = False,
        rule: Optional[PlacementRule] = None,
    ) -> Tuple[TemplateGroupItem, Template]:
        """
        Crea el grupo (ImageItem + ContourItem/CentroidItem) y lo da de alta en el panel.
        Devuelve (grupo, plantilla_lógica).

        - Requiere 1 ImageItem y 1 ContourItem.
        - Si allow_centroid=True, permite CentroidItem como marcador.
        """
        sel = list(self.get_selection())
        img = next((s for s in sel if isinstance(s, ImageItem)), None)
        tgt_types = (ContourItem,) if not allow_centroid else (
            CentroidItem, ContourItem)
        tgt = next((s for s in sel if isinstance(s, tgt_types)), None)

        if img is None or tgt is None:
            raise ValueError(
                "Selecciona un ImageItem y un ContourItem para crear la plantilla.")

        # Obtener firma geométrica del marcador
        sig = self.as_contour_signature(tgt)
        if sig is None:
            raise ValueError(
                "No se pudo obtener la firma geométrica del marcador seleccionado.")

        # Crear PLANTILLA LÓGICA midiendo la relación relativa (sin escalado)
        tpl = self.create_template(
            source_item=img, base_signature=sig, rule=rule, name=name)

        # Crear GRUPO VISUAL
        color = QColor("#6BCB77")  # o self._next_color()
        group = TemplateGroupItem(img, tgt, contour_color=color)
        self.scene.addItem(group)

        layer = getattr(img, "layer", None)
        self._set_layer_overlay_flag(layer, True)

        if layer is not None:
            group.destroyed.connect(lambda *_: self._set_layer_overlay_flag(layer, False))

        # Registrar en panel
        self._register_group_in_panel(group, name=name)

        # Seleccionar grupo resultante y centrar
        self.select_and_center(group)

        return group, tpl

    def _register_group_in_panel(self, group: TemplateGroupItem, *, name: str):
        """Inserta una fila (thumb, nombre, visible) en el modelo del panel."""
        if self.panel_model is None:
            # Aún así conserva registro interno
            self._groups.append(GroupRecord(group=group, name=name))
            return

        # Render miniatura desde la escena
        thumb = QPixmap(80, 80)
        thumb.fill(Qt.transparent)
        p = QPainter(thumb)
        p.setRenderHints(QPainter.Antialiasing |
                         QPainter.SmoothPixmapTransform, True)
        try:
            br = group.mapRectToScene(group.boundingRect())
            if br.width() <= 0 or br.height() <= 0:
                br = QRectF(0, 0, 10, 10)
            self.scene.render(p, QRectF(0, 0, 80, 80), br)
        finally:
            p.end()

        icon_it = QStandardItem()
        icon_it.setEditable(False)
        icon_it.setIcon(QIcon(thumb))
        icon_it.setData(group, role=Qt.UserRole + 1)

        name_it = QStandardItem(name)
        name_it.setEditable(True)
        name_it.setData(group, role=Qt.UserRole + 1)

        vis_it = QStandardItem()
        vis_it.setCheckable(True)
        vis_it.setCheckState(Qt.Checked)
        vis_it.setData(group, role=Qt.UserRole + 1)

        self.panel_model.appendRow([icon_it, name_it, vis_it])

        # Guarda registro interno
        self._groups.append(GroupRecord(group=group, name=name))

    # ---------------------------------------------------------------------
    # Acciones sobre grupos visibles en la escena / panel
    # ---------------------------------------------------------------------
    def toggle_visibility(self, group: TemplateGroupItem, visible: bool):
        group.setVisible(visible)

    def remove_group(self, group: TemplateGroupItem):
        image = group.image_item() if hasattr(group, "image_item") else None
        layer = getattr(image, "layer", None)
        self._set_layer_overlay_flag(layer, False)

        # Quita de lista interna
        self._groups = [g for g in self._groups if g.group is not group]
        # Quita de escena
        self.scene.removeItem(group)
        group.deleteLater()
        # Limpia fila(s) del panel si existe modelo
        if self.panel_model is not None:
            rows_to_remove = []
            for r in range(self.panel_model.rowCount()):
                it = self.panel_model.item(r, 1)  # columna "Nombre"
                if it and it.data(Qt.UserRole + 1) is group:
                    rows_to_remove.append(r)
            for r in reversed(rows_to_remove):
                self.panel_model.removeRow(r)

    def mark_group_as_editable(self, group: TemplateGroupItem) -> None:
        """Limpia la marca de overlay cuando un grupo vuelve a un ImageItem editable."""
        image = group.image_item() if hasattr(group, "image_item") else None
        layer = getattr(image, "layer", None)
        self._set_layer_overlay_flag(layer, False)

    def select_and_center(self, group: TemplateGroupItem):
        """Selecciona el grupo y centra la vista en él (si hay view)."""
        sc = group.scene()
        if sc:
            sc.clearSelection()
        group.setSelected(True)
        views = sc.views() if sc else []
        if views:
            views[0].centerOn(group)

    # ---------------------------------------------------------------------
    # Aplicar plantilla lógica a marcadores (clonación)
    # ---------------------------------------------------------------------
    def _apply_to_targets(
        self,
        template: Template,
        targets: Iterable[TargetItem],
        *,
        source_item: Optional[ImageItem],
    ) -> List[ImageItem]:
        # reservado por si luego usamos el src
        _ = self._ensure_source_item(template, source_item)
        created: List[ImageItem] = []
        for t in targets:
            sig = self.as_contour_signature(t)
            if sig is None:
                continue
            placement = placement_from_template(template, sig)

            # Crear clon y aplicar placement
            self.document._layer_seq += 1
            item = ImageItem.from_source_id(
                template.item_source_id, self.document, mm_to_scene=self.mm_to_scene)
            apply_placement_to_item(item, placement)

            self.scene.addItem(item)
            if self.bind_callback:
                self.bind_callback(item)
            created.append(item)
        return created

    def apply_template_to_all_markers(
        self,
        template: Template,
        *,
        source_item: Optional[ImageItem] = None,
    ) -> List[ImageItem]:
        """Clona sobre TODOS los marcadores (ContourItem y CentroidItem) presentes en la escena."""
        targets: List[TargetItem] = []
        try:
            for it in self.scene.items():
                if isinstance(it, (ContourItem)):
                    targets.append(it)
        except Exception:
            pass
        return self._apply_to_targets(template, targets, source_item=source_item)

    def apply_template_to_selection(
        self,
        template: Template,
        *,
        source_item: Optional[ImageItem] = None,
        centroids: Optional[Iterable[CentroidItem]
                            ] = None,  # compat con API anterior
    ) -> List[ImageItem]:
        """
        Clona sobre la selección actual (acepta CentroidItem y ContourItem).
        Si 'centroids' se pasa (API antigua), se respeta.
        """
        if centroids is not None:
            targets = list(centroids)
        else:
            sel = list(self.get_selection())
            targets = [s for s in sel if isinstance(
                s, (CentroidItem, ContourItem))]
            if not targets:
                return []
        return self._apply_to_targets(template, targets, source_item=source_item)

    # ---------------------------------------------------------------------
    # Gestión básica del repositorio lógico
    # ---------------------------------------------------------------------
    def find_template(self, name: str) -> Optional[Template]:
        rec = next((r for r in self._templates if r.name == name), None)
        return rec.template if rec else None

    def list_templates(self) -> List[str]:
        return [r.name for r in self._templates]
