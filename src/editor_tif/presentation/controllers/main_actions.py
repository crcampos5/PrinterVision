from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Iterable

from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import (
    QFileDialog, QMessageBox, QDialog, QGraphicsPixmapItem, QGraphicsScene
)
from PySide6.QtGui import QKeySequence, QGuiApplication, QTransform

from editor_tif.domain.models.image_document import ImageDocument
from editor_tif.infrastructure.qt_image import numpy_to_qpixmap
from editor_tif.infrastructure.tif_io import load_image_data
from editor_tif.presentation.views.scene_items import ImageItem, Layer, CentroidItem, ContourItem
from editor_tif.presentation.views.template_items import TemplateGroupItem  # nuevo grupo rígido
from editor_tif.presentation.views.workspace_dialog import WorkspaceDialog
from editor_tif.presentation.modes import EditorMode
from editor_tif.domain.services.placement import clone_item_to_centroids
from editor_tif.features.template_controller import TemplateController

# Undo/redo & clipboard
from editor_tif.domain.commands.commands import (
    AddItemCommand, RemoveItemCommand, PasteItemsCommand
)
from editor_tif.presentation.clipboard import serialize_items, deserialize_items


class MainActions:
    """
    Lógica desacoplada de MainWindow:
    - Administra fondo (pixmap) y su transform físico.
    - Materializa centroides/contornos mapeando desde px de referencia a escena.
    - Delegados para abrir/guardar, clonar y plantillas (grupos rígidos).
    """

    def __init__(
        self,
        *,
        window,                             # MainWindow (statusBar, viewer)
        scene: QGraphicsScene,
        document: ImageDocument,
        mm_to_scene: float,
        template_controller: TemplateController,
        selection_handler,
        undo_stack,
        toolbar_manager,
    ) -> None:
        self.window = window
        self.scene = scene
        self.document = document
        self.mm_to_scene = float(mm_to_scene)
        self.template_controller = template_controller
        self.selection_handler = selection_handler
        self.undo_stack = undo_stack
        self.toolbar_manager = toolbar_manager

        self._contour_items: list[ContourItem] = []  # track para visibilidad/limpieza
        self._bg_item: Optional[QGraphicsPixmapItem] = None
        self._mode: EditorMode = EditorMode.CloneByCentroid

        # Mantiene referencia al último Template lógico creado (para aplicar)
        self._last_template = None

    # ==================== Workspace / Referencia / Guardado ====================

    def configure_workspace(self) -> None:
        dialog = WorkspaceDialog(
            self.window,
            width_mm=self.document.workspace_width_mm,
            height_mm=self.document.workspace_height_mm,
        )
        if dialog.exec() == QDialog.Accepted:
            width_mm, height_mm = dialog.values()
            self.document.update_workspace(width_mm, height_mm)
            self._refresh_view()
            self._update_actions_state()
            self._update_status()

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Seleccionar imagen de referencia",
            str(Path.cwd()),
            "Imágenes JPG (*.jpg *.jpeg *.png);;Todos los archivos (*.*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        if not self.document.load_reference(path):
            QMessageBox.warning(
                self.window,
                "Error",
                "No se pudo cargar la referencia o no se detectaron objetos.",
            )
            return

        self._refresh_view()
        self._update_actions_state()
        self._update_status()

    def save_image(self) -> None:
        if not self.document.has_output and not self.document.layers:
            QMessageBox.information(self.window, "Sin resultado", "Genera un resultado antes de guardar.")
            return

        default_name = "resultado.tif"
        if self.document.tile_path is not None:
            default_name = f"{self.document.tile_path.stem}_sobre_centroides.tif"

        file_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Guardar imagen resultante",
            str(Path.cwd() / default_name),
            "Imágenes TIF (*.tif *.TIF)",
        )
        if not file_path:
            return

        if not self.document.save_output(Path(file_path)):
            QMessageBox.warning(self.window, "Error", "No se pudo guardar la imagen resultante.")
            return

        self.window.statusBar().showMessage(f"Imagen guardada en: {file_path}")

    # ============================== Items / Escena ==============================

    def add_item(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Seleccionar imagen (JPG/PNG/TIF)",
            str(Path.cwd()),
            "Imágenes (*.jpg *.jpeg *.png *.tif *.tiff);;Todos (*.*)",
        )
        if not file_path:
            return
        data = load_image_data(Path(file_path))
        if data is None:
            QMessageBox.warning(self.window, "Error", "No se pudo cargar la imagen.")
            return

        self.document._layer_seq += 1
        layer = Layer(
            id=self.document._layer_seq,
            path=Path(file_path),
            pixels=data.pixels,
            photometric=data.photometric,
            cmyk_order=data.cmyk_order,
            alpha_index=data.alpha_index,
            width_mm=data.width_mm,
            height_mm=data.height_mm,
            icc_profile=getattr(data, "icc_profile", None),
            ink_names=getattr(data, "ink_names", None),
            x=20.0 * len(self.document.layers),
            y=20.0 * len(self.document.layers),
            source_id=str(file_path),
        )
        self.document.layers.append(layer)

        item = ImageItem(layer, document=self.document, mm_to_scene=self.mm_to_scene)
        item.setSelected(True)
        self.selection_handler.bind_item(item)

        cmd = AddItemCommand(self.scene, item, pos=item.pos(), text="Add Item")
        self.undo_stack.push(cmd)

        self._fit_all_view()
        self._update_actions_state()

    def delete_selected(self) -> None:
        for it in self.selected_items():
            cmd = RemoveItemCommand(self.scene, it, text="Delete Item")
            self.undo_stack.push(cmd)

    # =========================== Modo / Plantillas ============================

    def set_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        # Sincroniza toolbar y selection handler
        self.toolbar_manager.update_mode(mode)
        self.selection_handler.update_mode(mode)
        if getattr(self.window, "viewer", None) is not None:
            try:
                self.window.viewer.apply_mode_drag(mode)
            except AttributeError:
                pass

        if mode == EditorMode.Template:
            self._populate_contours_items()
            self._set_contours_visible(True)
            self._set_centroids_visible(False)
        else:
            self._populate_centroids_items()
            self._set_centroids_visible(True)
            self._set_contours_visible(False)

        self._update_actions_state()

    def on_clone_centroids(self) -> None:
        sel = self.scene.selectedItems()
        if not sel:
            return
        src = sel[0]
        if not isinstance(src, ImageItem):
            QMessageBox.information(self.window, "Selección inválida", "Selecciona un ImageItem para clonar.")
            return

        created = clone_item_to_centroids(
            scene=self.scene,
            source_item=src,
            document=self.document,
            mm_to_scene=self.mm_to_scene,
            bind_callback=self.selection_handler.bind_item,
        )
        if not created:
            QMessageBox.information(self.window, "Sin centroides", "No hay centroides en escena para clonar.")
            return
        self._update_actions_state()

    def on_create_template(self) -> None:
        """
        Crea el grupo rígido (ImageItem + ContourItem) y registra la plantilla lógica.
        Guarda la última plantilla para aplicar luego desde el toolbar.
        """
        try:
            group, tpl = self.template_controller.create_group_from_selection(name="Plantilla 1")
        except ValueError as e:
            QMessageBox.information(self.window, "Selección insuficiente", str(e))
            return

        self._last_template = tpl
        # Selección y enfoque ya los hace el controller; status informativo:
        self.window.statusBar().showMessage("Plantilla creada y registrada en el panel.")
        self._update_actions_state()

    def on_apply_template_all(self) -> None:
        tpl = self._get_active_template()
        if tpl is None:
            QMessageBox.information(self.window, "Sin plantilla", "Crea primero una plantilla.")
            return
        created = self.template_controller.apply_template_to_all_markers(tpl)
        self.window.statusBar().showMessage(f"Plantilla aplicada a {len(created)} marcadores.")
        self._update_actions_state()

    def on_apply_template_selected(self) -> None:
        tpl = self._get_active_template()
        if tpl is None:
            QMessageBox.information(self.window, "Sin plantilla", "Crea primero una plantilla.")
            return
        created = self.template_controller.apply_template_to_selection(tpl)
        self.window.statusBar().showMessage(f"Plantilla aplicada a {len(created)} marcadores seleccionados.")
        self._update_actions_state()

    def _get_active_template(self):
        # Usa la última creada si existe; de lo contrario, intenta la primera registrada
        if self._last_template is not None:
            return self._last_template
        names = self.template_controller.list_templates()
        if not names:
            return None
        return self.template_controller.find_template(names[-1])

    # =============================== Clipboard ================================

    def copy_selected(self) -> None:
        items = self.selected_items()
        if not items:
            return
        md = serialize_items(items)
        QGuiApplication.clipboard().setMimeData(md)

    def paste_from_clipboard(self) -> None:
        md = QGuiApplication.clipboard().mimeData()
        items = deserialize_items(md, self.item_factory, offset=QPointF(16, 16))
        if not items:
            return
        cmd = PasteItemsCommand(self.scene, items, text="Paste Items")
        self.undo_stack.push(cmd)

        self.scene.clearSelection()
        for it in items:
            try:
                it.setSelected(True)
            except Exception:
                pass

    def item_factory(self, cls_name: str, source_id: str):
        if cls_name == "ImageItem":
            it = ImageItem.from_source_id(source_id, document=self.document, mm_to_scene=self.mm_to_scene)
            it.setSelected(True)
            self.selection_handler.bind_item(it)
            return it
        if cls_name == "CentroidItem":
            return CentroidItem(radius_px=6.0)
        raise ValueError(f"Tipo no soportado: {cls_name}")

    # =============================== Helpers UI ===============================

    def selected_items(self):
        return [it for it in self.scene.selectedItems()]

    # ---------- Fondo (pixmap de referencia u output) ----------
    def _set_background_pixmap(self, pixmap) -> None:
        """Crea/actualiza el QGraphicsPixmapItem de fondo y aplica el transform físico."""
        if self._bg_item is not None:
            self.scene.removeItem(self._bg_item)
            self._bg_item = None

        self._bg_item = QGraphicsPixmapItem(pixmap)
        self._bg_item.setZValue(-1000)
        self._bg_item.setFlag(self._bg_item.GraphicsItemFlag.ItemIsSelectable, False)
        self._bg_item.setFlag(self._bg_item.GraphicsItemFlag.ItemIsMovable, False)
        self.scene.addItem(self._bg_item)

        # Escala física: referencia mm/px → escena (mm)
        ref_mpp = self.document.get_mm_per_pixel()
        if ref_mpp is not None:
            ref_mpp_x, ref_mpp_y = ref_mpp
            t = QTransform()
            t.scale(float(ref_mpp_x) * self.mm_to_scene, float(ref_mpp_y) * self.mm_to_scene)
            self._bg_item.setTransform(t, False)
        else:
            self._bg_item.setTransform(QTransform(), False)

        self._bg_item.setPos(0.0, 0.0)
        self.selection_handler.rebind()
        self._fit_all_view()

    def _refresh_view(self) -> None:
        """
        - Si hay output: muestra output y NO pinta centroides (pertenecen a la referencia).
        - Si hay referencia: muestra overlay y PINTA centroides/contornos alineados con mapToScene.
        """
        if self.document.has_output:
            image = self.document.get_output_preview()
            if image is not None:
                pix = numpy_to_qpixmap(
                    image,
                    photometric_hint=getattr(self.document, "tile_photometric", None),
                    cmyk_order=getattr(self.document, "tile_cmyk_order", None),
                    alpha_index=getattr(self.document, "tile_alpha_index", None),
                )
                self._set_background_pixmap(pix)
                self._clear_centroids_items()
                self._clear_contours_items()
                return

        image = self.document.get_reference_preview()
        if image is not None:
            pix = numpy_to_qpixmap(image)
            self._set_background_pixmap(pix)
            self._populate_centroids_items()
            if self._mode == EditorMode.Template:
                self._populate_contours_items()
                self._set_contours_visible(True)
                self._set_centroids_visible(False)
        else:
            if self._bg_item is not None:
                self.scene.removeItem(self._bg_item)
                self._bg_item = None
            self._clear_centroids_items()
            self._clear_contours_items()

    # ---------- Centroides (px_ref → escena) ----------
    def _clear_centroids_items(self) -> None:
        for it in list(self.scene.items()):
            if isinstance(it, CentroidItem):
                self.scene.removeItem(it)

    def _iterate_centroids_px(self):
        # Compat: primero centroids_px; si no, centroids (modelo viejo)
        if hasattr(self.document, "centroids_px") and self.document.centroids_px:
            return self.document.centroids_px
        if hasattr(self.document, "centroids") and self.document.centroids:
            return self.document.centroids
        return []

    def _populate_centroids_items(self) -> None:
        """Crea CentroidItem alineados al fondo (mapToScene sobre px de referencia)."""
        self._clear_centroids_items()
        if self._bg_item is None:
            return

        for c in self._iterate_centroids_px():
            x = float(getattr(c, "x", c[0] if isinstance(c, (tuple, list)) else 0.0))
            y = float(getattr(c, "y", c[1] if isinstance(c, (tuple, list)) else 0.0))
            p_scene = self._bg_item.mapToScene(QPointF(x, y))
            dot = CentroidItem(radius_px=6.0)
            dot.setPos(p_scene)
            self.scene.addItem(dot)

    # ---------- Contornos (px_ref → escena) ----------
    def _iterate_contours_px(self):
        return getattr(self.document, "contours_px", []) or []

    def _clear_contours_items(self) -> None:
        for it in list(self.scene.items()):
            if isinstance(it, ContourItem):
                self.scene.removeItem(it)
        self._contour_items.clear()

    def _populate_contours_items(self) -> None:
        """Construye ContourItem desde contours_px y los posiciona en escena."""
        self._clear_contours_items()
        if self._bg_item is None:
            return

        mmpp = self.document.get_mm_per_pixel()
        if mmpp is None:
            return
        sx = float(mmpp[0]) * self.mm_to_scene
        sy = float(mmpp[1]) * self.mm_to_scene

        for ct in self._iterate_contours_px():
            cx, cy = float(ct.cx), float(ct.cy)
            p_scene = self._bg_item.mapToScene(QPointF(cx, cy))
            w_scene = float(ct.width) * sx
            h_scene = float(ct.height) * sy

            # Si tu ContourItem tiene set_from_signature:
            try:
                from editor_tif.domain.models.template import ContourSignature
                poly_scene = None
                if ct.polygon and len(ct.polygon) >= 3:
                    poly_scene = [
                        self._bg_item.mapToScene(QPointF(float(x), float(y)))
                        for (x, y) in ct.polygon
                    ]

                principal_axis = None
                if getattr(ct, "principal_axis", None):
                    ax, ay = ct.principal_axis
                    ax_scene = float(ax) * sx
                    ay_scene = float(ay) * sy
                    norm = math.hypot(ax_scene, ay_scene)
                    if norm > 1e-9:
                        principal_axis = (ax_scene / norm, ay_scene / norm)

                sig = ContourSignature(
                    cx=float(p_scene.x()),
                    cy=float(p_scene.y()),
                    width=w_scene,
                    height=h_scene,
                    angle_deg=float(ct.angle_deg),
                    polygon=poly_scene,
                    principal_axis=principal_axis,
                )
                item = ContourItem()
                item.set_from_signature(sig)
            except Exception:
                # Fallback simple: rect axis-aligned
                item = ContourItem()
                item.setRect(p_scene.x() - w_scene / 2.0, p_scene.y() - h_scene / 2.0, w_scene, h_scene)

            self.scene.addItem(item)
            self._contour_items.append(item)

    # ---------- Visibilidad por capa ----------
    def _set_centroids_visible(self, visible: bool) -> None:
        for it in self.scene.items():
            if isinstance(it, CentroidItem):
                it.setVisible(visible)

    def _set_contours_visible(self, visible: bool) -> None:
        for it in self._contour_items:
            try:
                it.setVisible(visible)
            except Exception:
                pass

    # ---------- Varios ----------
    def _fit_all_view(self) -> None:
        r = self.scene.itemsBoundingRect()
        if r.isNull():
            return
        m = 40
        r.adjust(-m, -m, m, m)
        self.window.viewer.fitInView(r, Qt.KeepAspectRatio)

    def _update_actions_state(self) -> None:
        """
        Habilita/deshabilita acciones de toolbar según selección, escena y modo:
        - Crear plantilla: 1 ImageItem + 1 (Contour|Centroid) seleccionados.
        - Aplicar a todos: existe al menos 1 plantilla en escena.
        - Aplicar a selección: hay una plantilla seleccionada.
        - Clonar en centroides: ImageItem seleccionado + >=1 centroid en escena.
        """
        # --- Guardar (si corresponde) ---
        try:
            acts = self.toolbar_manager.actions
            if "save" in acts:
                acts["save"].setEnabled(self.document.has_output or bool(self.document.layers))
        except Exception:
            pass

        # --- Recolecta estado de selección/escena ---
        sel = list(self.scene.selectedItems()) if self.scene else []
        scene_items = list(self.scene.items()) if self.scene else []

        n_img      = sum(isinstance(s, ImageItem) for s in sel)
        n_tgt      = sum(isinstance(s, (CentroidItem, ContourItem)) for s in sel)
        has_imgSel = n_img > 0
        has_tgtSel = n_tgt > 0

        # Plantillas en escena y seleccionadas
        has_templates     = any(isinstance(it, TemplateGroupItem) for it in scene_items)
        has_template_sel  = any(isinstance(s, TemplateGroupItem) for s in sel)

        # Centroides en escena
        has_any_centroid  = any(isinstance(it, CentroidItem) for it in scene_items)

        # --- Reglas por acción ---
        can_create_tpl = (n_img == 1 and n_tgt == 1)
        can_apply_all  = has_templates and has_template_sel
        can_apply_sel  = has_templates and has_template_sel and has_tgtSel
        can_clone      = has_imgSel and has_any_centroid

        # --- Si manejas modos, respétalos (opcional pero recomendable) ---
        mode = getattr(self, "mode", None)
        if mode == EditorMode.CloneByCentroid:
            # En modo Clone, desactiva acciones de plantilla
            self.toolbar_manager.set_template_enabled(create=False, apply_all=False, apply_sel=False)
            self.toolbar_manager.set_clone_enabled(can_clone)
            return
        elif mode == EditorMode.Template:
            # En modo Plantilla, desactiva "clone"
            self.toolbar_manager.set_clone_enabled(False)
            self.toolbar_manager.set_template_enabled(create=can_create_tpl, apply_all=can_apply_all, apply_sel=can_apply_sel)
            return

        # --- Sin modo específico: aplica reglas generales ---
        self.toolbar_manager.set_template_enabled(create=can_create_tpl, apply_all=can_apply_all, apply_sel=can_apply_sel)
        self.toolbar_manager.set_clone_enabled(can_clone)

    def _update_status(self) -> None:
        parts: list[str] = []
        if getattr(self.document, "reference_path", None) is not None:
            parts.append(f"Referencia: {self.document.reference_path.name}")

        n_c = len(self._iterate_centroids_px())
        parts.append(f"Objetos: {n_c}")

        mm_per_pixel = self.document.get_mm_per_pixel()
        if mm_per_pixel is not None:
            parts.append(f"Escala: {mm_per_pixel[0]:.3f} mm/px (X), {mm_per_pixel[1]:.3f} mm/px (Y)")

        tile_dims = self.document.get_tile_dimensions_mm()
        if tile_dims is not None:
            parts.append(f"TIF: {tile_dims[0]:.1f} x {tile_dims[1]:.1f} mm")

        message = " | ".join(parts) if parts else "Carga una referencia JPG para comenzar."
        self.window.statusBar().showMessage(message)
