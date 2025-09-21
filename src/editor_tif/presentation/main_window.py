# editor_tif/main_window.py
"""Main window housing menus, toolbars, and the image viewer."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QAction, QTransform, QUndoStack, QKeySequence, QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog, QMainWindow, QMessageBox, QDialog, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QPointF

from editor_tif.domain.models.image_document import ImageDocument
from editor_tif.infrastructure.qt_image import numpy_to_qpixmap
from editor_tif.presentation.views.image_viewer import ImageViewer
from editor_tif.presentation.views.workspace_dialog import WorkspaceDialog
from editor_tif.presentation.views.scene_items import ImageItem, Layer, CentroidItem
from editor_tif.presentation.views.properties_dock import PropertiesDock
from editor_tif.infrastructure.tif_io import load_image_data
from editor_tif.presentation.views.toolbar_manager import ToolbarManager
from editor_tif.presentation.views.selection_handler import SelectionHandler
from editor_tif.presentation.modes import EditorMode
from editor_tif.domain.services.placement import clone_item_to_centroids

# 👇 nuevos módulos propuestos en src/editor_tif/
from editor_tif.domain.commands.commands import (
    AddItemCommand, RemoveItemCommand, TransformItemCommand, PasteItemsCommand
)
from editor_tif.presentation.clipboard import serialize_items, deserialize_items


class MainWindow(QMainWindow):
    """Application main window with top toolbar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PrinterVision Editor")
        self.resize(1200, 800)

        # 1 unidad de escena = 1 mm (para futuros ítems físicos)
        self.mm_to_scene = 1.0

        # Canvas central (QGraphicsView/QGraphicsScene)
        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)

        # Dock de propiedades
        self.props = PropertiesDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.props)

        # Toolbar + selección
        self.toolbar_manager = ToolbarManager(self)
        self.selection_handler = SelectionHandler(self)

        # Fondo de referencia
        self._bg_item: Optional[QGraphicsPixmapItem] = None

        # Documento de datos (referencia, capas, resultado)
        self.document = ImageDocument()

        # ====== Undo/Redo ======
        self.undo_stack = QUndoStack(self)
        self._install_shortcuts()

        # Conexiones UI
        self.viewer.scene().selectionChanged.connect(
            self.selection_handler.on_selection_changed
        )
        self.props.posChanged.connect(self.selection_handler.on_props_pos)
        self.props.rotChanged.connect(self.selection_handler.on_props_rot)
        self.props.scaleChanged.connect(self.selection_handler.on_props_scale)
        self.props.opacityChanged.connect(self.selection_handler.on_props_opacity)

        # Estado inicial
        self.set_mode(EditorMode.CloneByCentroid)
        self._update_actions_state()
        self._update_status()

    # ===================== Shortcuts / acciones globales =====================

    def _install_shortcuts(self) -> None:
        """Crea acciones globales para Undo/Redo/Delete/Copy/Paste."""
        act_undo = QAction("Deshacer", self, shortcut=QKeySequence.Undo, triggered=self.undo_stack.undo)
        act_redo = QAction("Rehacer", self, shortcut=QKeySequence.Redo, triggered=self.undo_stack.redo)

        act_delete = QAction("Eliminar", self, shortcut=QKeySequence.Delete, triggered=self.delete_selected)
        act_copy = QAction("Copiar", self, shortcut=QKeySequence.Copy, triggered=self.copy_selected)
        act_paste = QAction("Pegar", self, shortcut=QKeySequence.Paste, triggered=self.paste_from_clipboard)

        # Añadimos al MainWindow para que capten teclas incluso si la vista tiene foco
        for a in (act_undo, act_redo, act_delete, act_copy, act_paste):
            self.addAction(a)

        # Si tu ToolbarManager expone acciones, también puedes conectarlas aquí
        try:
            self.toolbar_manager.bind_undo_stack(self.undo_stack)  # opcional si lo implementas
        except Exception:
            pass

    # =========================== Acciones núcleo ============================

    def configure_workspace(self) -> None:
        dialog = WorkspaceDialog(
            self,
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
            self,
            "Seleccionar imagen de referencia",
            str(Path.cwd()),
            "Imágenes JPG (*.jpg *.jpeg);;Todos los archivos (*.*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        if not self.document.load_reference(path):
            QMessageBox.warning(
                self,
                "Error",
                "No se pudo cargar la referencia o no se detectaron objetos.",
            )
            return

        self._refresh_view()
        self._update_actions_state()
        self._update_status()

    def save_image(self) -> None:
        if not self.document.has_output and not self.document.layers:
            QMessageBox.information(self, "Sin resultado", "Genera un resultado antes de guardar.")
            return

        default_name = "resultado.tif"
        if self.document.tile_path is not None:
            default_name = f"{self.document.tile_path.stem}_sobre_centroides.tif"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar imagen resultante",
            str(Path.cwd() / default_name),
            "Imágenes TIF (*.tif *.TIF)",
        )
        if not file_path:
            return

        if not self.document.save_output(Path(file_path)):
            QMessageBox.warning(self, "Error", "No se pudo guardar la imagen resultante.")
            return

        self.statusBar().showMessage(f"Imagen guardada en: {file_path}")

    def add_item(self) -> None:
        """Inserta una imagen (JPG/PNG/TIF) como ImageItem editable en la escena."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen (JPG/PNG/TIF)",
            str(Path.cwd()),
            "Imágenes (*.jpg *.jpeg *.png *.tif *.tiff);;Todos (*.*)",
        )
        if not file_path:
            return
        data = load_image_data(Path(file_path))
        if data is None:
            QMessageBox.warning(self, "Error", "No se pudo cargar la imagen.")
            return

        # Crea un Layer y su ImageItem asociado
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

        # 👉 En lugar de addItem directo, usamos comando Undoable
        cmd = AddItemCommand(self.viewer.scene(), item, pos=item.pos(), text="Add Item")
        self.undo_stack.push(cmd)

        self._fit_all_view()
        self._update_actions_state()

    # ============================ Modo y acciones ===========================

    def set_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        self.toolbar_manager.update_mode(mode)
        self.selection_handler.update_mode(mode)

    def on_clone_centroids(self) -> None:
        """Duplica el ImageItem seleccionado en todos los centroides detectados."""
        scene = self.viewer.scene()
        sel = scene.selectedItems()
        if not sel:
            return
        src = sel[0]
        if not isinstance(src, ImageItem):
            QMessageBox.information(self, "Selección inválida", "Selecciona un ImageItem para clonar.")
            return

        # clone_item_to_centroids ya crea y añade ítems; si quieres que cada clon sea undoable,
        # modifica clone_item_to_centroids para que devuelva los ítems creados SIN añadirlos y aquí empujar comandos.
        created = clone_item_to_centroids(
            scene=scene,
            source_item=src,
            document=self.document,
            mm_to_scene=self.mm_to_scene,
            bind_callback=self.selection_handler.bind_item,
        )
        if not created:
            QMessageBox.information(self, "Sin centroides", "No hay centroides en escena para clonar.")
            return

        # Si created ya está en escena, no los reapilamos.
        # (Opcional) podrías envolver cada uno en un AddItemCommand si cambias la función de placement.

        self._update_actions_state()

    # ============================ Clipboard ============================

    def copy_selected(self) -> None:
        items = self.selected_items()
        if not items:
            return
        md = serialize_items(items)
        QGuiApplication.clipboard().setMimeData(md)

    def paste_from_clipboard(self):
        md = QGuiApplication.clipboard().mimeData()
        items = deserialize_items(md, self.item_factory, offset=QPointF(16, 16))
        if not items:
            return

        # Si usas comando para añadir (recomendado), mantenlo:
        cmd = PasteItemsCommand(self.viewer.scene(), items, text="Paste Items")
        self.undo_stack.push(cmd)

        # 🔽 AQUI forzamos selección exclusiva de lo pegado
        scene = self.viewer.scene()
        scene.clearSelection()
        for it in items:
            try:
                it.setSelected(True)
            except Exception:
                pass


    def item_factory(self, cls_name: str, source_id: str):
        """Crea ítems al pegar. Ajusta a tu modelo real."""
        if cls_name == "ImageItem":
            # Implementa este método de clase en ImageItem para reconstruir desde source_id
            it = ImageItem.from_source_id(source_id, document=self.document, mm_to_scene=self.mm_to_scene)
            it.setSelected(True)
            self.selection_handler.bind_item(it)
            return it
        if cls_name == "CentroidItem":
            it = CentroidItem(radius_px=6.0)
            return it
        raise ValueError(f"Tipo no soportado: {cls_name}")

    # ============================ Eliminar ============================

    def delete_selected(self) -> None:
        scene = self.viewer.scene()
        # Eliminamos todos los seleccionados con comandos undoables
        for it in self.selected_items():
            cmd = RemoveItemCommand(scene, it, text="Delete Item")
            self.undo_stack.push(cmd)

    # ============================ Render de fondo ===========================

    def _set_background_pixmap(self, pixmap) -> None:
        """Coloca/actualiza el pixmap de fondo y lo escala a unidades físicas (mm)."""
        # quitar anterior
        if self._bg_item is not None:
            self.viewer.scene().removeItem(self._bg_item)
            self._bg_item = None

        # nuevo fondo
        self._bg_item = QGraphicsPixmapItem(pixmap)
        self._bg_item.setZValue(-1000)
        self._bg_item.setFlag(self._bg_item.GraphicsItemFlag.ItemIsSelectable, False)
        self._bg_item.setFlag(self._bg_item.GraphicsItemFlag.ItemIsMovable, False)
        self.viewer.scene().addItem(self._bg_item)

        # Escala física a partir de mm/px de la referencia
        ref_mpp = self.document.get_mm_per_pixel()
        if ref_mpp is not None:
            ref_mpp_x, ref_mpp_y = ref_mpp
            sx = float(ref_mpp_x) * float(self.mm_to_scene)
            sy = float(ref_mpp_y) * float(self.mm_to_scene)
            t = QTransform()
            t.scale(sx, sy)
            self._bg_item.setTransform(t, False)
        else:
            self._bg_item.setTransform(QTransform(), False)

        self._bg_item.setPos(0.0, 0.0)

        # Re-vincular selección (por si la escena cambió)
        self.selection_handler.rebind()
        self._fit_all_view()

    def _clear_background(self) -> None:
        if self._bg_item is not None:
            self.viewer.scene().removeItem(self._bg_item)
            self._bg_item = None

    def _refresh_view(self) -> None:
        """Actualiza el fondo y, si corresponde, materializa centroides en escena."""
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
                # En vista de OUTPUT no dibujamos centroides (pertenecen a la referencia)
                return

        image = self.document.get_reference_preview()
        if image is not None:
            pix = numpy_to_qpixmap(image)
            self._set_background_pixmap(pix)
            # ← IMPORTANTE: crear CentroidItem alineados al fondo de referencia
            self._populate_centroids_items()
        else:
            self._clear_background()

    def _populate_centroids_items(self) -> None:
        """
        Convierte self.document.centroids (px en referencia)
        a CentroidItem en la escena (mm de escena).
        """
        scene = self.viewer.scene()

        # Limpia centroides existentes (evita duplicados al recargar)
        for it in list(scene.items()):
            if isinstance(it, CentroidItem):
                scene.removeItem(it)

        ref_mpp = self.document.get_mm_per_pixel()
        if not ref_mpp:
            return
        mmpp_x, mmpp_y = ref_mpp
        s = float(getattr(self, "mm_to_scene", 1.0))  # 1 escena = 1 mm

        # document.centroids pueden ser dataclass o tuplas
        pts_scene = []
        for c in self.document.centroids:
            x = float(c.x) if hasattr(c, "x") else float(c[0])
            y = float(c.y) if hasattr(c, "y") else float(c[1])
            pts_scene.append(QPointF(x * mmpp_x * s, y * mmpp_y * s))

        # Crea los ítems
        for p in pts_scene:
            dot = CentroidItem(radius_px=6.0)
            dot.setPos(p)
            scene.addItem(dot)

    def _fit_all_view(self) -> None:
        scene = self.viewer.scene()
        r = scene.itemsBoundingRect()
        if r.isNull():
            return
        m = 40
        r.adjust(-m, -m, m, m)
        self.viewer.fitInView(r, Qt.KeepAspectRatio)

    # ============================ Estado/StatusBar ===========================

    def _update_actions_state(self) -> None:
        try:
            acts = self.toolbar_manager.actions
            if "save" in acts:
                acts["save"].setEnabled(self.document.has_output or bool(self.document.layers))
            if "add_item" in acts:
                acts["add_item"].setEnabled(True)
            # (Opcional) puedes reflejar disponibilidad de undo/redo:
            if "undo" in acts:
                acts["undo"].setEnabled(self.undo_stack.canUndo())
            if "redo" in acts:
                acts["redo"].setEnabled(self.undo_stack.canRedo())
        except Exception:
            pass

    def _update_status(self) -> None:
        parts: list[str] = []
        if self.document.reference_path is not None:
            parts.append(f"Referencia: {self.document.reference_path.name}")
            parts.append(f"Objetos: {len(self.document.centroids)}")

        mm_per_pixel = self.document.get_mm_per_pixel()
        if mm_per_pixel is not None:
            parts.append(
                f"Escala: {mm_per_pixel[0]:.3f} mm/px (X), {mm_per_pixel[1]:.3f} mm/px (Y)"
            )

        tile_dims = self.document.get_tile_dimensions_mm()
        if tile_dims is not None:
            parts.append(f"TIF: {tile_dims[0]:.1f} x {tile_dims[1]:.1f} mm")

        message = " | ".join(parts) if parts else "Carga una referencia JPG para comenzar."
        self.statusBar().showMessage(message)

    # ============================ Utilidades ===========================

    def current_scene(self):
        return self.viewer.scene()

    def selected_items(self):
        return [it for it in self.current_scene().selectedItems()]
