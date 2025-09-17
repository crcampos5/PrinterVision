"""Main window housing menus, toolbars, and the image viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QToolBar, QDialog, QGraphicsPixmapItem
from PySide6.QtCore import Qt


from editor_tif.models.image_document import ImageDocument
from editor_tif.utils.qt import numpy_to_qpixmap
from editor_tif.views.image_viewer import ImageViewer
from editor_tif.views.workspace_dialog import WorkspaceDialog
from editor_tif.views.scene_items import ImageItem, Layer
from editor_tif.views.properties_dock import PropertiesDock
from editor_tif.utils.io import load_image_data



class MainWindow(QMainWindow):
    """Application main window with top toolbar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PrinterVision Editor")
        self.resize(1200, 800)
        # Central canvas that handles zooming, panning, and rotation.
        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)
        self._bg_item: QGraphicsPixmapItem | None = None
        self.props = PropertiesDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.props)
        self.viewer.scene().selectionChanged.connect(self._on_selection_changed)

        self.props.posChanged.connect(self._on_props_pos)
        self.props.rotChanged.connect(self._on_props_rot)
        self.props.scaleChanged.connect(self._on_props_scale)
        self.props.opacityChanged.connect(self._on_props_opacity)

        # Data model encapsulating reference, mosaic, and workspace state. 
        self.document = ImageDocument()
        self._create_toolbar()
        self._update_actions_state()
        self._update_status()

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.settings_action = QAction("Parametros", self)
        self.settings_action.triggered.connect(self.configure_workspace)
        toolbar.addAction(self.settings_action)

        self.open_action = QAction("Abrir referencia", self)
        self.open_action.triggered.connect(self.open_image)
        toolbar.addAction(self.open_action)

        self.load_tif_action = QAction("Cargar .tif", self)
        self.load_tif_action.triggered.connect(self.load_tif)
        toolbar.addAction(self.load_tif_action)

        self.save_action = QAction("Guardar resultado", self)
        self.save_action.triggered.connect(self.save_image)
        toolbar.addAction(self.save_action)

        self.add_item_action = QAction("Agregar imagen como ítem", self)
        self.add_item_action.triggered.connect(self.add_item)
        toolbar.addAction(self.add_item_action)


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
            "Imagenes JPG (*.jpg *.jpeg);;Todos los archivos (*.*)",
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

    def load_tif(self) -> None:
        if not self.document.has_reference:
            QMessageBox.information(self, "Referencia requerida", "Carga una referencia primero.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen TIF",
            str(Path.cwd()),
            "Imagenes TIF (*.tif *.TIF);;Todos los archivos (*.*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        if not self.document.load_tile(path):
            QMessageBox.warning(self, "Error", "No se pudo cargar el mosaico .tif seleccionado.")
            return
        layer = self.document.add_layer_from_tile()
        if layer:
            self.viewer._scene.addItem(ImageItem(layer))

        self._refresh_view()
        self._update_actions_state()
        self._update_status()

    def save_image(self) -> None:
        if not self.document.has_output:
            QMessageBox.information(self, "Sin resultado", "Genera un resultado antes de guardar.")
            return
        default_name = "resultado.tif"
        if self.document.tile_path is not None:
            default_name = f"{self.document.tile_path.stem}_sobre_centroides.tif"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar imagen resultante",
            str(Path.cwd() / default_name),
            "Imagenes TIF (*.tif *.TIF)",
        )
        if not file_path:
            return
        path = Path(file_path)
        if not self.document.save_output(path):
            QMessageBox.warning(self, "Error", "No se pudo guardar la imagen resultante.")
            return
        self.statusBar().showMessage(f"Imagen guardada en: {path}")

    def _set_background_pixmap(self, pixmap) -> None:
        """Coloca/actualiza el pixmap de fondo sin afectar los ítems manipulables."""
        # eliminar fondo previo si existe
        if self._bg_item is not None:
            self.viewer._scene.removeItem(self._bg_item)
            self._bg_item = None
        # crear y poner el nuevo
        self._bg_item = QGraphicsPixmapItem(pixmap)
        self._bg_item.setZValue(-1000)  # bien al fondo
        # que no sea seleccionable ni movible
        self._bg_item.setFlag(self._bg_item.GraphicsItemFlag.ItemIsSelectable, False)
        self._bg_item.setFlag(self._bg_item.GraphicsItemFlag.ItemIsMovable, False)
        self.viewer._scene.addItem(self._bg_item)
        # ajustar vista a todo (fondo + ítems)
        self._fit_all_view()

    def _clear_background(self) -> None:
        """Quita el pixmap de fondo, conservando ítems en la escena."""
        if self._bg_item is not None:
            self.viewer._scene.removeItem(self._bg_item)
            self._bg_item = None


    def _refresh_view(self) -> None:
        """Actualiza sólo el fondo (preview) sin tocar los ítems manipulables."""
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
                return
        # Si no hay output válido, intenta mostrar la referencia/overlay
        image = self.document.get_reference_preview()
        if image is not None:
            pix = numpy_to_qpixmap(image)
            self._set_background_pixmap(pix)
        else:
            # No hay nada que mostrar como fondo (pero conservamos los ítems)
            self._clear_background()
    
    def _fit_all_view(self) -> None:
        scene = self.viewer.scene()
        r = scene.itemsBoundingRect()
        if r.isNull():
            return
        m = 40
        r.adjust(-m, -m, m, m)
        self.viewer.fitInView(r, Qt.KeepAspectRatio)

    def _update_actions_state(self) -> None:
        self.load_tif_action.setEnabled(self.document.has_reference)
        self.save_action.setEnabled(self.document.has_output)

    def _update_status(self) -> None:
        # Compose a short status message describing the current session.
        parts: list[str] = []
        if self.document.reference_path is not None:
            parts.append(f"Referencia: {self.document.reference_path.name}")
            parts.append(f"Objetos: {len(self.document.centroids)}")
        mm_per_pixel = self.document.get_mm_per_pixel()
        if mm_per_pixel is not None:
            parts.append(f"Escala: {mm_per_pixel[0]:.3f} mm/px (X), {mm_per_pixel[1]:.3f} mm/px (Y)")
        tile_dims = self.document.get_tile_dimensions_mm()
        if tile_dims is not None:
            parts.append(f"TIF: {tile_dims[0]:.1f} x {tile_dims[1]:.1f} mm")
        message = " | ".join(parts) if parts else "Carga una referencia JPG para comenzar."
        self.statusBar().showMessage(message)

    def add_item(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen (JPG/PNG/TIF)", str(Path.cwd()),
            "Imágenes (*.jpg *.jpeg *.png *.tif *.tiff);;Todos (*.*)",
        )
        if not file_path:
            return
        data = load_image_data(Path(file_path))
        if data is None:
            QMessageBox.warning(self, "Error", "No se pudo cargar la imagen.")
            return

        # crear layer y item
        self.document._layer_seq += 1
        layer = Layer(
            id=self.document._layer_seq, path=Path(file_path), pixels=data.pixels,
            photometric=data.photometric, cmyk_order=data.cmyk_order, alpha_index=data.alpha_index,
            icc_profile=getattr(data, "icc_profile", None), ink_names=getattr(data, "ink_names", None),
            x=20.0 * len(self.document.layers), y=20.0 * len(self.document.layers)
        )
        self.document.layers.append(layer)

        item = ImageItem(layer)
        self.viewer._scene.addItem(item)
        item.setSelected(True)
        self._fit_all_view()

    def _selected_item(self):
        # evitamos dependencia de helpers del viewer
        from editor_tif.views.scene_items import ImageItem
        items = self.viewer.scene().selectedItems()
        if not items:
            return None
        # si hay otros items (fondo, etc.), tomamos el primero que sea ImageItem
        for it in items:
            if isinstance(it, ImageItem):
                return it
        return None

    
    def _on_selection_changed(self):
        it = self._selected_item()
        if it is not None:
            self.props.set_layer(it.layer)
        else:
            self.props.set_layer(None)

    def _on_props_pos(self, x: float, y: float):
        it = self._selected_item()
        if it is not None:
            it.layer.x, it.layer.y = x, y
            it.sync_from_layer()

    def _on_props_rot(self, angle: float):
        it = self._selected_item()
        if it is not None:
            it.layer.rotation = angle
            it.sync_from_layer()

    def _on_props_scale(self, s: float):
        it = self._selected_item()
        if it is not None:
            it.layer.scale = s
            it.sync_from_layer()

    def _on_props_opacity(self, op: float):
        it = self._selected_item()
        if it is not None:
            it.layer.opacity = op
            it.sync_from_layer()



