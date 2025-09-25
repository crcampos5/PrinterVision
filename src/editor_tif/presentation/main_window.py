from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QUndoStack
from PySide6.QtWidgets import QMainWindow

from editor_tif.domain.models.image_document import ImageDocument
from editor_tif.presentation.views.image_viewer import ImageViewer
from editor_tif.presentation.views.properties_dock import PropertiesDock
from editor_tif.presentation.views.toolbar_manager import ToolbarManager
from editor_tif.presentation.views.selection_handler import SelectionHandler
from editor_tif.presentation.modes import EditorMode

# Panel de plantillas en /views
from editor_tif.presentation.views.template_panel import TemplatePanel

# Controlador de plantillas (crea grupos rígidos y aplica clonación)
from editor_tif.features.template_controller import TemplateController

# Acciones centralizadas
from editor_tif.presentation.controllers.main_actions import MainActions


class MainWindow(QMainWindow):
    """UI contenedora — delega lógica en MainActions y mantiene wiring de UI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PrinterVision Editor")
        self.resize(1280, 840)

        # Escala: 1 mm == 1 unidad de escena (ajústalo si lo necesitas)
        self.mm_to_scene = 1.0

        # ---------- Centro: Viewer ----------
        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)

        # ---------- Docks ----------
        self.props = PropertiesDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.props)

        self.template_panel = TemplatePanel(controller=None, parent=self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.template_panel)

        # ---------- Infraestructura/UI ----------
        self.selection_handler = SelectionHandler(self)
        self.undo_stack = QUndoStack(self)
        self._install_shortcuts()

        # Documento y controlador de plantillas
        self.document = ImageDocument()
        self.template_controller = TemplateController(
            scene=self.viewer.scene(),
            document=self.document,
            mm_to_scene=self.mm_to_scene,
            get_selection=lambda: self.viewer.scene().selectedItems(),
            bind_callback=getattr(self.selection_handler, "bind_item", None),
        )

        # Vincular panel ↔ controlador (el panel aporta el model)
        self.template_panel.set_controller(self.template_controller)
        setattr(self.template_controller, "panel_model", self.template_panel.model)

        # Acciones centralizadas
        self.toolbar_manager = ToolbarManager(self, attach_to_window=self)
        self.actions = MainActions(
            window=self,
            scene=self.viewer.scene(),
            document=self.document,
            mm_to_scene=self.mm_to_scene,
            template_controller=self.template_controller,
            selection_handler=self.selection_handler,
            undo_stack=self.undo_stack,
            toolbar_manager=self.toolbar_manager,
        )

        # Conectar toolbar a métodos finos del MainWindow (que delegan a MainActions)
        self.toolbar_manager.bind_actions(
            configure_workspace=self.configure_workspace,
            open_image=self.open_image,
            save_image=self.save_image,
            add_item=self.add_item,
            set_mode=self.set_mode,
            clone_centroids=self.on_clone_centroids,
            create_template=self.on_create_template,
            apply_template_all=self.on_apply_template_all,
            apply_template_selected=self.on_apply_template_selected,
        )

        # ---------- Conexiones ----------
        self.viewer.scene().selectionChanged.connect(self.selection_handler.on_selection_changed)
        self.props.posChanged.connect(self.selection_handler.on_props_pos)
        self.props.rotChanged.connect(self.selection_handler.on_props_rot)
        self.props.scaleChanged.connect(self.selection_handler.on_props_scale)
        self.props.opacityChanged.connect(self.selection_handler.on_props_opacity)

        # Estado inicial
        self.set_mode(EditorMode.CloneByCentroid)

    # ===================== Atajos globales =====================
    def _install_shortcuts(self) -> None:
        act_undo = QAction("Deshacer", self, shortcut=QKeySequence.Undo, triggered=self.undo_stack.undo)
        act_redo = QAction("Rehacer", self, shortcut=QKeySequence.Redo, triggered=self.undo_stack.redo)
        act_delete = QAction("Eliminar", self, shortcut=QKeySequence.Delete, triggered=self.delete_selected)
        act_copy = QAction("Copiar", self, shortcut=QKeySequence.Copy, triggered=self.copy_selected)
        act_paste = QAction("Pegar", self, shortcut=QKeySequence.Paste, triggered=self.paste_from_clipboard)
        for a in (act_undo, act_redo, act_delete, act_copy, act_paste):
            self.addAction(a)

    # ===================== Delegación a MainActions =====================
    def configure_workspace(self): self.actions.configure_workspace()
    def open_image(self): self.actions.open_image()
    def save_image(self): self.actions.save_image()
    def add_item(self): self.actions.add_item()

    # Plantillas
    def on_clone_centroids(self): self.actions.on_clone_centroids()
    def on_create_template(self): self.actions.on_create_template()
    def on_apply_template_all(self): self.actions.on_apply_template_all()
    def on_apply_template_selected(self): self.actions.on_apply_template_selected()

    # Edición básica
    def delete_selected(self): self.actions.delete_selected()
    def copy_selected(self): self.actions.copy_selected()
    def paste_from_clipboard(self): self.actions.paste_from_clipboard()

    # Modo de trabajo
    def set_mode(self, mode: EditorMode) -> None:
        self.actions.set_mode(mode)
