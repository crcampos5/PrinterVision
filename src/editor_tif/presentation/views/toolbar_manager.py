from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtWidgets import QToolBar

from editor_tif.presentation.modes import EditorMode

ICON_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"

def _icon(name: str) -> QIcon:
    """
    Intenta cargar desde /resources/icons. Si no existe, devuelve un icono vacío.
    """
    try:
        return QIcon(str(ICON_DIR / name))
    except Exception:
        return QIcon()


class ToolbarManager(QToolBar):
    """
    Toolbar desacoplada de la lógica. Expone:
      - bind_actions(...): para conectar acciones a callables (p.ej., métodos finos del MainWindow).
      - update_mode(EditorMode)
      - set_clone_enabled(bool)
      - set_template_enabled(create=..., apply_all=..., apply_sel=...)

    Diseño:
      [Parámetros] [Abrir referencia] [Guardar resultado] [Agregar ítem]
      [Modo] Centroide | Plantilla
      (Centroide) [Clonar en centroides]
      (Plantilla) [Crear plantilla] [Aplicar (todos)] [Aplicar (selección)]
    """

    def __init__(self, parent=None, *, attach_to_window: Optional[object] = None):
        super().__init__("Herramientas", parent)
        self.setObjectName("MainToolbar")
        self.setMovable(False)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Acciones núcleo
        self._create_core_actions()
        # Acciones de modo
        self._create_mode_actions()
        # Acciones contextuales
        self._create_context_actions()

        # Estado por defecto
        self.update_mode(EditorMode.CloneByCentroid)
        self.set_clone_enabled(False)
        self.set_template_enabled(create=False, apply_all=False, apply_sel=False)

        # Opcional: adjuntar directamente a una ventana (QMainWindow)
        if attach_to_window is not None and hasattr(attach_to_window, "addToolBar"):
            attach_to_window.addToolBar(self)

    # ------------------------------------------------------------------ Bindings
    def bind_actions(
        self,
        *,
        configure_workspace: Callable,   # p.ej. MainWindow.configure_workspace
        open_image: Callable,
        save_image: Callable,
        add_item: Callable,
        set_mode: Callable[[EditorMode], None],
        clone_centroids: Callable,
        create_template: Callable,
        apply_template_all: Callable,
        apply_template_selected: Callable,
    ) -> None:
        """
        Conecta las señales triggered/toggled a los callables que definas.
        Normalmente vas a pasar los métodos finos expuestos por MainWindow,
        que a su vez delegan en MainActions. (Ver main_window.py reescrito)
        """
        # Núcleo
        self.act_settings.triggered.connect(configure_workspace)
        self.act_open_ref.triggered.connect(open_image)
        self.act_save.triggered.connect(save_image)
        self.act_add_item.triggered.connect(add_item)

        # Modos
        self.act_mode_centroid.toggled.connect(
            lambda checked: checked and set_mode(EditorMode.CloneByCentroid)
        )
        self.act_mode_template.toggled.connect(
            lambda checked: checked and set_mode(EditorMode.Template)
        )

        # Contextuales
        self.act_clone_centroids.triggered.connect(clone_centroids)
        self.act_tpl_create.triggered.connect(create_template)
        self.act_tpl_apply_all.triggered.connect(apply_template_all)
        self.act_tpl_apply_sel.triggered.connect(apply_template_selected)

    # --------------------------------------------------------------- Core actions
    def _create_core_actions(self):
        self.act_settings = QAction(_icon("settings.svg"), "Parámetros", self)
        self.act_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.act_settings.setStatusTip("Configurar espacio de trabajo")
        self.addAction(self.act_settings)

        self.act_open_ref = QAction(_icon("open.svg"), "Abrir referencia", self)
        self.act_open_ref.setShortcut(QKeySequence.Open)
        self.act_open_ref.setStatusTip("Abrir imagen de referencia")
        self.addAction(self.act_open_ref)

        self.act_save = QAction(_icon("save.svg"), "Guardar resultado", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.setStatusTip("Guardar imagen compuesta")
        self.addAction(self.act_save)

        self.act_add_item = QAction(_icon("add-image.svg"), "Agregar imagen como ítem", self)
        self.act_add_item.setShortcut(QKeySequence("Ctrl+I"))
        self.act_add_item.setStatusTip("Insertar una imagen en la escena como ítem")
        self.addAction(self.act_add_item)

        self.addSeparator()

    # --------------------------------------------------------------- Mode actions
    def _create_mode_actions(self):
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)

        self.act_mode_centroid = QAction(_icon("target.svg"), "Centroide", self, checkable=True)
        self.act_mode_centroid.setShortcut(QKeySequence("Alt+1"))
        self.act_mode_centroid.setStatusTip("Modo de clonado por centroides")
        self.mode_group.addAction(self.act_mode_centroid)

        self.act_mode_template = QAction(_icon("template.svg"), "Plantilla", self, checkable=True)
        self.act_mode_template.setShortcut(QKeySequence("Alt+2"))
        self.act_mode_template.setStatusTip("Modo de creación/aplicación de plantillas")
        self.mode_group.addAction(self.act_mode_template)

        # Arrancamos en Centroide
        self.act_mode_centroid.setChecked(True)

        self.addAction(self.act_mode_centroid)
        self.addAction(self.act_mode_template)
        self.addSeparator()

    # ----------------------------------------------------------- Context actions
    def _create_context_actions(self):
        # --- Modo Centroide ---
        self.act_clone_centroids = QAction(_icon("clone.svg"), "Clonar en centroides", self)
        self.act_clone_centroids.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.act_clone_centroids.setStatusTip("Duplicar ítem seleccionado a todos los centroides")
        self.addAction(self.act_clone_centroids)

        # --- Modo Plantilla ---
        self.act_tpl_create = QAction(_icon("template-add.svg"), "Crear plantilla", self)
        self.act_tpl_create.setShortcut(QKeySequence("Ctrl+T"))
        self.act_tpl_create.setStatusTip("Crear plantilla desde la selección (1 ítem + 1 contorno)")
        self.addAction(self.act_tpl_create)

        self.act_tpl_apply_all = QAction(_icon("template-apply-all.svg"), "Aplicar (todos)", self)
        self.act_tpl_apply_all.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.act_tpl_apply_all.setStatusTip("Aplicar plantilla a todos los marcadores")
        self.addAction(self.act_tpl_apply_all)

        self.act_tpl_apply_sel = QAction(_icon("template-apply-sel.svg"), "Aplicar (selección)", self)
        self.act_tpl_apply_sel.setShortcut(QKeySequence("Ctrl+Alt+T"))
        self.act_tpl_apply_sel.setStatusTip("Aplicar plantilla a los marcadores seleccionados")
        self.addAction(self.act_tpl_apply_sel)

    # --------------------------------------------------------------- Public API
    def update_mode(self, mode: EditorMode):
        """Sincroniza checks y visibilidad contextual con el modo actual."""
        is_centroid = (mode == EditorMode.CloneByCentroid)
        if self.act_mode_centroid.isChecked() != is_centroid:
            self.act_mode_centroid.setChecked(is_centroid)
        if self.act_mode_template.isChecked() == is_centroid:
            self.act_mode_template.setChecked(not is_centroid)

        # Mostrar/ocultar acciones según el modo
        self.act_clone_centroids.setVisible(is_centroid)

        self.act_tpl_create.setVisible(not is_centroid)
        self.act_tpl_apply_all.setVisible(not is_centroid)
        self.act_tpl_apply_sel.setVisible(not is_centroid)

    def set_clone_enabled(self, enabled: bool):
        """Habilita/Deshabilita el botón 'Clonar en centroides'."""
        self.act_clone_centroids.setEnabled(enabled)

    def set_template_enabled(self, *, create: bool, apply_all: bool, apply_sel: bool):
        """
        Habilita/Deshabilita acciones del modo Plantilla.
        - create: 1 ImageItem + 1 ContourItem seleccionados
        - apply_all: existe al menos una plantilla activa
        - apply_sel: plantilla activa + marcadores seleccionados
        """
        self.act_tpl_create.setEnabled(create)
        self.act_tpl_apply_all.setEnabled(apply_all)
        self.act_tpl_apply_sel.setEnabled(apply_sel)

    @property
    def actions(self):
        """Acceso opcional a las QAction por nombre."""
        return {
            "settings": self.act_settings,
            "open_ref": self.act_open_ref,
            "save": self.act_save,
            "add_item": self.act_add_item,
            "mode_centroid": self.act_mode_centroid,
            "mode_template": self.act_mode_template,
            "clone_centroids": self.act_clone_centroids,
            "tpl_create": self.act_tpl_create,
            "tpl_apply_all": self.act_tpl_apply_all,
            "tpl_apply_sel": self.act_tpl_apply_sel,
        }
