from pathlib import Path
from PySide6.QtWidgets import QToolBar
from PySide6.QtGui import QAction, QKeySequence, QIcon, QActionGroup
from PySide6.QtCore import Qt
from editor_tif.presentation.modes import EditorMode

ICON_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"

def _icon(name: str) -> QIcon:
    """
    Intenta cargar desde tu qrc (p.ej. :/icons/open.svg).
    Si no existe, devuelve un icono vacío y no rompe.
    """
    path = ICON_DIR / name
    try:
        return QIcon(str(path))
    except Exception:
        return QIcon()

class ToolbarManager:
    """
    Barra principal alineada al nuevo flujo:
      Parámetros | Abrir referencia | Guardar resultado | Agregar imagen como ítem
      [Modo] Centroide / Plantilla
      (Centroide) Clonar en centroides
      (Plantilla) Crear plantilla | Aplicar (todos) | Aplicar (selección)
    """
    def __init__(self, main_window):
        self.main_window = main_window

        self.toolbar = QToolBar("Main Toolbar", main_window)
        self.toolbar.setObjectName("MainToolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        main_window.addToolBar(self.toolbar)

        self._create_core_actions()
        self._create_mode_actions()
        self._create_context_actions()

        # Estado inicial
        self.update_mode(EditorMode.CloneByCentroid)
        self.set_clone_enabled(False)
        self.set_template_enabled(create=False, apply_all=False, apply_sel=False)

    # --------------------------------------------------------------------- core
    def _create_core_actions(self):
        self.act_settings = QAction(_icon("settings.svg"), "Parámetros", self.main_window)
        self.act_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.act_settings.setStatusTip("Configurar espacio de trabajo")
        self.act_settings.triggered.connect(self.main_window.configure_workspace)
        self.toolbar.addAction(self.act_settings)

        self.act_open_ref = QAction(_icon("open.svg"), "Abrir referencia", self.main_window)
        self.act_open_ref.setShortcut(QKeySequence.Open)  # Ctrl+O
        self.act_open_ref.setStatusTip("Abrir imagen de referencia")
        self.act_open_ref.triggered.connect(self.main_window.open_image)
        self.toolbar.addAction(self.act_open_ref)

        self.act_save = QAction(_icon("save.svg"), "Guardar resultado", self.main_window)
        self.act_save.setShortcut(QKeySequence.Save)  # Ctrl+S
        self.act_save.setStatusTip("Guardar imagen compuesta")
        self.act_save.triggered.connect(self.main_window.save_image)
        self.toolbar.addAction(self.act_save)

        self.act_add_item = QAction(_icon("add-image.svg"), "Agregar imagen como ítem", self.main_window)
        self.act_add_item.setShortcut(QKeySequence("Ctrl+I"))
        self.act_add_item.setStatusTip("Insertar una imagen en la escena como ítem")
        self.act_add_item.triggered.connect(self.main_window.add_item)
        self.toolbar.addAction(self.act_add_item)

        self.toolbar.addSeparator()

    # ---------------------------------------------------------------------- modo
    def _create_mode_actions(self):
        self.mode_group = QActionGroup(self.toolbar)
        self.mode_group.setExclusive(True)

        self.act_mode_centroid = QAction(_icon("target.svg"), "Centroide", self.toolbar, checkable=True)
        self.act_mode_centroid.setShortcut(QKeySequence("Alt+1"))
        self.act_mode_centroid.setStatusTip("Modo de clonado por centroides")
        self.mode_group.addAction(self.act_mode_centroid)

        self.act_mode_template = QAction(_icon("template.svg"), "Plantilla", self.toolbar, checkable=True)
        self.act_mode_template.setShortcut(QKeySequence("Alt+2"))
        self.act_mode_template.setStatusTip("Modo de creación/aplicación de plantillas")
        self.mode_group.addAction(self.act_mode_template)

        # Por defecto arrancamos en Centroide
        self.act_mode_centroid.setChecked(True)

        # Cambios de modo → delega en MainWindow
        self.act_mode_centroid.toggled.connect(
            lambda checked: checked and self.main_window.set_mode(EditorMode.CloneByCentroid)
        )
        self.act_mode_template.toggled.connect(
            lambda checked: checked and self.main_window.set_mode(EditorMode.Template)
        )

        self.toolbar.addAction(self.act_mode_centroid)
        self.toolbar.addAction(self.act_mode_template)
        self.toolbar.addSeparator()

    # --------------------------------------------------------------- contextuales
    def _create_context_actions(self):
        # --- Modo Centroide ---
        self.act_clone_centroids = QAction(_icon("clone.svg"), "Clonar en centroides", self.toolbar)
        self.act_clone_centroids.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.act_clone_centroids.setStatusTip("Duplicar ítem seleccionado a todos los centroides")
        self.act_clone_centroids.triggered.connect(self.main_window.on_clone_centroids)
        self.toolbar.addAction(self.act_clone_centroids)

        # --- Modo Plantilla ---
        self.act_tpl_create = QAction(_icon("template-add.svg"), "Crear plantilla", self.toolbar)
        self.act_tpl_create.setShortcut(QKeySequence("Ctrl+T"))
        self.act_tpl_create.setStatusTip("Crear plantilla desde la selección (1 ítem + 1 centroide)")
        self.act_tpl_create.triggered.connect(self.main_window.on_create_template)
        self.toolbar.addAction(self.act_tpl_create)

        self.act_tpl_apply_all = QAction(_icon("template-apply-all.svg"), "Aplicar (todos)", self.toolbar)
        self.act_tpl_apply_all.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.act_tpl_apply_all.setStatusTip("Aplicar plantilla a todos los centroides")
        self.act_tpl_apply_all.triggered.connect(self.main_window.on_apply_template_all)
        self.toolbar.addAction(self.act_tpl_apply_all)

        self.act_tpl_apply_sel = QAction(_icon("template-apply-sel.svg"), "Aplicar (selección)", self.toolbar)
        self.act_tpl_apply_sel.setShortcut(QKeySequence("Ctrl+Alt+T"))
        self.act_tpl_apply_sel.setStatusTip("Aplicar plantilla a los centroides seleccionados")
        self.act_tpl_apply_sel.triggered.connect(self.main_window.on_apply_template_selected)
        self.toolbar.addAction(self.act_tpl_apply_sel)

    # ------------------------------------------------------------------- públicos
    def update_mode(self, mode: EditorMode):
        """Sincroniza checks/visibilidad con el modo actual."""
        is_centroid = (mode == EditorMode.CloneByCentroid)
        # Evita retroalimentación recursiva de toggled:
        if self.act_mode_centroid.isChecked() != is_centroid:
            self.act_mode_centroid.setChecked(is_centroid)
        if self.act_mode_template.isChecked() == is_centroid:
            self.act_mode_template.setChecked(not is_centroid)

        # Visibilidad contextual
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
        - create: requiere 1 ImageItem + 1 CentroidItem seleccionados
        - apply_all: requiere que exista al menos una plantilla activa
        - apply_sel: requiere plantilla activa + centroides seleccionados
        """
        self.act_tpl_create.setEnabled(create)
        self.act_tpl_apply_all.setEnabled(apply_all)
        self.act_tpl_apply_sel.setEnabled(apply_sel)

    # Accesos útiles desde fuera (opcional)
    @property
    def actions(self):
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
