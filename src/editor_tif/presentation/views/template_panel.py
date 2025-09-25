from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QRectF, QModelIndex, Signal
from PySide6.QtGui import (
    QAction,
    QIcon,
    QKeyEvent,
    QPainter,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QMenu,
    QMessageBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

# Tip: el controller debe exponer:
# - panel_model: QStandardItemModel (o se lo inyectamos aquí)
# - toggle_visibility(group, bool)
# - remove_group(group)
# - select_and_center(group)
# - scene (para render miniaturas si se desean refrescar)


class TemplatePanel(QDockWidget):
    """Dock lateral para gestionar plantillas (grupos rígidos)."""

    # Señales opcionales para integrar con MainWindow si hiciera falta
    groupVisibilityToggled = Signal(object, bool)  # (group, visible)
    groupRemoved = Signal(object)                  # (group)
    groupActivated = Signal(object)                # (group) doble clic / Enter

    def __init__(self, controller=None, parent=None):
        super().__init__("Plantillas", parent)
        self.setObjectName("TemplatePanel")
        self._controller = controller

        # UI básica
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)

        self.view = QTreeView(w)
        self.view.setRootIsDecorated(False)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)

        # Modelo de 3 columnas: [thumb, name, visible]
        self.model = QStandardItemModel(0, 3, self)
        self.model.setHorizontalHeaderLabels(["", "Nombre", "Visible"])

        self.view.setModel(self.model)
        self.view.setColumnWidth(0, 36)
        self.view.setColumnWidth(1, 180)
        self.view.setColumnWidth(2, 70)

        lay.addWidget(self.view)
        self.setWidget(w)

        # Conexiones
        self.model.itemChanged.connect(self._on_item_changed)
        self.view.doubleClicked.connect(self._on_double_clicked)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._on_context_menu)

        # Si nos pasan el controller, lo enlazamos
        if self._controller is not None:
            self._bind_controller(self._controller)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_controller(self, controller):
        self._controller = controller
        self._bind_controller(controller)

    def controller(self):
        return self._controller

    def add_group_row(self, group, name: str, icon: Optional[QIcon] = None):
        """
        Inserta una fila (thumb, nombre, visible) apuntando a 'group'.
        Si no viene icono, intenta renderizar una miniatura desde la escena.
        """
        if icon is None:
            icon = self._render_group_icon(group)

        it_icon = QStandardItem()
        it_icon.setEditable(False)
        if icon is not None:
            it_icon.setIcon(icon)
        it_icon.setData(group, role=Qt.UserRole + 1)

        it_name = QStandardItem(name or "Plantilla")
        it_name.setEditable(True)
        it_name.setData(group, role=Qt.UserRole + 1)

        it_vis = QStandardItem()
        it_vis.setCheckable(True)
        it_vis.setCheckState(Qt.Checked)
        it_vis.setData(group, role=Qt.UserRole + 1)

        self.model.appendRow([it_icon, it_name, it_vis])

    def remove_selected_rows(self):
        """Elimina las filas seleccionadas (y sus grupos asociados)."""
        sel = self.view.selectionModel().selectedRows()
        if not sel:
            return

        # Confirmación (opcional)
        resp = QMessageBox.question(
            self,
            "Eliminar plantillas",
            f"¿Eliminar {len(sel)} plantilla(s) seleccionada(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        rows = sorted((idx.row() for idx in sel), reverse=True)
        for r in rows:
            group = self._group_from_row(r)
            if group is not None:
                # avisa al controller para limpiar escena y estructuras
                if self._controller is not None:
                    self._controller.remove_group(group)
                self.groupRemoved.emit(group)
            # (la fila será removida por el controller.remove_group; si no, lo forzamos)
            if r < self.model.rowCount():
                self.model.removeRow(r)

    def select_row_for_group(self, group):
        """Selecciona la fila del panel asociada a 'group'."""
        row = self._row_for_group(group)
        if row is None:
            return
        idx = self.model.index(row, 1)  # columna nombre
        self.view.selectionModel().clearSelection()
        self.view.selectionModel().select(idx, self.view.selectionModel().Select | self.view.selectionModel().Rows)
        self.view.scrollTo(idx)

    def refresh_group_icon(self, group):
        """Re-renderiza la miniatura de 'group'."""
        row = self._row_for_group(group)
        if row is None:
            return
        icon = self._render_group_icon(group)
        if icon is None:
            return
        it = self.model.item(row, 0)
        if it:
            it.setIcon(icon)

    # ---------------------------------------------------------------------
    # Internos
    # ---------------------------------------------------------------------
    def _bind_controller(self, controller):
        # Si el controller quiere que usemos su model, lo adoptamos
        if getattr(controller, "panel_model", None) is not None:
            # OJO: si ya había filas aquí, podrías migrarlas; en la práctica,
            # lo más simple es que el controller use ESTE model desde el inicio.
            self.model = controller.panel_model
            self.view.setModel(self.model)
            self.model.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item: QStandardItem):
        # Columna visibilidad
        if item.column() == 2:
            group = item.data(Qt.UserRole + 1)
            if group is None:
                return
            visible = (item.checkState() == Qt.Checked)
            # Propaga
            if self._controller is not None:
                self._controller.toggle_visibility(group, visible)
            self.groupVisibilityToggled.emit(group, visible)

    def _on_double_clicked(self, index: QModelIndex):
        # Doble clic -> centra en escena y selecciona
        row = index.row()
        group = self._group_from_row(row)
        if group is None:
            return
        if self._controller is not None:
            self._controller.select_and_center(group)
        self.groupActivated.emit(group)

    def _on_context_menu(self, pos):
        idx = self.view.indexAt(pos)
        menu = QMenu(self)

        act_show = QAction("Mostrar", self)
        act_hide = QAction("Ocultar", self)
        act_remove = QAction("Eliminar", self)

        act_show.triggered.connect(lambda: self._set_selected_visibility(True))
        act_hide.triggered.connect(lambda: self._set_selected_visibility(False))
        act_remove.triggered.connect(self.remove_selected_rows)

        # Habilita/Deshabilita según selección
        has_sel = len(self.view.selectionModel().selectedRows()) > 0
        act_show.setEnabled(has_sel)
        act_hide.setEnabled(has_sel)
        act_remove.setEnabled(has_sel)

        menu.addAction(act_show)
        menu.addAction(act_hide)
        menu.addSeparator()
        menu.addAction(act_remove)

        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _set_selected_visibility(self, visible: bool):
        """Marca/Desmarca visibilidad y notifica al controller."""
        rows = self.view.selectionModel().selectedRows()
        for idx in rows:
            it_vis = self.model.item(idx.row(), 2)
            if it_vis is None:
                continue
            it_vis.setCheckState(Qt.Checked if visible else Qt.Unchecked)
            # _on_item_changed disparará la notificación al controller

    def _row_for_group(self, group) -> Optional[int]:
        for r in range(self.model.rowCount()):
            it = self.model.item(r, 1)  # nombre
            if it and it.data(Qt.UserRole + 1) is group:
                return r
        return None

    def _group_from_row(self, row: int):
        it = self.model.item(row, 1)
        return it.data(Qt.UserRole + 1) if it else None

    def _render_group_icon(self, group) -> Optional[QIcon]:
        """Genera una miniatura 80x80 renderizando el grupo desde la escena."""
        try:
            sc = group.scene()
            if sc is None:
                return None
            br_scene: QRectF = group.mapRectToScene(group.boundingRect())
            if br_scene.width() <= 0 or br_scene.height() <= 0:
                return None
            pm = QPixmap(80, 80)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)
            sc.render(p, QRectF(0, 0, 80, 80), br_scene)
            p.end()
            return QIcon(pm)
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Manejo del teclado: Supr = eliminar
    # ---------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_selected_rows()
            event.accept()
            return
        super().keyPressEvent(event)
