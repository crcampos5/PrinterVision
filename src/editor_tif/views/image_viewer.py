# editor_tif/views/image_viewer.py

from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Fondo (pixmap de referencia/output)
        self._bg_item: QGraphicsPixmapItem | None = None

        # Configuración de interacción
        self.setRenderHints(self.renderHints() |
                            QPainter.Antialiasing |
                            QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        # Parámetros de zoom
        self._zoom_factor = 1.25   # cuánto aumenta/disminuye por paso

    # ---------- Compatibilidad con código antiguo ----------
    def set_pixmap(self, pixmap):
        """Muestra un único pixmap de fondo (preview de referencia o output),
        sin interferir con los ImageItem de la escena."""
        if self._bg_item is not None:
            self._scene.removeItem(self._bg_item)
            self._bg_item = None

        self._bg_item = QGraphicsPixmapItem(pixmap)
        self._bg_item.setZValue(-1000)  # detrás de los ítems manipulables
        self._bg_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._bg_item.setFlag(QGraphicsItem.ItemIsMovable, False)
        self._scene.addItem(self._bg_item)
        self.fit_all()

    def clear(self):
        """Compat: limpiar sólo el fondo y items (mantén reset de transform)."""
        self.clear_all()

    # ---------- Utilidades ----------
    def clear_all(self):
        """Elimina todos los ítems de la escena (incluye fondo)."""
        self._scene.clear()
        self._bg_item = None
        self.resetTransform()

    def fit_all(self):
        """Ajusta la vista a todos los ítems de la escena."""
        r = self._scene.itemsBoundingRect()
        if not r.isNull():
            m = 40
            r.adjust(-m, -m, m, m)
            self.fitInView(r, Qt.KeepAspectRatio)

    def scene(self) -> QGraphicsScene:
        return self._scene

    # ---------- Zoom con la rueda ----------
    def wheelEvent(self, event):
        # Zoom SOLO si el usuario mantiene Ctrl
        if event.modifiers() & Qt.ControlModifier:
            zoom = self._zoom_factor if event.angleDelta().y() > 0 else 1 / self._zoom_factor
            self.scale(zoom, zoom)
            event.accept()
            return

        # Sin Ctrl: comportamiento normal (scroll / zoom del touchpad si aplica)
        super().wheelEvent(event)
