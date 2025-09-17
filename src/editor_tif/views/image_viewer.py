# imports (agrega QGraphicsPixmapItem)
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsPixmapItem

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        # ...
        self._bg_item: QGraphicsPixmapItem | None = None  # <— NUEVO

    # ---------- Compatibilidad con código antiguo ----------
    def set_pixmap(self, pixmap):
        """Muestra un único pixmap de fondo (preview de referencia o output),
        sin interferir con los ImageItem de la escena."""
        # elimina fondo previo
        if self._bg_item is not None:
            self._scene.removeItem(self._bg_item)
            self._bg_item = None
        # crea y agrega fondo nuevo
        self._bg_item = QGraphicsPixmapItem(pixmap)
        self._bg_item.setZValue(-1000)                     # detrás de los ítems manipulables
        self._bg_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._bg_item.setFlag(QGraphicsItem.ItemIsMovable, False)
        self._scene.addItem(self._bg_item)
        self.fit_all()

    def clear(self):
        """Compat: limpiar sólo el fondo y items (mantén reset de transform)."""
        self.clear_all()
