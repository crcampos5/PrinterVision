# src/editor_tif/presentation/views/template_items.py
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QPainter, QPen, QPainterPath
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem

class TemplateGroupItem(QGraphicsObject):
    """Agrupa un ImageItem y su ContourItem. Se mueven juntos; hijos no movibles/seleccionables."""
    changed = Signal(object)  # emite self cuando cambian pos/rot/escala

    def __init__(self, image_item, contour_item, contour_color, parent=None):
        super().__init__(parent)
        # IMPORTANTE: usar QGraphicsItem.GraphicsItemFlag
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self._image = image_item
        self._contour = contour_item
        self._pen = QPen(contour_color)
        self._pen.setWidthF(2.0)
        self._pen.setCosmetic(True)  # grosor constante con zoom

        self.sel_pen = QPen(Qt.green)
        self.sel_pen.setCosmetic(True)
        self.sel_pen.setStyle(Qt.PenStyle.DashLine)
        self.sel_pen.setWidthF(1.2)

        self._paint_pad = max(self._pen.widthF(), self.sel_pen.widthF()) * 0.5 + 2.0

        # Hacer a los hijos “pasivos”
        for child in (self._image, self._contour):
            child.setParentItem(self)
            child.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            child.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            child.setAcceptedMouseButtons(Qt.NoButton)

        # (Opcional) Re-centrar el grupo al centro del contorno
        try:
            c = self._contour.boundingRect().center()
            self._image.setPos(self._image.pos() - c)
            self._contour.setPos(self._contour.pos() - c)
            self.setPos(self.pos() + c)
        except Exception:
            pass

    def boundingRect(self) -> QRectF:
        # Abarca exactamente a los hijos
        r = self.childrenBoundingRect()
        pad = self._paint_pad
        return r.adjusted(-pad, -pad, pad, pad)

    def paint(self, p: QPainter, option, widget=None):
        p.save()
        try:
            # El path del contorno
            path = self._contour.mapToParent(self._contour.shape())
        except Exception:
            path = QPainterPath()
            path.addRect(self._contour.mapRectToParent(self._contour.boundingRect()))

        # Dibujar siempre el contorno base en color de plantilla
        p.setPen(self._pen)
        p.drawPath(path)

        # Si está seleccionado, dibujar borde punteado (igual que otros items de Qt)
        if self.isSelected():
            
            
            rect = self.boundingRect()
            margin = 0.5
            rect = rect.adjusted(-margin, -margin, margin, margin)
            p.setPen(self.sel_pen)
            p.drawRect(rect)

        p.restore()


    def itemChange(self, change, value):
        # Notificar cambios relevantes
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged,
        ):
            self.changed.emit(self)
        return super().itemChange(change, value)

    # Helpers opcionales
    def image_item(self):
        return self._image

    def contour_item(self):
        return self._contour
