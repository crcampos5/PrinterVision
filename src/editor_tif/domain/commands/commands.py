# src/editor_tif/commands.py
from PySide6.QtGui import QUndoCommand
from PySide6.QtCore import QPointF

class AddItemCommand(QUndoCommand):
    def __init__(self, scene, item, pos=None, text="Add Item"):
        super().__init__(text)
        self.scene = scene
        self.item = item
        self.pos = QPointF(pos) if pos is not None else None
        self._done = False

    def redo(self):
        if self.pos is not None:
            self.item.setPos(self.pos)
        if self.item.scene() is None:
            self.scene.addItem(self.item)
        self._done = True

    def undo(self):
        if self._done and self.item.scene() is self.scene:
            self.scene.removeItem(self.item)

class RemoveItemCommand(QUndoCommand):
    def __init__(self, scene, item, text="Remove Item"):
        super().__init__(text)
        self.scene = scene
        self.item = item
        self.parentItem = item.parentItem()
        self._pos = item.pos()
        self._z = item.zValue()

    def redo(self):
        if self.item.scene() is self.scene:
            self.scene.removeItem(self.item)

    def undo(self):
        if self.item.scene() is None:
            if self.parentItem:
                self.item.setParentItem(self.parentItem)
            self.scene.addItem(self.item)
            self.item.setPos(self._pos)
            self.item.setZValue(self._z)

class TransformItemCommand(QUndoCommand):
    """Úsalo para mover/rotar/escalar una sola acción atómica."""
    def __init__(self, item, old_pos, new_pos, old_rot=None, new_rot=None,
                 old_scale=None, new_scale=None, text="Transform Item"):
        super().__init__(text)
        self.item = item
        self.old_pos, self.new_pos = old_pos, new_pos
        self.old_rot, self.new_rot = old_rot, new_rot
        self.old_scale, self.new_scale = old_scale, new_scale

    def _apply(self, pos, rot, scale):
        self.item.setPos(pos)
        if rot is not None:
            self.item.setRotation(rot)
        if scale is not None:
            self.item.setScale(scale)

    def redo(self):
        self._apply(self.new_pos, self.new_rot, self.new_scale)

    def undo(self):
        self._apply(self.old_pos, self.old_rot, self.old_scale)

class PasteItemsCommand(QUndoCommand):
    """Pega múltiples ítems (ya construidos) en la escena en un offset."""
    def __init__(self, scene, items, text="Paste Items"):
        super().__init__(text)
        self.scene = scene
        self.items = items
        self._added = False

    def redo(self):
        if not self._added:
            for it in self.items:
                if it.scene() is None:
                    self.scene.addItem(it)
            self._added = True
        else:
            for it in self.items:
                if it.scene() is None:
                    self.scene.addItem(it)

    def undo(self):
        for it in self.items:
            if it.scene() is self.scene:
                self.scene.removeItem(it)
