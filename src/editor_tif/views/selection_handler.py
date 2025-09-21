# editor_tif/views/selection_handler.py
from __future__ import annotations

from PySide6.QtCore import QObject, Qt
from shiboken6 import Shiboken

from editor_tif.views.scene_items import ImageItem
from editor_tif.utils.modes import EditorMode
from editor_tif.commands import TransformItemCommand


# -----------------------------
# Helpers
# -----------------------------
def _is_valid(obj) -> bool:
    return (obj is not None) and Shiboken.isValid(obj)

def _addr(obj) -> int:
    try:
        return int(Shiboken.getCppPointer(obj)[0])
    except Exception:
        return id(obj)


class SelectionHandler(QObject):
    """
    - Sincroniza el panel de propiedades con la selección.
    - Empuja TransformItemCommand al recibir item.events.committed (mover/rotar/escalar).
    - Habilita/inhabilita acciones según modo.
    """
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main = main_window
        self._viewer = None
        self._scene = None
        self._mode: EditorMode = EditorMode.Idle

        # Para evitar dobles conexiones a item.events
        self._bound_event_ids: set[int] = set()
        # Snapshot previo por ítem: (pos, rot, scale, opacity)
        self._pre: dict[object, tuple] = {}

        self.rebind()

    # -----------------------------
    # Rebind a la escena actual
    # -----------------------------
    def rebind(self):
        # Desconecta escena anterior
        if _is_valid(getattr(self, "_scene", None)):
            try: self._scene.selectionChanged.disconnect(self.on_selection_changed)
            except Exception: pass
            try: self._scene.destroyed.disconnect(self._on_scene_destroyed)
            except Exception: pass

        # Desconecta viewer anterior
        if _is_valid(getattr(self, "_viewer", None)):
            try: self._viewer.destroyed.disconnect(self._on_viewer_destroyed)
            except Exception: pass

        self._viewer = getattr(self.main, "viewer", None)
        if not _is_valid(self._viewer):
            self._scene = None
            return

        self._scene = self._viewer.scene()
        if not _is_valid(self._scene):
            self._scene = None
            return

        # Conecta señales de la escena
        try:
            self._scene.selectionChanged.connect(self.on_selection_changed, Qt.ConnectionType.UniqueConnection)
        except TypeError:
            self._scene.selectionChanged.connect(self.on_selection_changed)
        self._scene.destroyed.connect(self._on_scene_destroyed)

        # Vincula ítems existentes
        for it in self._scene.items():
            if isinstance(it, ImageItem):
                self.bind_item(it)

        try:
            self._viewer.destroyed.connect(self._on_viewer_destroyed, Qt.ConnectionType.UniqueConnection)
        except TypeError:
            self._viewer.destroyed.connect(self._on_viewer_destroyed)

    # -----------------------------
    # Modo
    # -----------------------------
    def update_mode(self, mode: EditorMode):
        self._mode = mode
        self._refresh_actions()

    # -----------------------------
    # Selección
    # -----------------------------
    def selected_item(self):
        if not _is_valid(self._scene):
            return None
        items = self._scene.selectedItems()
        return items[0] if items else None

    def on_selection_changed(self):
        it = self.selected_item()
        self.main.props.set_layer(it.layer if it else None)
        if it is not None:
            self._pre[it] = self._capture(it)
            self.bind_item(it)
        self._refresh_actions()

    # -----------------------------
    # Bind de un ImageItem (evita duplicados)
    # -----------------------------
    def bind_item(self, item: ImageItem):
        if not _is_valid(item):
            return
        ev = getattr(item, "events", None)
        if not _is_valid(ev):
            return

        eid = _addr(ev)
        if eid in self._bound_event_ids:
            return

        # snapshot inicial
        self._pre[item] = self._capture(item)

        ev.committed.connect(self.on_item_committed)
        try:
            ev.destroyed.connect(lambda *_: self._bound_event_ids.discard(eid))
        except Exception:
            pass

        self._bound_event_ids.add(eid)

    # -----------------------------
    # Commit desde el item (mover/rotar/escalar)
    # -----------------------------
    def on_item_committed(self, item):
        if not _is_valid(item):
            return

        old_pos, old_rot, old_scale, _ = self._pre.get(item, self._capture(item))
        new_pos, new_rot, new_scale, _ = self._capture(item)

        if (old_pos != new_pos) or (old_rot != new_rot) or (old_scale != new_scale):
            cmd = TransformItemCommand(
                item=item,
                old_pos=old_pos, new_pos=new_pos,
                old_rot=old_rot, new_rot=new_rot,
                old_scale=old_scale, new_scale=new_scale,
                text="Transform Item",
            )
            self.main.undo_stack.push(cmd)

        self.main.props.set_layer(getattr(item, "layer", None))
        self._pre[item] = (new_pos, new_rot, new_scale, item.opacity())

    # -----------------------------
    # Cambios desde el panel de propiedades (undoable)
    # -----------------------------
    def on_props_pos(self, x, y):
        it = self.selected_item()
        if not it:
            return
        old = self._pre.get(it, self._capture(it))
        it.layer.x, it.layer.y = x, y
        it.sync_from_layer()
        new = self._capture(it)
        if old[:3] != new[:3]:
            self.main.undo_stack.push(TransformItemCommand(it, old[0], new[0], old[1], new[1], old[2], new[2], "Move Item"))
            self._pre[it] = new

    def on_props_rot(self, angle):
        it = self.selected_item()
        if not it:
            return
        old = self._pre.get(it, self._capture(it))
        it.layer.rotation = angle
        it.sync_from_layer()
        new = self._capture(it)
        if old[:3] != new[:3]:
            self.main.undo_stack.push(TransformItemCommand(it, old[0], new[0], old[1], new[1], old[2], new[2], "Rotate Item"))
            self._pre[it] = new

    def on_props_scale(self, s):
        it = self.selected_item()
        if not it:
            return
        old = self._pre.get(it, self._capture(it))
        it.layer.scale = s
        it.sync_from_layer()
        new = self._capture(it)
        if old[:3] != new[:3]:
            self.main.undo_stack.push(TransformItemCommand(it, old[0], new[0], old[1], new[1], old[2], new[2], "Scale Item"))
            self._pre[it] = new

    def on_props_opacity(self, op):
        it = self.selected_item()
        if not it:
            return
        it.layer.opacity = op
        it.sync_from_layer()
        self._pre[it] = self._capture(it)

    # -----------------------------
    # Toolbar / acciones por modo
    # -----------------------------
    def _refresh_actions(self):
        sel = isinstance(self.selected_item(), ImageItem)
        if self._mode == EditorMode.CloneByCentroid:
            self.main.toolbar_manager.set_clone_enabled(sel)
        else:
            self.main.toolbar_manager.set_clone_enabled(False)

    # -----------------------------
    # Utilidades
    # -----------------------------
    def _capture(self, item: ImageItem):
        """Devuelve (pos, rot, scale, opacity) actuales del item."""
        try:
            return (item.pos(), item.rotation(), item.scale(), item.opacity())
        except Exception:
            # fallback si algún ítem no soporta propiedad
            pos = getattr(item, "pos", lambda: None)()
            rot = getattr(item, "rotation", lambda: None)()
            sca = getattr(item, "scale", lambda: None)()
            op  = getattr(item, "opacity", lambda: None)()
            return (pos, rot, sca, op)

    # -----------------------------
    # Limpieza
    # -----------------------------
    def _on_scene_destroyed(self, *args):
        self._scene = None

    def _on_viewer_destroyed(self, *args):
        self._viewer = None
        self._scene = None
        self._pre.clear()
        self._bound_event_ids.clear()
