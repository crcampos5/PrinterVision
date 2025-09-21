# editor_tif/clipboard.py
from PySide6.QtCore import QByteArray, QDataStream, QIODevice, QPointF, QMimeData

MIME_TYPE = "application/x-printervision-items"

def _item_state_tuple(it):
    # Posición, rotación, escala
    try:
        pos = it.pos()
        rot = it.rotation()
        sca = it.scale()
    except Exception:
        pos, rot, sca = QPointF(0, 0), 0.0, 1.0
    return pos, rot, sca

def _item_source_id(it):
    # Prioridad: layer.source_id -> item.source_id -> layer.path
    sid = None
    layer = getattr(it, "layer", None)
    if layer is not None:
        sid = getattr(layer, "source_id", None) or (str(layer.path) if getattr(layer, "path", None) else None)
    if (not sid) and hasattr(it, "source_id"):
        sid = getattr(it, "source_id")
    return sid or ""

def serialize_items(items):
    buf = QByteArray()
    stream = QDataStream(buf, QIODevice.WriteOnly)
    stream.writeInt32(len(items))
    for it in items:
        cls_name = it.__class__.__name__
        stream.writeString(cls_name)

        pos, rot, sca = _item_state_tuple(it)
        stream.writeDouble(pos.x()); stream.writeDouble(pos.y())
        stream.writeDouble(float(rot)); stream.writeDouble(float(sca))

        source_id = _item_source_id(it)
        stream.writeString(source_id)

    md = QMimeData()
    md.setData(MIME_TYPE, buf)
    return md

def deserialize_items(md, factory, offset=QPointF(10, 10)):
    if not md or not md.hasFormat(MIME_TYPE):
        return []

    data = md.data(MIME_TYPE)
    stream = QDataStream(data, QIODevice.ReadOnly)

    count = stream.readInt32()
    created = []
    for _ in range(count):
        cls_name = stream.readString()
        x = stream.readDouble(); y = stream.readDouble()
        rot = stream.readDouble(); sca = stream.readDouble()
        source_id = stream.readString()

        # Validación de source_id para ImageItem
        if cls_name == "ImageItem" and (not source_id or source_id in [".", "./"]):
            # No hay cómo reconstruir desde disco; salta el ítem
            # (Opcional: podrías levantar una excepción con mensaje más claro)
            continue

        item = factory(cls_name, source_id)
        # Aplica offset de pegado
        item.setPos(QPointF(x, y) + offset)
        try:
            item.setRotation(rot)
        except Exception:
            pass
        try:
            item.setScale(sca)
        except Exception:
            pass

        created.append(item)

    return created
