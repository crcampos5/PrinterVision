# src/editor_tif/views/properties_dock.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal, QSignalBlocker
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QFormLayout, QLineEdit, QDoubleSpinBox,
    QHBoxLayout, QPushButton
)

# Solo para hints
from editor_tif.presentation.views.scene_items import Layer


class PropertiesDock(QDockWidget):
    """
    Panel de propiedades para el item seleccionado.
    Emite señales SOLO cuando el usuario confirma (Enter / pierde foco / flechas),
    no en tiempo real mientras teclea, gracias a setKeyboardTracking(False).
    """
    posChanged = Signal(float, float)  # x, y en mm
    rotChanged = Signal(float)         # grados
    scaleChanged = Signal(float)       # factor (1.0 = 100%)
    opacityChanged = Signal(float)     # 0..1

    def __init__(self, parent=None):
        super().__init__("Propiedades", parent)

        self._w = QWidget(self)
        self.setWidget(self._w)

        self._layer: Optional[Layer] = None

        form = QFormLayout(self._w)

        # Campos info
        self.le_file = QLineEdit(); self.le_file.setReadOnly(True)
        self.le_type = QLineEdit(); self.le_type.setReadOnly(True)
        self.le_size = QLineEdit(); self.le_size.setReadOnly(True)

        # Controles numéricos
        self.sb_x = QDoubleSpinBox()
        self.sb_x.setDecimals(3)
        self.sb_x.setRange(-1e6, 1e6)
        self.sb_x.setSingleStep(1.0)
        self.sb_x.setSuffix(" mm")

        self.sb_y = QDoubleSpinBox()
        self.sb_y.setDecimals(3)
        self.sb_y.setRange(-1e6, 1e6)
        self.sb_y.setSingleStep(1.0)
        self.sb_y.setSuffix(" mm")

        self.sb_rot = QDoubleSpinBox()
        self.sb_rot.setDecimals(2)
        self.sb_rot.setRange(-360.0, 360.0)
        self.sb_rot.setSingleStep(1.0)
        self.sb_rot.setSuffix(" °")

        self.sb_scale = QDoubleSpinBox()
        self.sb_scale.setDecimals(3)
        self.sb_scale.setRange(0.010, 100.000)
        self.sb_scale.setSingleStep(0.010)
        self.sb_scale.setValue(1.000)

        self.sb_op = QDoubleSpinBox()
        self.sb_op.setDecimals(2)
        self.sb_op.setRange(0.00, 1.00)
        self.sb_op.setSingleStep(0.05)
        self.sb_op.setValue(1.00)

        # Fila de botones reset
        row = QHBoxLayout()
        btn_rpos = QPushButton("Reset pos")
        btn_rrot = QPushButton("Reset rot")
        btn_rsca = QPushButton("Reset escala")
        row.addWidget(btn_rpos)
        row.addWidget(btn_rrot)
        row.addWidget(btn_rsca)

        # Layout
        form.addRow("Archivo", self.le_file)
        form.addRow("Tipo", self.le_type)
        form.addRow("Tamaño (mm)", self.le_size)
        form.addRow("X (mm)", self.sb_x)
        form.addRow("Y (mm)", self.sb_y)
        form.addRow("Rotación (°)", self.sb_rot)
        form.addRow("Escala", self.sb_scale)
        form.addRow("Opacidad", self.sb_op)
        form.addRow(row)

        # === Emisión SOLO al confirmar ===
        # Evita emitir por cada tecla; valueChanged salta cuando el usuario confirma
        for sb in (self.sb_x, self.sb_y, self.sb_rot, self.sb_scale, self.sb_op):
            sb.setKeyboardTracking(False)

        # Posición: emite siempre el par (x, y)
        self.sb_x.valueChanged.connect(self._emit_pos)
        self.sb_y.valueChanged.connect(self._emit_pos)

        # Rotación / escala / opacidad
        self.sb_rot.valueChanged.connect(lambda v: self.rotChanged.emit(float(v)))
        self.sb_scale.valueChanged.connect(lambda v: self.scaleChanged.emit(float(v)))
        self.sb_op.valueChanged.connect(lambda v: self.opacityChanged.emit(float(v)))

        # Botones reset
        btn_rpos.clicked.connect(self._reset_pos)
        btn_rrot.clicked.connect(lambda: self.sb_rot.setValue(0.0))
        btn_rsca.clicked.connect(lambda: self.sb_scale.setValue(1.0))

        self.setEnabled(False)

    # ----------------------------
    # Helpers de emisión/control
    # ----------------------------
    def _emit_pos(self, *_):
        x = float(self.sb_x.value())
        y = float(self.sb_y.value())
        self.posChanged.emit(x, y)

    def _reset_pos(self):
        # Evita emitir dos veces: bloquea ambas y emite una sola vez
        with QSignalBlocker(self.sb_x), QSignalBlocker(self.sb_y):
            self.sb_x.setValue(0.0)
            self.sb_y.setValue(0.0)
        self._emit_pos()

    # ----------------------------
    # API pública
    # ----------------------------
    def set_layer(self, layer: Optional[Layer]) -> None:
        """Refresca el dock con los datos del layer seleccionado (o None)."""
        self._layer = layer

        if layer is None:
            with QSignalBlocker(self.sb_x), QSignalBlocker(self.sb_y), \
                 QSignalBlocker(self.sb_rot), QSignalBlocker(self.sb_scale), \
                 QSignalBlocker(self.sb_op):
                self.le_file.clear()
                self.le_type.clear()
                self.le_size.clear()
                self.sb_x.setValue(0.0)
                self.sb_y.setValue(0.0)
                self.sb_rot.setValue(0.0)
                self.sb_scale.setValue(1.0)
                self.sb_op.setValue(1.0)

            self.setEnabled(False)
            return

        self.setEnabled(True)

        # Archivo
        name = layer.path.name if isinstance(layer.path, Path) else "(memoria)"
        self.le_file.setText(name)

        # Tipo
        tp = (layer.photometric or "RGB/GRAY").upper()
        self.le_type.setText(tp)

        # Tamaño (mm)
        w = round(layer.width_mm) if getattr(layer, "width_mm", None) else 0
        h = round(layer.height_mm) if getattr(layer, "height_mm", None) else 0
        self.le_size.setText(f"{w} x {h}")

        # Transformaciones (sin disparar señales)
        with QSignalBlocker(self.sb_x), QSignalBlocker(self.sb_y), \
             QSignalBlocker(self.sb_rot), QSignalBlocker(self.sb_scale), \
             QSignalBlocker(self.sb_op):
            self.sb_x.setValue(float(layer.x))
            self.sb_y.setValue(float(layer.y))
            self.sb_rot.setValue(float(layer.rotation))
            self.sb_scale.setValue(float(layer.scale))
            self.sb_op.setValue(float(layer.opacity))
