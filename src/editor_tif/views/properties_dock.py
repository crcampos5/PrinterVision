# src/editor_tif/views/properties_dock.py
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QFormLayout, QLineEdit, QDoubleSpinBox, QHBoxLayout, QPushButton
)
from editor_tif.views.scene_items import Layer


class PropertiesDock(QDockWidget):
    posChanged = Signal(float, float)
    rotChanged = Signal(float)
    scaleChanged = Signal(float)
    opacityChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__("Propiedades", parent)
        self._w = QWidget(self)
        self.setWidget(self._w)

        self._layer: Layer | None = None
        form = QFormLayout(self._w)

        self.le_file = QLineEdit(); self.le_file.setReadOnly(True)
        self.le_type = QLineEdit(); self.le_type.setReadOnly(True)
        self.le_size = QLineEdit(); self.le_size.setReadOnly(True)

        self.sb_x = QDoubleSpinBox(); self.sb_x.setDecimals(2); self.sb_x.setRange(-1e6, 1e6)
        self.sb_y = QDoubleSpinBox(); self.sb_y.setDecimals(2); self.sb_y.setRange(-1e6, 1e6)
        self.sb_rot = QDoubleSpinBox(); self.sb_rot.setDecimals(2); self.sb_rot.setRange(-360, 360)
        self.sb_scale = QDoubleSpinBox(); self.sb_scale.setDecimals(3); self.sb_scale.setRange(0.01, 100.0); self.sb_scale.setValue(1.0)
        self.sb_op = QDoubleSpinBox(); self.sb_op.setDecimals(2); self.sb_op.setRange(0.0, 1.0); self.sb_op.setValue(1.0)

        row = QHBoxLayout()
        btn_rpos = QPushButton("Reset pos"); btn_rrot = QPushButton("Reset rot"); btn_rsca = QPushButton("Reset escala")
        row.addWidget(btn_rpos); row.addWidget(btn_rrot); row.addWidget(btn_rsca)

        form.addRow("Archivo", self.le_file)
        form.addRow("Tipo", self.le_type)
        form.addRow("Tamaño (px)", self.le_size)
        form.addRow("X", self.sb_x); form.addRow("Y", self.sb_y)
        form.addRow("Rotación (°)", self.sb_rot)
        form.addRow("Escala", self.sb_scale)
        form.addRow("Opacidad", self.sb_op)
        form.addRow(row)

        # eventos UI -> señales
        self.sb_x.valueChanged.connect(lambda v: self.posChanged.emit(self.sb_x.value(), self.sb_y.value()))
        self.sb_y.valueChanged.connect(lambda v: self.posChanged.emit(self.sb_x.value(), self.sb_y.value()))
        self.sb_rot.valueChanged.connect(self.rotChanged)
        self.sb_scale.valueChanged.connect(self.scaleChanged)
        self.sb_op.valueChanged.connect(self.opacityChanged)

        btn_rpos.clicked.connect(lambda: (self.sb_x.setValue(0.0), self.sb_y.setValue(0.0)))
        btn_rrot.clicked.connect(lambda: self.sb_rot.setValue(0.0))
        btn_rsca.clicked.connect(lambda: self.sb_scale.setValue(1.0))

        self.setEnabled(False)

    def set_layer(self, layer: Layer | None) -> None:
        self._layer = layer
        if layer is None:
            self.le_file.clear(); self.le_type.clear(); self.le_size.clear()
            self.sb_x.setValue(0.0); self.sb_y.setValue(0.0); self.sb_rot.setValue(0.0)
            self.sb_scale.setValue(1.0); self.sb_op.setValue(1.0)
            self.setEnabled(False)
            return

        self.setEnabled(True)
        name = layer.path.name if isinstance(layer.path, Path) else "(memoria)"
        self.le_file.setText(name)
        # tipo/tamaño
        h, w = layer.pixels.shape[:2]
        self.le_size.setText(f"{w} x {h}")
        tp = (layer.photometric or "RGB/GRAY").upper()
        self.le_type.setText(tp)

        # transform
        self.sb_x.setValue(layer.x)
        self.sb_y.setValue(layer.y)
        self.sb_rot.setValue(layer.rotation)
        self.sb_scale.setValue(layer.scale)
        self.sb_op.setValue(layer.opacity)
