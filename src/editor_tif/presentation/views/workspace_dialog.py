"""Dialog to configure workspace/table dimensions."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class WorkspaceDialog(QDialog):
    """Allow the user to edit the workspace dimensions in millimeters."""

    def __init__(self, parent=None, width_mm: float = 480.0, height_mm: float = 600.0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parametros de trabajo")

        self.width_spin = QDoubleSpinBox(self)
        self.width_spin.setSuffix(" mm")
        self.width_spin.setDecimals(2)
        self.width_spin.setRange(10.0, 5000.0)
        self.width_spin.setValue(width_mm)

        self.height_spin = QDoubleSpinBox(self)
        self.height_spin.setSuffix(" mm")
        self.height_spin.setDecimals(2)
        self.height_spin.setRange(10.0, 5000.0)
        self.height_spin.setValue(height_mm)

        form = QFormLayout()
        form.addRow("Ancho de la mesa", self.width_spin)
        form.addRow("Alto de la mesa", self.height_spin)

        info_label = QLabel("Las dimensiones se usaran para convertir milimetros a pixeles.")
        info_label.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(info_label)
        layout.addWidget(buttons)

    def values(self) -> tuple[float, float]:
        """Return the configured width and height in millimeters."""
        return float(self.width_spin.value()), float(self.height_spin.value())
