"""Worker types for long-running tasks."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class BaseWorker(QObject):
    """Base worker emitting progress and result signals."""

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str)

    def start(self) -> None:
        raise NotImplementedError
