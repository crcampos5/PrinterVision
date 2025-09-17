"""Application entry point for the TIF editor."""

from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def run() -> int:
    """Create the QApplication and launch the main window."""
    # Qt expects a single QApplication instance controlling the event loop.
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    # When the module is executed directly, return the exit code to the shell.
    raise SystemExit(run())
