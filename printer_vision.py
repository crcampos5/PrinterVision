# printer_vision.py  (único entry point)
import sys
from pathlib import Path

# Asegura que ./src esté en sys.path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import QApplication
from editor_tif.presentation.main_window import MainWindow  # ajusta ruta si difiere

def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
