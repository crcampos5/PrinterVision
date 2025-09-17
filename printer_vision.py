"""Entry point helper to launch the editor from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src/ directory is available for imports when invoking this script directly.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from editor_tif.app import run

if __name__ == "__main__":
    # Delegate execution to the real application entry point.
    raise SystemExit(run())
