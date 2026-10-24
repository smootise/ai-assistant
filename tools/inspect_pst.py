"""Run the standalone discovery CLI without installing JARVIS."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.pst_discovery import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
