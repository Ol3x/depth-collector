"""Allow local imports from src/ without installing the package."""

from pathlib import Path
import sys


SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
