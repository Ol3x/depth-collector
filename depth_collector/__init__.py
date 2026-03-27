"""Local namespace bridge for the src/ package layout."""

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "depth_collector"
__path__.append(str(_SRC_PACKAGE))
