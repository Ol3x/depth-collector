from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from depth_collector.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Process already extracted datasets.")
    parser.add_argument("--config", default="configs/default.json", help="Path to config JSON.")
    args = parser.parse_args()
    cli_main(["process", "--config", args.config])


if __name__ == "__main__":
    main()
