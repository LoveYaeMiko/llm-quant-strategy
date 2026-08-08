"""Thin launcher so `python cli.py <cmd>` works without `pip install -e .`.

Blueprint §5 runnable forms:
    python cli.py mine     --config configs/master_config.yaml
    python cli.py backtest --factor-pool ./outputs/factors.json
    python cli.py export   --compiled-path ./online/compiled_factors.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
