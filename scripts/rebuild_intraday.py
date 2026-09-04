"""Rebuild the intraday daily-feature rollup from cached minute bars (adds
open30 / range / afternoon to the existing five features)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.intraday import _intraday_dir, build_intraday_frames


def main() -> int:
    from src.config import load_config
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300_500")
    wide = build_intraday_frames(cfg, symbols)
    out_path = _intraday_dir(cfg) / "daily_features.parquet"
    rollup = pd.concat({k: v for k, v in wide.items()}, axis=1)
    rollup.columns.names = ["feature", "symbol"]
    rollup.to_parquet(out_path)
    print(f"wrote {out_path} ({len(rollup)} days × {rollup.shape[1]} series)", flush=True)
    print("features:", list(wide), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
