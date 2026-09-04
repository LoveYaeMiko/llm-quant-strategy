"""Daily after-close intraday rollup refresh (PAICC 15:30 scheduler job).

Fetches the latest minute bars for the shadow universe (symbol batches),
merges them into the per-symbol caches and rebuilds
``data/intraday/daily_features.parquet`` so the D-track tail-volume entry gate
never starves on a missing current-day row. Idempotent and incremental — a
no-op cost when the 17:30 run has already self-healed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.config import load_config
    from src.data.intraday import refresh_intraday_daily
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300_500")
    print(f"intraday refresh: {len(symbols)} symbols", flush=True)
    summary = refresh_intraday_daily(cfg, symbols)
    print(
        f"done: calls={summary['calls']} symbols_updated={summary['symbols_updated']} "
        f"rollup={summary['first_day']}..{summary['last_day']} "
        f"({summary['rollup_days']} days × {summary['rollup_symbols']} symbols)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
