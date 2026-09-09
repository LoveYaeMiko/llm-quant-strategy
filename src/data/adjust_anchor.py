"""Explicit, versioned price-adjustment anchor — defect C3.

**The defect.** The PIT price payload stores a backward-adjusted series::

    {"open": .., "high": .., "low": ..,
     "close": <ADJUSTED>, "raw_close": <RAW>,
     "adjust_factor": <float>, "volume": .., "amount": .., "name": ..}

with ``close = round(raw_close * adjust_factor, 4)``. The factor is the product
of the inverse per-event ex-factors of every corporate action *strictly after*
the bar (``src/data/ingestion/alphafeed_adapter.py::to_price_records``), so it is
implicitly anchored at the **newest bar of the ingest batch** — the newest bar
always has ``adjust_factor == 1.0``.

Nothing in the store or in any artifact records *which* anchor a stored series
was built against. The consequences are structural, not cosmetic:

1. **Every re-ingest silently re-bases the whole history.** One new ex-event
   (a dividend, a bonus issue, a split) multiplies the factor of *every* earlier
   bar by ``1/ex_factor``. Yesterday's ``close`` series is not today's ``close``
   series, even though no code and no config changed.
2. **A backtest re-run is therefore not reproducible.** Identical parameters
   over an identical date window produce different numbers purely because the
   basis moved.
3. **Forward "tracking error" is confounded.** Measuring a live book against a
   historical run mixes basis drift into what is supposed to measure strategy
   behaviour, and no artifact says ``data_as_of`` to disambiguate.

**The fix (this module).** The anchor is made explicit, versioned and
drift-detectable without rewriting a single stored price:

* :func:`capture_anchor` reads the price records and freezes the *anchor
  fingerprint*: ``data_as_of`` (the newest bar date), the per-symbol
  ``adjust_factor`` at each symbol's **last** bar, the factor distribution, and
  a ``factors_sha256`` over the canonical per-symbol factor map. The factor at a
  symbol's last bar is the observable proxy for the anchor: it is 1.0 exactly
  when that symbol's newest bar is the anchor bar, and it shifts for every
  symbol whose history sits behind a newly ingested event.
* :func:`write_anchor` / :func:`load_anchor` persist that fingerprint as
  canonical JSON (sorted keys, no whitespace, ``ensure_ascii=False``) with a
  ``record_sha256`` over the payload excluding the hash field, refuse to
  overwrite an existing file unless ``force=True``, and append every write to a
  history directory when one is configured.
* :func:`compare_anchors` compares two fingerprints and reports drift.

**How drift is interpreted.** ``drifted=True`` means the stored price basis
moved: at least one common symbol's last-bar factor changed by more than ``tol``
relative to the previous fingerprint, or symbols disappeared from the store.
A moved ``data_as_of`` alone (new bars ingested, no re-basing) is reported as
``data_as_of_moved`` but is *not* drift by itself. Operationally:

* a **frozen baseline** is the reference for any long forward window — capture
  it before the window opens, never re-freeze silently;
* ``drifted=True`` against that baseline means any metric computed before and
  after the change is on **different price bases** and must not be compared
  (re-run the earlier measurement or re-state it with its ``data_as_of``);
* ``drifted=False`` plus an unchanged ``factors_sha256`` is the evidence that a
  re-run is comparable to the baseline run.

Only ``numpy``/``pandas`` plus the optional Postgres driver are required, so the
module is safe to import from the data, research and strategy layers alike.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

#: The only adjustment policy this repo implements today.
ANCHOR_POLICY = "newest_bar_backward_adjust"

#: Field holding the integrity hash of a written anchor file.
RECORD_HASH_KEY = "record_sha256"

#: Env var that configures a default history directory for :func:`write_anchor`.
HISTORY_DIR_ENV = "FQA_ADJUST_ANCHOR_HISTORY_DIR"

#: ``adjust_factor`` values within this of 1.0 count as "unchanged / unadjusted".
FACTOR_ONE_TOL = 1e-12

#: Default drift tolerance for :func:`compare_anchors` (relative factor change).
DEFAULT_TOL = 1e-9

#: How many extreme factors :func:`capture_anchor` reports.
TOP_N_FACTORS = 10

_STAMP_RE = re.compile(r"[^0-9A-Za-z]+")


# ---------------------------------------------------------------------------
# canonical serialization / hashing
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, non-ASCII kept literal.

    Deterministic for equal values regardless of dict insertion order, which is
    what makes the stored hashes stable across runs.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def factors_sha256(per_symbol_factor: Mapping[str, float]) -> str:
    """sha256 of the canonical per-symbol factor map.

    Stable across runs on unchanged data: the map is sorted by key and floats
    are serialized by their shortest round-tripping repr. Dict key order of the
    input is irrelevant.
    """
    clean = {str(k): float(v) for k, v in per_symbol_factor.items()}
    return _sha256(canonical_json(clean))


def record_sha256(anchor: Mapping[str, Any]) -> str:
    """sha256 of an anchor payload, excluding the stored hash field itself."""
    payload = {k: v for k, v in anchor.items() if k != RECORD_HASH_KEY}
    return _sha256(canonical_json(payload))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Local ISO8601 timestamp with the UTC offset (second resolution)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _as_of_str(value: Any) -> Optional[str]:
    """``YYYY-MM-DD`` for a timestamp-ish value (None passes through)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # non-scalar — fall through to str()
        pass
    if isinstance(value, str):
        return value[:10]
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(value)[:10]


def _normalize_symbols(symbols: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Canonicalize + sort a symbol list (None means "every stored symbol")."""
    if symbols is None:
        return None
    try:
        from .schema.symbols import normalize_symbol
    except Exception:  # pragma: no cover - schema package always present in-repo
        normalize_symbol = None  # type: ignore[assignment]
    out: list[str] = []
    for sym in symbols:
        s = str(sym)
        if normalize_symbol is not None:
            try:
                s = normalize_symbol(s)
            except Exception:
                s = str(sym)
        out.append(s)
    return sorted(dict.fromkeys(out))


def _resolve_url(cfg_or_url: Any) -> str:
    """Accept a :class:`~src.config.Config` or a database URL string.

    Resolution matches the rest of the repo (``src/cli.py::_market_data``):
    ``config.get("data.pit_database_url")``.
    """
    if isinstance(cfg_or_url, str):
        url = cfg_or_url
    elif hasattr(cfg_or_url, "get"):
        url = cfg_or_url.get("data.pit_database_url")
    else:
        raise TypeError(
            "cfg_or_url must be a Config (dotted .get) or a pit_database_url string, "
            f"got {type(cfg_or_url).__name__}"
        )
    if not url:
        raise ValueError(
            "empty pit_database_url — set PIT_DATABASE_URL (see .env) or pass a URL"
        )
    return str(url)


# ---------------------------------------------------------------------------
# reading the price records (one DB pass)
# ---------------------------------------------------------------------------

# One server-side aggregate per capture: per symbol the factor at the LAST bar,
# that bar's date, and the symbol's row count. This is the memory-lean
# equivalent of ``store.snapshot("price")`` + ``groupby.tail(1)``: the snapshot
# materializes all ~12.5M bars and is documented in
# ``src/portfolio/backtest_runner.py`` as exceeding 32 GB RAM, while this query
# returns one row per symbol (~5k rows, tens of seconds).
_PG_ANCHOR_SQL = """
SELECT symbol,
       (array_agg(COALESCE((payload->>'adjust_factor')::float8, 1.0)
                  ORDER BY valid_from DESC))[1] AS factor,
       max(valid_from) AS last_bar,
       count(*)        AS n_rows
FROM pit_records
WHERE payload->>'record_type' = 'price'
"""


def _postgres_anchor_rows(url: str, symbols: Optional[Sequence[str]]) -> list[tuple]:
    """One aggregate query → ``[(symbol, factor, last_bar, n_rows), ...]``."""
    import psycopg2

    sql = _PG_ANCHOR_SQL
    params: list = []
    if symbols:
        sql += " AND symbol = ANY(%s)"
        params.append(list(symbols))
    sql += " GROUP BY symbol ORDER BY symbol"
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()
    return [(str(s), float(f if f is not None else 1.0), lb, int(n)) for s, f, lb, n in rows]


def anchor_rows_from_frame(
    records: pd.DataFrame, symbols: Optional[Sequence[str]] = None
) -> list[tuple]:
    """Same shape as :func:`_postgres_anchor_rows`, from an in-memory frame.

    The frame is the output of ``store.snapshot("price")`` (or any price record
    frame). ``adjust_factor`` may be missing/NaN — such bars count as 1.0, the
    same convention as ``src/data/basis.py``.
    """
    if records is None or len(records) == 0:
        return []
    df = records
    if symbols:
        df = df[df["symbol"].isin(list(symbols))]
        if df.empty:
            return []
    if "adjust_factor" in df.columns:
        factor = pd.to_numeric(df["adjust_factor"], errors="coerce").fillna(1.0)
    else:
        factor = pd.Series(1.0, index=df.index)
    work = pd.DataFrame(
        {
            "symbol": df["symbol"].astype(str).to_numpy(),
            "valid_from": pd.to_datetime(df["valid_from"]).to_numpy(),
            "factor": factor.astype(float).to_numpy(),
        }
    )
    counts = work.groupby("symbol", sort=True).size()
    last = work.sort_values(["symbol", "valid_from"]).groupby("symbol", sort=True).tail(1)
    return [
        (str(row.symbol), float(row.factor), row.valid_from, int(counts[row.symbol]))
        for row in last.itertuples(index=False)
    ]


def _store_anchor_rows(url: str, symbols: Optional[Sequence[str]]) -> list[tuple]:
    """Bulk-read price records through the PIT loader — exactly one ``snapshot``."""
    from .point_in_time_loader import from_url

    store = from_url(url)
    try:
        records = store.snapshot("price")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    return anchor_rows_from_frame(records, symbols)


# ---------------------------------------------------------------------------
# building / capturing the anchor
# ---------------------------------------------------------------------------


def build_anchor(
    rows: Iterable[tuple],
    *,
    captured_at: Optional[str] = None,
) -> dict:
    """Assemble the anchor fingerprint from ``(symbol, factor, last_bar, n)`` rows."""
    per_symbol: dict[str, float] = {}
    last_bars: dict[str, Any] = {}
    n_rows = 0
    for symbol, factor, last_bar, n in rows:
        sym = str(symbol)
        per_symbol[sym] = float(factor)
        last_bars[sym] = last_bar
        n_rows += int(n)
    per_symbol = {sym: per_symbol[sym] for sym in sorted(per_symbol)}

    values = list(per_symbol.values())
    stats = {
        "n_symbols_factor_ne_1": sum(1 for v in values if abs(v - 1.0) > FACTOR_ONE_TOL),
        "n_symbols_factor_lt_1": sum(1 for v in values if v < 1.0 - FACTOR_ONE_TOL),
        "n_symbols_factor_gt_1": sum(1 for v in values if v > 1.0 + FACTOR_ONE_TOL),
        "min": (min(values) if values else None),
        "max": (max(values) if values else None),
    }
    top = sorted(per_symbol.items(), key=lambda kv: (-abs(kv[1] - 1.0), kv[0]))
    top_factors = [[sym, factor] for sym, factor in top[:TOP_N_FACTORS]]

    data_as_of = None
    if last_bars:
        data_as_of = _as_of_str(max(last_bars.values()))

    return {
        "captured_at": captured_at or _now_iso(),
        "data_as_of": data_as_of,
        "n_symbols": len(per_symbol),
        "n_rows": n_rows,
        "anchor_policy": ANCHOR_POLICY,
        "factor_stats": stats,
        "top_factors": top_factors,
        "per_symbol_factor": per_symbol,
        "factors_sha256": factors_sha256(per_symbol),
    }


def anchor_from_frame(
    records: pd.DataFrame,
    *,
    symbols: Optional[Sequence[str]] = None,
    captured_at: Optional[str] = None,
) -> dict:
    """Offline capture from a price-record frame (no database)."""
    return build_anchor(
        anchor_rows_from_frame(records, _normalize_symbols(symbols)),
        captured_at=captured_at,
    )


def capture_anchor(cfg_or_url: Any, symbols: Optional[Iterable[str]] = None) -> dict:
    """Capture the current adjustment anchor fingerprint from the PIT store.

    Parameters
    ----------
    cfg_or_url
        A :class:`~src.config.Config` (uses ``data.pit_database_url``) or the
        database URL itself.
    symbols
        Optional subset. ``None`` (default) covers every stored price symbol.
        Entries are canonicalized (``600519.SH``) and sorted deterministically.

    Returns
    -------
    dict
        ``captured_at``, ``data_as_of``, ``n_symbols``, ``n_rows``,
        ``anchor_policy``, ``factor_stats``, ``top_factors``,
        ``per_symbol_factor`` and ``factors_sha256``.

    Exactly one read pass hits the database: a Postgres URL uses a single
    per-symbol aggregate query; any other backend goes through one
    ``store.snapshot("price")`` call.
    """
    url = _resolve_url(cfg_or_url)
    syms = _normalize_symbols(symbols)
    if url.startswith("postgresql"):
        rows = _postgres_anchor_rows(url, syms)
    else:
        rows = _store_anchor_rows(url, syms)
    return build_anchor(rows)


# ---------------------------------------------------------------------------
# comparing two anchors
# ---------------------------------------------------------------------------


def _rel_change(prev: float, curr: float) -> Optional[float]:
    """Relative change ``(curr - prev) / |prev|``; ``None`` when undefined."""
    if prev != 0.0:
        return (curr - prev) / abs(prev)
    if curr == 0.0:
        return 0.0
    return None


def compare_anchors(prev: Mapping[str, Any], curr: Mapping[str, Any], tol: float = DEFAULT_TOL) -> dict:
    """Compare two anchor fingerprints and report basis drift.

    ``drifted`` is True when any *common* symbol's last-bar ``adjust_factor``
    moved by more than ``tol`` (relative, or absolute when the previous factor
    was 0) **or** when symbols were lost from the store. Symbols that are merely
    new, and a moved ``data_as_of`` with unchanged factors, are reported but do
    not by themselves set ``drifted``.

    Returns a JSON-serializable dict; ``drifted_symbols`` is sorted by
    descending ``|rel_change|`` (undefined relative changes first).
    """
    prev_f = {str(k): float(v) for k, v in (prev.get("per_symbol_factor") or {}).items()}
    curr_f = {str(k): float(v) for k, v in (curr.get("per_symbol_factor") or {}).items()}

    common = sorted(set(prev_f) & set(curr_f))
    new_symbols = sorted(set(curr_f) - set(prev_f))
    lost_symbols = sorted(set(prev_f) - set(curr_f))

    drifted_symbols: list[dict] = []
    max_rel_change = 0.0
    for sym in common:
        p = prev_f[sym]
        c = curr_f[sym]
        rel = _rel_change(p, c)
        if rel is not None:
            max_rel_change = max(max_rel_change, abs(rel))
            moved = abs(rel) > tol
        else:
            moved = abs(c - p) > tol
        if moved:
            drifted_symbols.append(
                {"symbol": sym, "prev": p, "curr": c, "rel_change": rel}
            )
    drifted_symbols.sort(
        key=lambda d: (
            -(abs(d["rel_change"]) if d["rel_change"] is not None else float("inf")),
            d["symbol"],
        )
    )

    prev_as_of = _as_of_str(prev.get("data_as_of"))
    curr_as_of = _as_of_str(curr.get("data_as_of"))
    prev_hash = prev.get("factors_sha256")
    curr_hash = curr.get("factors_sha256")
    return {
        "data_as_of_prev": prev_as_of,
        "data_as_of_curr": curr_as_of,
        "data_as_of_moved": prev_as_of != curr_as_of,
        "n_common": len(common),
        "n_new_symbols": len(new_symbols),
        "n_lost_symbols": len(lost_symbols),
        "new_symbols": new_symbols,
        "lost_symbols": lost_symbols,
        "n_drifted": len(drifted_symbols),
        "drifted_symbols": drifted_symbols,
        "max_rel_change": float(max_rel_change),
        "factors_sha256_prev": prev_hash,
        "factors_sha256_curr": curr_hash,
        "hash_changed": bool(prev_hash != curr_hash),
        "drifted": bool(drifted_symbols or lost_symbols),
        "tol": float(tol),
    }


def format_drift_report(report: Mapping[str, Any], *, top: int = 10) -> str:
    """Human-readable rendering of a :func:`compare_anchors` report."""
    lines = [
        f"data_as_of: {report.get('data_as_of_prev')} -> {report.get('data_as_of_curr')} "
        f"(moved={report.get('data_as_of_moved')})",
        f"symbols: common={report.get('n_common')} new={report.get('n_new_symbols')} "
        f"lost={report.get('n_lost_symbols')}",
        f"factors_sha256: {report.get('factors_sha256_prev')} -> {report.get('factors_sha256_curr')} "
        f"(changed={report.get('hash_changed')})",
        f"drifted: {report.get('n_drifted')}/{report.get('n_common')} common symbols "
        f"(tol={report.get('tol')}, max_rel_change={report.get('max_rel_change'):.3e})",
    ]
    rows = list(report.get("drifted_symbols") or [])
    if rows:
        lines.append(f"top drifted symbols (of {len(rows)}):")
        for row in rows[:top]:
            rel = row.get("rel_change")
            rel_txt = "n/a" if rel is None else f"{rel:+.3e}"
            lines.append(
                f"  {row['symbol']:<12} {row['prev']!r} -> {row['curr']!r}  rel={rel_txt}"
            )
    elif not report.get("n_lost_symbols"):
        lines.append("top drifted symbols: (none)")
    if report.get("lost_symbols"):
        lost = report["lost_symbols"]
        shown = ", ".join(lost[:top])
        more = "" if len(lost) <= top else f" ... (+{len(lost) - top} more)"
        lines.append(f"lost symbols: {shown}{more}")
    if report.get("new_symbols"):
        lines.append(f"new symbols: {len(report['new_symbols'])}")
    lines.append(f"VERDICT: {'DRIFTED' if report.get('drifted') else 'stable'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _history_dir(history_dir: Any = None) -> Optional[Path]:
    """Explicit arg wins; otherwise the ``FQA_ADJUST_ANCHOR_HISTORY_DIR`` env var."""
    if history_dir is not None:
        return Path(history_dir)
    env = os.environ.get(HISTORY_DIR_ENV)
    return Path(env) if env else None


def _stamp(anchor: Mapping[str, Any]) -> str:
    """Filename-safe timestamp derived from ``captured_at``.

    ``:`` is illegal on Windows, so it is dropped; ``+`` becomes ``p`` to keep
    the UTC offset visible (``2026-09-09T17:38:13+08:00`` →
    ``20260909T173813p0800``).
    """
    raw = str(anchor.get("captured_at") or _now_iso())
    stamp = _STAMP_RE.sub("", raw.replace(":", "").replace("+", "p"))
    return stamp or _STAMP_RE.sub("", _now_iso().replace(":", "").replace("+", "p"))


def write_anchor(
    path: str | Path,
    anchor: Mapping[str, Any],
    *,
    force: bool = False,
    history_dir: str | Path | None = None,
) -> Path:
    """Write ``anchor`` as canonical JSON and return the written path.

    * the payload (everything except ``record_sha256``) is hashed with
      :func:`record_sha256` and stored alongside it;
    * an existing file is **never** overwritten unless ``force=True``
      (:class:`FileExistsError`);
    * when a history directory is configured (``history_dir`` argument or
      ``FQA_ADJUST_ANCHOR_HISTORY_DIR``), a timestamped copy is appended there
      on every successful write, without clobbering an existing copy.
    """
    dest = Path(path)
    if dest.exists() and not force:
        raise FileExistsError(
            f"{dest} already exists — pass force=True to overwrite "
            "(re-freezing a reference anchor must be deliberate)"
        )
    payload = {k: v for k, v in anchor.items() if k != RECORD_HASH_KEY}
    stored = dict(payload)
    stored[RECORD_HASH_KEY] = record_sha256(payload)
    text = canonical_json(stored)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")

    hist = _history_dir(history_dir)
    if hist is not None:
        hist.mkdir(parents=True, exist_ok=True)
        stamp = _stamp(anchor)
        target = hist / f"{dest.stem}_{stamp}.json"
        suffix = 1
        while target.exists():
            target = hist / f"{dest.stem}_{stamp}_{suffix}.json"
            suffix += 1
        target.write_text(text, encoding="utf-8")
    return dest


def load_anchor(path: str | Path, *, verify: bool = True) -> dict:
    """Load an anchor file, verifying its stored ``record_sha256`` by default.

    Raises :class:`ValueError` for a missing/corrupt hash or a payload that no
    longer matches it — a silently edited anchor file must never be trusted.
    """
    src = Path(path)
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{src} does not contain a JSON object")
    if verify:
        stored = data.get(RECORD_HASH_KEY)
        if not stored:
            raise ValueError(f"{src} has no {RECORD_HASH_KEY} — not a written anchor file")
        actual = record_sha256(data)
        if actual != stored:
            raise ValueError(
                f"{src} failed integrity check: stored {RECORD_HASH_KEY}={stored} "
                f"but payload hashes to {actual}"
            )
    return data
