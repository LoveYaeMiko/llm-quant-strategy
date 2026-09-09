"""Paper-trading ledger — persistent, resumable account state (stdlib SQLite).

The three gaps between "gate-passing backtest" and "paper trading" are (1) a
daily runner, (2) cross-day persistence and (3) real-time point-in-time
alignment. This module is gap (2): the ``OrderExecutor`` holds ``cash`` and
``positions`` in memory and forgets them on exit; :class:`PaperLedger` makes
that state durable so a crashed/killed run resumes from its last recorded day
instead of restarting from zero.

Schema mirrors the PIT loader (``SQLitePointInTimeLoader``) — stdlib only, no
driver dependency:

* ``daily_state`` — one row per processed day (cash / equity / gross / fills);
* ``positions`` — the end-of-day share holdings (``date`` × ``symbol``);
* ``fills`` — the execution ledger (slippage / commission / notional per fill),
  which is also the §7 cost-model calibration source.

Every method is a straight SQL read/write; ``latest_state`` returns exactly the
three scalars the runner needs to reconstruct an :class:`OrderExecutor`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..online.order_executor import Fill


class PaperLedger:
    """Append-only account ledger on SQLite. Safe to reopen and resume."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_state (
                date           TEXT PRIMARY KEY,
                cash           REAL NOT NULL,
                equity         REAL NOT NULL,
                gross_exposure REAL NOT NULL DEFAULT 0.0,
                n_positions    INTEGER NOT NULL DEFAULT 0,
                n_fills        INTEGER NOT NULL DEFAULT 0,
                commission     REAL NOT NULL DEFAULT 0.0,
                notional       REAL NOT NULL DEFAULT 0.0
            );
            CREATE TABLE IF NOT EXISTS positions (
                date   TEXT NOT NULL,
                symbol TEXT NOT NULL,
                shares REAL NOT NULL,
                PRIMARY KEY (date, symbol)
            );
            CREATE TABLE IF NOT EXISTS fills (
                seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                date       TEXT NOT NULL,
                symbol     TEXT NOT NULL,
                side       TEXT NOT NULL,
                shares     REAL NOT NULL,
                price      REAL NOT NULL,
                commission REAL NOT NULL,
                notional   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_positions_date ON positions (date);
            CREATE INDEX IF NOT EXISTS ix_fills_date ON fills (date);
            """
        )
        self._conn.commit()
        # migration: intraday fills carry their bar timestamp
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(fills)").fetchall()]
        if "time" not in cols:
            self._conn.execute("ALTER TABLE fills ADD COLUMN time TEXT NOT NULL DEFAULT ''")
            self._conn.commit()
        # migration: fill provenance — live (real-time trader) vs replay (minute
        # sweep) vs close (rebalance) vs auction (15:00 order list). Defect D-4:
        # the audit found 96% of intraday stops were replays presented as if they
        # had been executed in real time.
        if "source" not in cols:
            self._conn.execute("ALTER TABLE fills ADD COLUMN source TEXT NOT NULL DEFAULT ''")
            self._conn.commit()

    # ------------------------------------------------------------------ write
    def record_day(
        self,
        date: str | pd.Timestamp,
        cash: float,
        equity: float,
        positions: dict[str, float],
        fills: Iterable[Fill],
        gross_exposure: float,
        protect_after: int | None = None,
    ) -> None:
        """Persist one day's end-of-day state plus its execution fills.

        Idempotent on ``date`` (``INSERT OR REPLACE``) so a re-run of the same
        day overwrites rather than duplicates — the resume guard in the runner
        skips already-recorded dates, this is a second line of defence.

        ``protect_after`` is the highest fill ``seq`` observed for this date at
        the start of the close run: live fills appended by the real-time trader
        WHILE the close run is processing get seqs above it and survive the
        delete, so a mid-day manual run can never drop a live fill. ``None``
        (the legacy behaviour) deletes every row for the date first.
        """
        d = str(pd.Timestamp(date).date())
        fill_rows = list(fills)
        commission = sum(f.commission for f in fill_rows)
        notional = sum(f.notional for f in fill_rows)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO daily_state
                    (date, cash, equity, gross_exposure, n_positions, n_fills, commission, notional)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (d, float(cash), float(equity), float(gross_exposure),
                 len(positions), len(fill_rows), float(commission), float(notional)),
            )
            self._conn.execute("DELETE FROM positions WHERE date = ?", (d,))
            self._conn.executemany(
                "INSERT INTO positions (date, symbol, shares) VALUES (?, ?, ?)",
                [(d, s, float(v)) for s, v in positions.items()],
            )
            if protect_after is None:
                self._conn.execute("DELETE FROM fills WHERE date = ?", (d,))
            else:
                self._conn.execute(
                    "DELETE FROM fills WHERE date = ? AND seq <= ?", (d, int(protect_after))
                )
            self._conn.executemany(
                "INSERT INTO fills (date, symbol, side, shares, price, commission, notional, time, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (d, f.symbol, f.side, float(f.shares), float(f.price),
                     float(f.commission), float(f.notional), str(getattr(f, "time", "") or ""),
                     str(getattr(f, "source", "") or ""))
                    for f in fill_rows
                ],
            )

    # ------------------------------------------------------------------- read
    def last_date(self) -> Optional[str]:
        row = self._conn.execute("SELECT MAX(date) FROM daily_state").fetchone()
        return row[0] if row and row[0] is not None else None

    def has_date(self, date: str | pd.Timestamp) -> bool:
        d = str(pd.Timestamp(date).date())
        return self._conn.execute("SELECT 1 FROM daily_state WHERE date = ?", (d,)).fetchone() is not None

    def latest_state(self) -> tuple[Optional[str], Optional[float], dict[str, float]]:
        """``(last_date, cash, positions)`` for the most recent recorded day.

        ``cash`` is ``None`` when nothing has been recorded yet (the runner
        falls back to its configured initial cash); ``positions`` is then empty.
        """
        last = self.last_date()
        if last is None:
            return None, None, {}
        row = self._conn.execute(
            "SELECT cash FROM daily_state WHERE date = ?", (last,)
        ).fetchone()
        cash = float(row[0]) if row else None
        rows = self._conn.execute(
            "SELECT symbol, shares FROM positions WHERE date = ?", (last,)
        ).fetchall()
        return last, cash, {s: float(v) for s, v in rows}

    def equity_curve(self) -> pd.Series:
        """Day-by-day equity, chronological (``date`` → equity)."""
        df = pd.read_sql_query(
            "SELECT date, equity FROM daily_state ORDER BY date", self._conn
        )
        if df.empty:
            return pd.Series(dtype=float)
        return pd.Series(df["equity"].to_numpy(dtype=float), index=df["date"].tolist())

    def daily_states(self) -> pd.DataFrame:
        """Full ``daily_state`` table as a DataFrame (digest / audit)."""
        return pd.read_sql_query(
            "SELECT * FROM daily_state ORDER BY date", self._conn
        )

    def n_fills(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM fills").fetchone()
        return int(row[0]) if row else 0

    def fills(self) -> pd.DataFrame:
        """Execution ledger as a DataFrame (the §7 cost-model calibration source)."""
        return pd.read_sql_query("SELECT * FROM fills ORDER BY seq", self._conn)

    def fills_for_date(self, date: str | pd.Timestamp) -> list[Fill]:
        """Fills already recorded for one day (the live intraday trader appends
        during the session; the close run must merge, not delete, them)."""
        d = str(pd.Timestamp(date).date())
        rows = self._conn.execute(
            "SELECT date, time, symbol, side, shares, price, commission, notional, "
            "COALESCE(source, '') FROM fills WHERE date = ? ORDER BY seq",
            (d,),
        ).fetchall()
        return [
            Fill(date=r[0], time=r[1] or "", symbol=r[2], side=r[3], shares=float(r[4]),
                 price=float(r[5]), commission=float(r[6]), notional=float(r[7]),
                 source=r[8] or "")
            for r in rows
        ]

    def max_fill_seq(self, date: str | pd.Timestamp) -> int:
        """Highest fill ``seq`` recorded for one day (0 when none).

        The close runner snapshots this before merging live fills, then passes
        it as ``protect_after`` to :meth:`record_day` so fills appended by the
        real-time trader while the run is in flight are never deleted.
        """
        d = str(pd.Timestamp(date).date())
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM fills WHERE date = ?", (d,)
        ).fetchone()
        return int(row[0]) if row else 0

    def append_fill(self, fill: Fill) -> None:
        """Append one live intraday fill (used by the real-time trader)."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO fills (date, time, symbol, side, shares, price, commission, notional, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(pd.Timestamp(fill.date).date()), str(getattr(fill, "time", "") or ""),
                 fill.symbol, fill.side, float(fill.shares), float(fill.price),
                 float(fill.commission), float(fill.notional),
                 str(getattr(fill, "source", "") or "")),
            )

    def fills_by_source(self) -> dict[str, int]:
        """Fill counts per provenance (``live`` / ``replay`` / ``close`` / ``auction``).

        Defect D-4: the panel and reports must show how many intraday stops were
        actually executed in real time versus replayed from minute bars.
        """
        rows = self._conn.execute(
            "SELECT COALESCE(source, ''), COUNT(*) FROM fills GROUP BY 1"
        ).fetchall()
        return {(r[0] or "unlabelled"): int(r[1]) for r in rows}

    def cash_stats(self) -> dict[str, float | int | str | None]:
        """Cash discipline of the recorded days (audit finding V-1).

        The pre-2026-09-09 executor could buy before its own sells settled and
        drove cash negative on 12 days (max ≈ 8.7% of a 50k account = implicit
        leverage). The guard is fixed; these historical days stay as recorded, so
        the status/report must keep disclosing them instead of hiding the hole.
        """
        rows = self._conn.execute(
            "SELECT date, cash FROM daily_state ORDER BY date"
        ).fetchall()
        if not rows:
            return {"days": 0, "negative_cash_days": 0, "min_cash": None,
                    "worst_date": None}
        cash = [(str(r[0]), float(r[1])) for r in rows]
        negatives = [(d, c) for d, c in cash if c < 0]
        worst = min(cash, key=lambda kv: kv[1])
        return {
            "days": len(cash),
            "negative_cash_days": len(negatives),
            "min_cash": round(worst[1], 2),
            "worst_date": worst[0],
            "negative_cash_first": negatives[0][0] if negatives else None,
            "negative_cash_last": negatives[-1][0] if negatives else None,
        }

    def total_commission(self) -> float:
        row = self._conn.execute("SELECT COALESCE(SUM(commission), 0) FROM fills").fetchone()
        return float(row[0]) if row else 0.0

    def close(self) -> None:
        self._conn.close()


def clone_ledger_before(
    src: str | Path,
    dst: str | Path,
    before: str | pd.Timestamp,
) -> dict:
    """Copy ``src``'s state as of the day BEFORE ``before`` into ``dst``.

    A shadow candidate or a replay must START from the same account state the
    live book was in on the eve of the window — otherwise the first submitted
    order list refers to holdings the copy does not have (found 2026-09-09: a
    candidate with a fresh ledger executed production's sell list and opened a
    short). This is a file-level copy plus a truncation, so it is exact and
    cheap; the source ledger is opened read-only by ``shutil.copy2`` and never
    modified.

    Returns ``{days, positions, fills, first_date, last_date}`` describing the
    cloned state (all zeros for an empty source).
    """
    import shutil
    import sqlite3

    cutoff = str(pd.Timestamp(before).date())
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.unlink(missing_ok=True)
    shutil.copy2(Path(src), dst_path)
    con = sqlite3.connect(str(dst_path))
    try:
        with con:
            for table in ("daily_state", "positions", "fills"):
                con.execute(f"DELETE FROM {table} WHERE date >= ?", (cutoff,))
        row = con.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_state").fetchone()
        n_pos = con.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        n_fills = con.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    finally:
        con.close()
    return {
        "days": int(row[0] or 0),
        "first_date": row[1],
        "last_date": row[2],
        "positions": int(n_pos or 0),
        "fills": int(n_fills or 0),
        "cutoff": cutoff,
    }
