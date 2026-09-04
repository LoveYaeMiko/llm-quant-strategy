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

    # ------------------------------------------------------------------ write
    def record_day(
        self,
        date: str | pd.Timestamp,
        cash: float,
        equity: float,
        positions: dict[str, float],
        fills: Iterable[Fill],
        gross_exposure: float,
    ) -> None:
        """Persist one day's end-of-day state plus its execution fills.

        Idempotent on ``date`` (``INSERT OR REPLACE``) so a re-run of the same
        day overwrites rather than duplicates — the resume guard in the runner
        skips already-recorded dates, this is a second line of defence.
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
            self._conn.execute("DELETE FROM fills WHERE date = ?", (d,))
            self._conn.executemany(
                "INSERT INTO fills (date, symbol, side, shares, price, commission, notional, time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (d, f.symbol, f.side, float(f.shares), float(f.price),
                     float(f.commission), float(f.notional), str(getattr(f, "time", "") or ""))
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

    def total_commission(self) -> float:
        row = self._conn.execute("SELECT COALESCE(SUM(commission), 0) FROM fills").fetchone()
        return float(row[0]) if row else 0.0

    def close(self) -> None:
        self._conn.close()
