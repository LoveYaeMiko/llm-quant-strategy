import sqlite3
import sys

path = sys.argv[1]
conn = sqlite3.connect(path)
cur = conn.cursor()
print("raw daily_state 06-22..07-14:")
for r in cur.execute(
    "SELECT date, cash, equity, gross_exposure, n_positions, n_fills, commission FROM daily_state "
    "WHERE date BETWEEN '2026-06-22' AND '2026-07-14' ORDER BY date"
).fetchall():
    print("  ", r)
conn.close()
