"""Live trader resilience — pid-lock liveness check (recycled-pid protection)."""
from __future__ import annotations

import subprocess
import sys

import pytest

from src.live.trader import _is_live_process


def test_recycled_pid_is_not_live():
    """A dead/recycled pid must NOT block the trader (os.kill(0) alone would
    accept a recycled pid pointing at an unrelated process)."""
    # spawn a short-lived unrelated process, capture its pid, wait for death
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.5)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    pid = proc.pid
    proc.wait()
    assert not _is_live_process(pid)


def test_unrelated_live_pid_is_not_live():
    """An ALIVE pid whose command line is not `cli.py live` must not count —
    the current pytest process itself satisfies this."""
    import os

    assert not _is_live_process(os.getpid())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
