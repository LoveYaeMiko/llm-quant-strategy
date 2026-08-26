"""Exploration track — a parallel, non-invasive high-risk/high-reward research arm.

New operators (higher moments / volatility structure / illiquidity), a re-opened
controlled LLM hypothesis path, and a LOOSE exploration gate — all isolated from
the production mining/autopilot loop, which keeps running every day unchanged.
"""

from .run import cmd_explore, run_exploration

__all__ = ["cmd_explore", "run_exploration"]
