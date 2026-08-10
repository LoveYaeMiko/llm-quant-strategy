"""Configuration loading — single source of truth (blueprint §3).

Every IC gate, lookback and execution limit is edited in the YAML files under
`configs/`; this module loads them, interpolates ``${ENV_VAR}`` references and
exposes typed dotted-path access. Editing a gate here never requires touching
source code.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_DOTENV_COMMENT = re.compile(r"^\s*(?:#.*)?$")
_DOTENV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _load_dotenv(path: str | Path = ROOT / ".env") -> None:
    """Minimal .env loader (stdlib). Existing env vars win; no dependency.

    Values may be quoted; quotes are stripped. Kept deliberately small —
    python-dotenv is not required to run the project.
    """
    p = Path(path)
    if not p.is_file():
        return
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if _DOTENV_COMMENT.match(line):
                continue
            m = _DOTENV_LINE.match(line)
            if not m:
                continue
            key, value = m.group(1), m.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value


def _interpolate_env(value: Any) -> Any:
    """Replace ``${VAR}`` in strings with the environment value ('' if unset)."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


class Config:
    """Read-only view over a nested dict, accessed via dotted paths."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = _interpolate_env(data)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def require(self, path: str) -> Any:
        """Like :meth:`get` but raises for a missing or empty key.

        ``${ENV_VAR}`` interpolation yields ``""`` for an unset var, so a
        "required" key can be present-but-empty — treat that as missing too
        (otherwise a missing ALPHAFEED_API_KEY surfaces as per-batch failures).
        """
        value = self.get(path)
        if value is None or (isinstance(value, str) and not value):
            raise KeyError(f"config key missing or empty: {path!r}")
        return value

    def section(self, prefix: str) -> dict[str, Any]:
        return dict(self.get(prefix, {}) or {})

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Config({sorted(self._data)!r})"


def load_config(
    *paths: str | Path,
    default_master: bool = True,
) -> Config:
    """Load and merge YAML files in order (later files override earlier).

    With no paths given, loads the three canonical configs from ``configs/``.
    """
    if not paths and default_master:
        paths = (
            CONFIGS_DIR / "master_config.yaml",
            CONFIGS_DIR / "factor_thresholds.yaml",
            CONFIGS_DIR / "llm_routing.yaml",
        )

    _load_dotenv()  # API keys & db urls from .env, interpolated below

    merged: dict[str, Any] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config {path!s} must be a YAML mapping")
        merged.update(loaded)
    return Config(merged)
