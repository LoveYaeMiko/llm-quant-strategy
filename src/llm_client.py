"""DeepSeek API client (OpenAI-compatible) — the production LLM backend.

``OpenAICompatibleBackend`` is a drop-in :class:`~src.bias_control.context_decoder.LLMBackend`
(the ``complete(...) -> str`` protocol) that:

* lazily imports ``openai`` so the package still installs/tests without it;
* reads ``base_url`` / ``api_key`` / ``model`` from the routing config
  (``configs/llm_routing.yaml``, itself interpolated from ``.env``);
* records token usage into a :class:`~src.cost_tracker.CostTracker` so the
  $500/month budget gate is real, not theoretical;
* degrades with a clear error when the key is missing — the CLI then falls back
  to the deterministic offline path.

The FinCAD wrapper (:mod:`src.bias_control.context_decoder`) is applied on top of
this backend by :class:`~src.agents.base_agent.BaseAgent`, so every call is
look-ahead-controlled at the caller's information horizon.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from .config import Config


@dataclass
class OpenAICompatibleBackend:
    """OpenAI-compatible chat backend (DeepSeek et al.)."""

    model: str
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    cost_tracker: object = None  # CostTracker — records usage when provided
    timeout_seconds: int = 60

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai is not installed; `pip install -e '.[llm]'` to enable the DeepSeek path"
            ) from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        if self.cost_tracker is not None:
            usage = getattr(resp, "usage", None)
            tin = int(usage.prompt_tokens) if usage is not None and usage.prompt_tokens else _estimate_tokens(prompt)
            tout = int(usage.completion_tokens) if usage is not None and usage.completion_tokens else _estimate_tokens(text)
            self.cost_tracker.record(self.model, tin, tout, purpose="deepseek-v4-flash")
        return text


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4) when the API omits usage."""
    return max(1, len(text) // 4)


def build_llm_backend(
    config: Optional[Config] = None,
    cost_tracker: object = None,
    model_override: Optional[str] = None,
):
    """Factory from the routing config. Returns ``None`` when no key is set.

    ``None`` is the signal for the CLI to run the deterministic offline path —
    the whole pipeline works without any LLM (agents have fallbacks).
    """
    from .config import Config as _Config  # local to avoid import-cycle at module load

    cfg = config or _Config({})
    api = cfg.section("routing.api") or {}
    key = api.get("api_key") or ""
    if not key:
        return None
    try:
        import openai  # noqa: F401  (optional dependency — degrade if absent)
    except ImportError:
        print(
            "[llm-client] DEEPSEEK_API_KEY is set but 'openai' is not installed; "
            "running OFFLINE. `pip install -e '.[llm]'` to enable the DeepSeek path.",
            file=sys.stderr,
        )
        return None
    return OpenAICompatibleBackend(
        model=model_override or api.get("default_model", "deepseek-v4-flash"),
        api_key=key,
        base_url=api.get("base_url", "https://api.deepseek.com/v1"),
        cost_tracker=cost_tracker,
        timeout_seconds=int(api.get("timeout_seconds", 60)),
    )
