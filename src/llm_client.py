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

import os
import sys
from dataclasses import dataclass
from typing import Optional

from .config import Config

# Provider normalization (B1): name -> OpenAI-compatible endpoint + the env var
# that carries its credential. The default gateway is DeepSeek; swapping a
# provider is a one-line config change (``routing.api.provider``), never a code
# change. Deliberately no qwen entry — the project pins DeepSeek (blueprint §3.6).
PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "key_env": "MOONSHOT_API_KEY"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "key_env": "ZHIPU_API_KEY"},
}


def resolve_provider(name: Optional[str] = None) -> dict[str, str]:
    """Normalise a provider name to ``{base_url, key_env}`` (default DeepSeek)."""
    return PROVIDERS.get((name or "deepseek").strip().lower(), PROVIDERS["deepseek"])


@dataclass
class OpenAICompatibleBackend:
    """OpenAI-compatible chat backend (DeepSeek et al.)."""

    model: str
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    cost_tracker: object = None  # CostTracker — records usage when provided
    timeout_seconds: int = 60

    def complete(
        self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024,
        timeout: Optional[int] = None,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai is not installed; `pip install -e '.[llm]'` to enable the DeepSeek path"
            ) from exc

        client = OpenAI(
            api_key=self.api_key, base_url=self.base_url,
            timeout=self.timeout_seconds if timeout is None else timeout,
        )
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
            self.cost_tracker.record(self.model, tin, tout, purpose=self.model)
        return text


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4) when the API omits usage."""
    return max(1, len(text) // 4)


def build_llm_backend(
    config: Optional[Config] = None,
    cost_tracker: object = None,
    model_override: Optional[str] = None,
    tier: str = "quick",
):
    """Factory from the routing config. Returns ``None`` when no key is set.

    ``tier`` selects the model tier (B2): ``"quick"`` reads ``routing.quick.model``
    and ``"deep"`` reads ``routing.deep.model``, both falling back to
    ``routing.api.default_model``. ``None`` is the signal for the CLI to run the
    deterministic offline path — the whole pipeline works without any LLM.
    """
    from .config import Config as _Config  # local to avoid import-cycle at module load

    cfg = config or _Config({})
    api = cfg.section("routing.api") or {}
    provider = resolve_provider(api.get("provider"))
    base_url = api.get("base_url") or provider["base_url"]
    key = api.get("api_key") or os.environ.get(provider["key_env"], "")
    if not key:
        return None
    try:
        import openai  # noqa: F401  (optional dependency — degrade if absent)
    except ImportError:
        print(
            f"[llm-client] {provider['key_env']} is set but 'openai' is not installed; "
            "running OFFLINE. `pip install -e '.[llm]'` to enable the LLM path.",
            file=sys.stderr,
        )
        return None
    tier_cfg = cfg.section(f"routing.{tier}") or {}
    model = model_override or tier_cfg.get("model") or api.get("default_model", "deepseek-v4-pro")
    return OpenAICompatibleBackend(
        model=model,
        api_key=key,
        base_url=base_url,
        cost_tracker=cost_tracker,
        timeout_seconds=int(api.get("timeout_seconds", 60)),
    )


@dataclass
class LLMRouter:
    """Quick/deep dual-backend router (B2).

    ``quick`` is the cheap intermediate-reasoning model (generator / code);
    ``deep`` is the synthesis/judgment model (critic / diagnosis). Either may be
    ``None`` (offline), in which case ``pick`` degrades to the available tier.
    """

    quick: Optional[OpenAICompatibleBackend] = None
    deep: Optional[OpenAICompatibleBackend] = None

    def pick(self, tier: str = "quick") -> Optional[OpenAICompatibleBackend]:
        if (tier or "quick").lower() == "deep":
            return self.deep or self.quick
        return self.quick or self.deep

    get = pick  # alias — callers read ``router.get("deep")``


def build_llm_router(config: Optional[Config] = None, cost_tracker: object = None) -> LLMRouter:
    """Build the quick/deep pair from the routing config (B2).

    Returns a router whose tiers are ``None`` when the credential is absent, so
    callers can always ``router.pick("deep")`` and get a backend or ``None``.
    """
    return LLMRouter(
        quick=build_llm_backend(config, cost_tracker=cost_tracker, tier="quick"),
        deep=build_llm_backend(config, cost_tracker=cost_tracker, tier="deep"),
    )
