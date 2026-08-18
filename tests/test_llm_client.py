"""LLM client — provider normalization (B1) + quick/deep dual-model routing (B2)."""

from __future__ import annotations

import sys
import types

from src.config import Config
from src.llm_client import (
    LLMRouter,
    OpenAICompatibleBackend,
    build_llm_backend,
    build_llm_router,
    resolve_provider,
)


def test_resolve_provider_default_and_known():
    assert resolve_provider(None)["base_url"] == "https://api.deepseek.com/v1"
    assert resolve_provider("deepseek")["key_env"] == "DEEPSEEK_API_KEY"
    assert resolve_provider("OPENAI")["base_url"] == "https://api.openai.com/v1"
    assert resolve_provider("moonshot")["base_url"] == "https://api.moonshot.cn/v1"
    # unknown name falls back to the default gateway (DeepSeek)
    assert resolve_provider("not-a-provider")["base_url"] == "https://api.deepseek.com/v1"


def test_build_llm_backend_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = Config({"routing": {"api": {"api_key": "", "provider": "deepseek"}}})
    assert build_llm_backend(cfg) is None


def test_build_llm_backend_tier_selection(monkeypatch):
    # fake `openai` module so `import openai` in the factory succeeds — the
    # backend itself only imports OpenAI inside `.complete()`, which we never call.
    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = Config({
        "routing": {
            "api": {"api_key": "sk-test", "provider": "deepseek", "default_model": "default-model"},
            "quick": {"model": "quick-model"},
            "deep": {"model": "deep-model"},
        }
    })
    q = build_llm_backend(cfg, tier="quick")
    d = build_llm_backend(cfg, tier="deep")
    assert q is not None and q.model == "quick-model"
    assert d is not None and d.model == "deep-model"
    # provider registry fills the endpoint; model_override wins over the tier
    assert q.base_url == "https://api.deepseek.com/v1"
    over = build_llm_backend(cfg, tier="deep", model_override="override-model")
    assert over.model == "override-model"


def test_llm_router_pick_degrades_to_available_tier():
    quick = OpenAICompatibleBackend(model="quick-model", api_key="k")
    deep = OpenAICompatibleBackend(model="deep-model", api_key="k")
    both = LLMRouter(quick=quick, deep=deep)
    assert both.pick("quick") is quick
    assert both.pick("deep") is deep
    assert both.get("deep") is deep  # alias
    # degrade: deep missing -> quick; quick missing -> deep
    assert LLMRouter(quick=quick, deep=None).pick("deep") is quick
    assert LLMRouter(quick=None, deep=deep).pick("quick") is deep


def test_build_llm_router_offline_returns_empty_router(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = Config({"routing": {"api": {"api_key": "", "provider": "deepseek"}}})
    router = build_llm_router(cfg)
    assert router.quick is None
    assert router.deep is None
    assert router.pick("deep") is None
