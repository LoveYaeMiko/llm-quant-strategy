"""Cost tracking + reproducibility audit tests."""

from __future__ import annotations

import json

from src.audit import AuditRecord, ExperimentAuditor
from src.config import load_config
from src.cost_tracker import CostTracker


def test_cost_tracker_records_and_budget():
    t = CostTracker(monthly_budget_usd=100.0)
    t.record("deepseek-v4-flash", 1000, 500)
    assert t.total_cost() > 0
    assert t.under_budget()
    assert t.budget_remaining() < 100.0
    assert "deepseek-v4-flash" in t.cost_by_model()


def test_cost_monthly_projection():
    t = CostTracker(monthly_budget_usd=500.0)
    for _ in range(200):
        t.record("deepseek-v4-flash", 1200, 400)
    proj = t.monthly_projection(days_elapsed=30.0)
    assert proj < 500.0  # ~200 calls * $0.0008 ≈ $0.16


def test_cost_budget_breach():
    t = CostTracker(monthly_budget_usd=0.001)
    t.record("gpt-4o", 100_000, 100_000)
    assert t.under_budget() is False


def test_audit_record_add_factor_and_write(tmp_path):
    rec = AuditRecord(run_id="abc", name="test", created_at="2026-08-08T00:00:00Z")
    rec.add_factor({"formula": "Rank(Close)"}, {"rank_ic": 0.04}, "accepted")
    path = rec.write(tmp_path / "audit.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "abc"
    assert data["factors"][0]["verdict"] == "accepted"


def test_experiment_auditor_snapshots_config():
    cfg = load_config()
    auditor = ExperimentAuditor()
    rec = auditor.begin("mine")
    auditor.snapshot_config(rec, cfg)
    assert rec.config_hash
    assert rec.evaluation_assumptions["ic_threshold"] == 0.02
    # LIMIT_DOWN blueprint 方案 D: medium/low-frequency lookback floor
    assert rec.evaluation_assumptions["min_lookback"] == 60
    assert rec.evaluation_assumptions["max_lookback"] == 240


def test_experiment_auditor_routing_and_pit():
    cfg = load_config()
    auditor = ExperimentAuditor()
    rec = auditor.begin("mine")
    auditor.snapshot_routing(rec, cfg)
    assert rec.model_versions["generator"] == "deepseek-v4-flash"
    auditor.set_pit_window(rec, "2020-01-01", "2024-12-31", 500)
    assert rec.pit_window["universe_size"] == 500


def test_env_interpolation_from_dotenv(monkeypatch):
    # test_cli.py clears DEEPSEEK_API_KEY process-wide at import; drop it here so
    # _load_dotenv re-reads the real key from .env
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = load_config()
    key = cfg.get("routing.api.api_key")
    assert key and key.startswith("sk-")
