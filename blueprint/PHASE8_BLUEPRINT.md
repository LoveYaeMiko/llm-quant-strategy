# Implementation Blueprint: Phase 8 — Formal Factor Research

> **Target**: Claude Code (or AI Agent)  
> **Current State**: Phase 7 complete — 165 tests pass, 12,480,696 price records ingested, B1–B5 all green, `data.real_data=true`, mining smoke test passed.  
> **Goal**: Run formal factor mining (50 iterations × 20 trials), produce first batch of deployable Alpha factors validated via walk-forward, and establish factor pool management discipline.  
> **Prerequisite**: Read `PHASE7_REPORT.md` (including Appendix A) — Phase 7.1–7.5 all executed. This blueprint picks up at Phase 8.

---

## 1. Pre-flight Checklist

Before starting Phase 8, confirm these conditions:

```bash
# 1. Verify real_data is true
grep "real_data" configs/master_config.yaml
# Expected: real_data: true

# 2. Verify database has real data
psql -U pit -d pit_data -c "SELECT COUNT(*) FROM pit_records WHERE payload->>'record_type' = 'price';"
# Expected: > 10,000,000

# 3. Verify DeepSeek API key is set
echo $DEEPSEEK_API_KEY
# Expected: non-empty

# 4. Verify cost tracking is enabled
python -m src.cli cost --monthly
# Expected: < $10 projected

# 5. Run a quick sanity check
python -m src.cli verify --mode backfill
# Expected: 8/8 PASS (pit, fincad, diversity, cost, B1-B5)
```

**If any check fails, stop and resolve before proceeding.**

---

## 2. Phase 8.1 — Formal Factor Mining (50 Iterations)

### 2.1 Configuration Review

Ensure `configs/master_config.yaml` has these settings:

```yaml
research:
  train_start: "2010-01-01"
  train_end: "2019-12-31"
  val_start: "2020-01-01"
  val_end: "2021-12-31"
  test_start: "2022-01-01"
  test_end: "2025-12-31"
  universe: "hs300_500"           # Bounded for research efficiency

factor_mining:
  iterations: 50
  trials: 20                       # MCTS trials per iteration
  ic_threshold: 0.02
  rank_ic_threshold: 0.035
  icir_keep_threshold: 0.30       # Q5 decision
  max_ast_depth: 30
  min_ast_distance: 0.40          # Diversity gate

llm:
  generator_model: "deepseek-v3"   # Signal Agent
  code_model: "deepseek-v3"        # Code Agent
  critic_model: "deepseek-v3"      # Reflection Agent
  temperature: 0.0                 # Deterministic code generation
  max_tokens: 4096

risk_management:
  max_sharpe_drawdown: 0.15
  max_position_pct: 0.05
```

### 2.2 Run Formal Mining

```bash
# Background execution (recommended for 50 iterations)
nohup python -m src.cli mine \
    --config configs/master_config.yaml \
    --iterations 50 \
    --trials 20 \
    --memory-path ./outputs/memory_state.pkl \
    --output ./outputs/factor_pool_raw.json \
    --log-file ./logs/mine_50_$(date +%Y%m%d_%H%M%S).log \
    > ./logs/mine_50_$(date +%Y%m%d_%H%M%S).out 2>&1 &

# Capture PID for monitoring
echo $! > ./logs/mine.pid

# Monitor progress
tail -f ./logs/mine_*.out
```

### 2.3 Progress Monitoring

Check these indicators every 10 iterations:

```bash
# 1. View latest accepted factors
cat ./outputs/factor_pool_raw.json | jq '.factors[-5:]'

# 2. Check LLM cost so far
python -m src.cli cost --monthly

# 3. Check memory state (search trajectory)
python -c "import pickle; m=pickle.load(open('./outputs/memory_state.pkl','rb')); print(f'Successes: {len(m.success_trajectories)}, Failures: {len(m.failure_trajectories)}')"
```

### 2.4 Expected Output

After 50 iterations:

| Metric | Expected Range |
|--------|---------------|
| Total factor candidates | 450-550 |
| Training IC > 0.02 | 50-100 |
| Training IC > 0.04 | 10-20 |
| Validation IC > 0.02 | 10-30 |
| Final factor pool | 5-15 |
| LLM cost | < $15 |
| Runtime | 2-4 hours |

**Critical**: If after 30 iterations < 3 factors pass validation IC > 0.02, stop and diagnose:

```bash
# Stop the job
kill $(cat ./logs/mine.pid)

# Check logs for patterns
grep "rejected" ./logs/mine_*.log | head -20
grep "accepted" ./logs/mine_*.log | head -20
```

---

## 3. Phase 8.2 — Factor Pool Management

### 3.1 Filter Candidates

```bash
# 1. Extract factors that pass validation gate
python -m src.cli pool filter \
    --input ./outputs/factor_pool_raw.json \
    --min-val-ic 0.02 \
    --min-icir 0.30 \
    --output ./outputs/factor_pool_filtered.json

# 2. Check filtered count
cat ./outputs/factor_pool_filtered.json | jq '.factors | length'
# If < 5, consider lowering threshold to 0.015 and re-run
```

### 3.2 Diversity Screening

```bash
# 3. Apply AST diversity constraint
python -m src.cli pool diversify \
    --input ./outputs/factor_pool_filtered.json \
    --min-distance 0.40 \
    --output ./outputs/factor_pool_diverse.json

# 4. Generate diversity report
python -m src.cli pool report \
    --input ./outputs/factor_pool_diverse.json \
    --output ./outputs/diversity_report.html
```

### 3.3 Factor Metadata Enrichment

Each factor in the final pool should include:

```json
{
  "name": "factor_momentum_quality_v2",
  "schema": {
    "event": "PriceMomentum",
    "context": "SidewaysMarket",
    "qualities": ["Momentum", "Quality"],
    "direction": "Long"
  },
  "code": "def factor_momentum_quality_v2(symbols, as_of_date): ...",
  "performance": {
    "train_ic": 0.035,
    "train_icir": 0.85,
    "val_ic": 0.028,
    "val_icir": 0.62,
    "turnover": 0.35,
    "correlation": 0.22
  },
  "status": "candidate",
  "created_at": "2026-08-10T14:32:18Z"
}
```

---

## 4. Phase 8.3 — Multi-Factor Combination Backtest

### 4.1 Run Combination Backtest

```bash
# 1. Equally weighted (baseline)
python -m src.cli backtest \
    --factor-pool ./outputs/factor_pool_diverse.json \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --weights equal \
    --neutralize industry,size \
    --output ./outputs/backtest_equal.html

# 2. ICIR-weighted (recommended)
python -m src.cli backtest \
    --factor-pool ./outputs/factor_pool_diverse.json \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --weights icir_weighted \
    --neutralize industry,size \
    --output ./outputs/backtest_icir.html

# 3. Dynamic weighting (adaptive, if enough factors)
python -m src.cli backtest \
    --factor-pool ./outputs/factor_pool_diverse.json \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --weights dynamic \
    --neutralize industry,size \
    --output ./outputs/backtest_dynamic.html
```

### 4.2 Success Criteria

| Metric | Target | Minimum |
|--------|--------|---------|
| Sharpe Ratio | > 1.5 | > 1.0 |
| Annual Return | > 15% | > 10% |
| Max Drawdown | < 15% | < 20% |
| Win Rate | > 55% | > 50% |
| Monthly Positive | > 60% | > 55% |

**If Sharpe < 1.0**:
- Check correlation matrix (are factors too similar?)
- Check turnover (is it too high?)
- Consider reducing factor count (keep top 5)

### 4.3 Promotion Decision

```bash
# If backtest passes, promote to deployable
python -m src.cli pool promote \
    --input ./outputs/factor_pool_diverse.json \
    --backtest ./outputs/backtest_icir.json \
    --min-sharpe 1.0 \
    --output ./outputs/factors_deployable.json

# Mark with metadata
cat ./outputs/factors_deployable.json | jq '.status = "deployable"' > ./outputs/factors_deployable_final.json
```

---

## 5. Phase 8.4 — Decay Monitoring

### 5.1 Run Decay Monitor

```bash
# Monitor all deployable factors on out-of-sample window
python -m src.cli monitor \
    --watchlist ./outputs/factors_deployable_final.json \
    --window 90 \
    --threshold 0.30 \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --output ./outputs/monitor_report.html
```

### 5.2 Expected Output

Each factor should show:
- Rolling ICIR > 0.30 over 90-day windows
- No monotonic decline trend
- Current IC > 0.015

**If any factor shows decay**:
```bash
# Flag for review
python -m src.cli pool flag \
    --input ./outputs/factors_deployable_final.json \
    --monitor ./outputs/monitor_report.json \
    --output ./outputs/factors_with_decay.json
```

---

## 6. Phase 8 Deliverables

At the end of Phase 8, the following files should exist:

```
outputs/
├── factor_pool_raw.json          # All 50 iterations candidates
├── factor_pool_filtered.json     # Validation gate passed
├── factor_pool_diverse.json      # Diversity screened
├── factors_deployable_final.json # ⭐ Final deployable factors
├── backtest_equal.html           # Equal-weighted backtest report
├── backtest_icir.html            # ⭐ ICIR-weighted backtest report
├── backtest_dynamic.html         # Dynamic-weighted backtest report
├── diversity_report.html         # AST diversity analysis
├── monitor_report.html           # Decay monitoring report
├── memory_state.pkl              # AlphaMemo state (reusable)
└── factor_metadata.json          # All factor metadata (structured)
```

---

## 7. Phase 8 Success Criteria Checklist

After completing this blueprint, confirm:

- [ ] `mine --iterations 50` completed without errors
- [ ] At least 10 factors passed training IC > 0.02
- [ ] At least 5 factors passed validation IC > 0.02
- [ ] Final diverse factor pool has ≥ 5 factors (AST distance ≥ 0.40)
- [ ] ICIR-weighted backtest Sharpe > 1.0 on 2022-2025
- [ ] Max drawdown < 20%
- [ ] Decay monitoring shows all factors ICIR > 0.30
- [ ] `factors_deployable_final.json` is tagged and committed
- [ ] Phase 8 report written (see Section 9)

---

## 8. Phase 9 Preview (Next After Phase 8)

When Phase 8 completes successfully, Phase 9 will add **textual signals**:

```yaml
# Phase 9 tasks (planned)
- Enable TriAgent news sentiment (FinBERT + LLM)
- Expand fundamentals ingestion (--fundamentals all)
- Combine textual + price factors
- Paper trading simulation
```

**Phase 9 start condition**: `factors_deployable_final.json` has ≥ 5 factors and Sharpe > 1.0.

---

## 9. Phase 8 Report Template

After completing Phase 8, create `PHASE8_REPORT.md`:

```markdown
# Phase 8 报告 — 正式因子研究

> 项目：FQA · 阶段：Phase 8（因子挖掘）
> 日期：YYYY-MM-DD · 分支：main · 测试：165+ passed

## 1. 执行摘要

- 挖掘轮次：50 迭代 × 20 试炼
- 候选因子总数：XXX
- 通过训练集 IC > 0.02：XXX
- 通过验证集 IC > 0.02：XXX
- 最终因子池：XX 个因子

## 2. 因子质量分布

| IC 区间 | 训练集数量 | 验证集数量 |
|---------|-----------|-----------|
| 0.02–0.03 | XX | XX |
| 0.03–0.04 | XX | XX |
| > 0.04 | XX | XX |

## 3. 多因子组合表现

| 指标 | 等权 | ICIR 加权 | 动态加权 |
|------|------|-----------|---------|
| Sharpe | X.XX | X.XX | X.XX |
| 年化收益 | XX% | XX% | XX% |
| 最大回撤 | XX% | XX% | XX% |

## 4. 因子列表

| # | 名称 | 训练 IC | 验证 IC | ICIR | 换手率 | 描述 |
|---|------|--------|--------|------|--------|------|
| 1 | factor_xxx | 0.035 | 0.028 | 0.62 | 35% | 动量+质量复合 |

## 5. 衰减监控

所有因子在 2022–2025 样本外窗口 ICIR > 0.30，无显著衰减。

## 6. 结论与下一步

Phase 8 完成。正式进入 Phase 9（文本信号接入）。
```

---

## 10. Troubleshooting Guide

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| Mining returns < 3 accepted factors | IC threshold too high | Lower `ic_threshold` to 0.015 temporarily |
| Mining hangs mid-run | Network/API timeout | Check `--resume` support; memory state saves progress |
| Factor AST distance < 0.4 | LLM generating similar structures | Increase `temperature` for Signal Agent or inject more diverse seed prompts |
| Backtest Sharpe < 1.0 | Factors not diversifying | Check correlation matrix; consider reducing factor count |
| Decay monitor flags all factors | Window too short (90 days) | Extend window to 180 days for low-frequency factors |
| LLM cost exceeds $15 for 50 iterations | Context caching not effective | Check `CostTracker`; DeepSeek context caching should reduce 90% |

---

## 11. Final Prompt to Claude Code

> **Claude Code**, execute Phase 8 blueprint in full. Follow this order strictly:
>
> 1. **Pre-flight check** (Section 1) — confirm all conditions pass.
> 2. **Run formal mining** (Section 2) — `mine --iterations 50 --trials 20` in background. Monitor every 10 iterations.
> 3. **If mining completes** with ≥ 5 validation-accepted factors, proceed to pool management (Section 3).
> 4. **Run combination backtest** (Section 4) — use ICIR-weighted as primary, compare with equal.
> 5. **If Sharpe > 1.0**, promote factors to deployable and run decay monitoring (Section 5).
> 6. **Generate Phase 8 report** (Section 9) with all metrics.
> 7. **Tag milestone** — `git tag phase8-factor-research` after all checks pass.
> 8. **Report back** with the final factor count, backtest Sharpe, and any issues encountered.
>
> **Critical**: If at any point fewer than 3 factors pass validation, stop and diagnose. Do not proceed to backtest with insufficient factors.

---

**Blueprint version**: 1.0  
**Created**: 2026-08-10  
**Based on**: PHASE7_REPORT.md (with Appendix A), Grilling Round-1 decisions  
**Prerequisite**: Phase 7 complete (165 tests, 12.48M price bars, B1-B5 green)  
**Next review**: After Phase 8 completes, move to Phase 9 (textual signals)