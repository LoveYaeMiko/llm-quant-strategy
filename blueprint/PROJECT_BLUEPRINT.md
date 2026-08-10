这是一份专为 **Claude Code**（或其他 AI 编程助手）设计的**项目落地指令文档**。

请将下文保存为 `PROJECT_BLUEPRINT.md` 放置在项目根目录。同时，请将上一轮回复中 **“六、论文与下载渠道”** 里的 PDF 链接全部下载，放入 `./papers/` 文件夹。Claude Code 将优先读取这些论文原文来指导编码。

---

# PROJECT BLUEPRINT: LLM-Driven Quantitative Trading System

> **Role**: You are **Claude Code**, the lead AI architect and engineer for this project.
> **Goal**: Build a production-ready, research-grade LLM-powered quantitative trading system based on the 2026 SOTA papers provided in `./papers/`.
> **Constraint**: Strictly separate **Offline R&D (LLM-heavy)** from **Online Execution (Deterministic)** . Minimize look-ahead bias. Prioritize reproducibility.

---

## 1. Project Directory Structure (Must Generate)

Create the following exact tree structure in the root directory:

```
project_root/
├── papers/                          # <<< DOWNLOAD ALL PDFs HERE
│   ├── AlphaSchema_arXiv2607.26642.pdf
│   ├── FinCAD_arXiv2605.24564.pdf
│   ├── EvoQuant_arXiv2607.12455.pdf
│   ├── AlphaMemo_arXiv2606.20625.pdf
│   ├── CognitiveAlphaMining_ACL2026.pdf
│   ├── AgenticAITA_arXiv2605.12532.pdf
│   └── FINSABER_KDD2026.pdf
│
├── configs/
│   ├── master_config.yaml          # Single source of truth
│   ├── factor_thresholds.yaml      # IC/RankIC gates
│   └── llm_routing.yaml            # Model mapping (cheap vs. expensive)
│
├── src/
│   ├── data/
│   │   ├── point_in_time_loader.py # PIT data to prevent look-ahead
│   │   └── text_feeds.py          # News/Transcripts ingestion
│   │
│   ├── bias_control/               # <<< CRITICAL: FinCAD integration
│   │   ├── look_ahead_detector.py
│   │   └── context_decoder.py     # FinCAD inference-time adaptation
│   │
│   ├── factors/                    # <<< AlphaSchema + CogAlpha
│   │   ├── semantic_space.py      # Event/Context/Qualities/Direction/Output
│   │   ├── schema_explorer.py     # LLM-guided MCTS over semantic space
│   │   ├── code_generator.py      # Translate schema to Python code
│   │   └── memory_manager.py      # AlphaMemo structured memory
│   │
│   ├── agents/                     # <<< Multi-Agent System
│   │   ├── base_agent.py          # LangGraph / Sleipnir base
│   │   ├── signal_agent.py        # Generates factor hypotheses
│   │   ├── code_agent.py          # Implements factor code
│   │   ├── eval_agent.py          # IC/RankIC evaluator + FinCAD wrapper
│   │   └── risk_agent.py          # EvoQuant style validator
│   │
│   ├── backtest/
│   │   ├── engine.py              # Custom PIT backtester
│   │   └── metrics.py             # IC, ICIR, Sharpe, MaxDD
│   │
│   └── online/                     # <<< TiMi Paradigm: Deterministic execution
│       ├── signal_calculator.py   # C++/Rust compiled factors (no LLM)
│       ├── portfolio_optimizer.py # PCA neutralization
│       └── order_executor.py      # Market simulation
│
├── tests/                          # Unit tests for every module
├── .env.example                    # API keys (OpenAI, Finnhub, etc.)
├── pyproject.toml
└── README.md
```

---

## 2. Implementation Workflow (Step-by-Step for Claude Code)

### Phase 1: Data & Bias Control (Anti-Look-Ahead)

1.  **Read `FINSABER_KDD2026.pdf`** (focus on Section 3: Bias Traps).
2.  **Implement `point_in_time_loader.py`**:
    - Fetch data as it existed at timestamp `T`. Ensure delisted stocks are included.
    - Create a SQLite/Postgres schema with `valid_from` and `valid_to` fields.
3.  **Implement `look_ahead_detector.py`**:
    - Parse **`FinCAD_arXiv2605.24564.pdf`** .
    - Code the "Context-Aware Decoding" algorithm: During LLM inference, suppress the model's attention to tokens/dates that occur after the current timestamp `T`.
    - *Hint: Modify the logits during generation to penalize future-date mentions.*

### Phase 2: Semantic Factor Mining (AlphaSchema + AlphaMemo)

1.  **Read `AlphaSchema_arXiv2607.26642.pdf`** .
2.  **Build `semantic_space.py`**:
    - Define the 5 core components of a schema plan:
      - `Event`: (e.g., "Earnings Surprise")
      - `Context`: (e.g., "Bull Market", "High Volatility")
      - `Qualities`: (e.g., "Momentum", "Mean Reversion")
      - `Direction`: (Long/Short)
      - `Output`: (Continuous score or Binary signal)
3.  **Read `AlphaMemo_arXiv2606.20625.pdf`** .
4.  **Build `memory_manager.py`**:
    - Implement structured memory to store search trajectories.
    - Implement the "Frequent Subtree Avoidance" mechanism (from Alpha Jungle) to ensure schema diversity.

### Phase 3: Multi-Agent Orchestration (Sleipnir / TradingAgents)

1.  **Read `AgenticAITA_arXiv2605.12532.pdf`** .
2.  **Build the core loop in `agents/`**:
    - **Signal Agent**: Prompted to generate 10 independent, complementary hypotheses using the Schema space.
    - **Code Agent**: Temperature = 0.0. Translates Schema into Python code using the predefined 66+ operator library.
    - **Eval Agent**: Runs backtest. Calculates IC and RankIC.
    - **Gatekeeper**: Implement the *Adaptive Z-Score Trigger Engine* from AgenticAITA. Only call the heavy LLM when statistical anomalies occur (节省推理成本).

### Phase 4: Strategy Self-Evolution (EvoQuant)

1.  **Read `EvoQuant_arXiv2607.12455.pdf`** .
2.  **Build `evolver.py`**:
    - LLM diagnoses performance bottlenecks (e.g., "Factor decays after 5 days").
    - Generate semantic candidate edits.
    - Multi-stage validation pipeline (Overfitting check -> Robustness check).
    - Distill successful experiences back into `memory_manager.py`.

### Phase 5: Online Deployment (The "TiMi" Separation)

1.  **Factor Distillation**:
    - Once a factor passes the thresholds (`IC > 0.04`), automatically export it to `online/signal_calculator.py`.
    - **CRITICAL**: The online module must have **ZERO** imports from `transformers` or `torch`. It must be pure NumPy/Pandas (or C++ bindings).
2.  **Implementation**:
    - Write `online/portfolio_optimizer.py` to perform PCA neutralization (removing market, size, and industry factors) to keep the Alpha pure.

---

## 3. Configuration Files (Master YAML)

Create `configs/master_config.yaml` exactly as follows:

```yaml
project:
  name: "LLM_Quant_2026"
  start_date: "2015-01-01"
  end_date: "2025-12-31" # Prevent using data after 2025 for training

data:
  pit_database_url: "postgresql://user:pass@localhost:5432/pit_data"
  news_api_key: ${NEWS_API_KEY}

llm_routing:
  generator_model: "deepseek-v3"      # Cheap for schema generation
  code_model: "gpt-4o"               # Accurate for code
  critic_model: "claude-3.5-sonnet"  # For EvoQuant diagnosis
  max_tokens: 4096

factor_mining:
  ic_threshold: 0.02                 # Minimum to keep
  rank_ic_threshold: 0.035
  max_lookback: 60                   # days
  min_lookback: 5

risk_management:
  max_sharpe_drawdown: 0.15
  market_state_bins: ["bull", "bear", "sideways"]

online_execution:
  latency_target_ms: 10              # Must execute fast
  max_position_pct: 0.05
```

---

## 4. Critical Implementation Notes for Claude Code

### A. How to Read Papers
When you start writing code for a specific module, **always open the corresponding PDF in `./papers/` first**.
- Extract the pseudo-code directly.
- If the pseudo-code is missing, look for the "Algorithm 1" or "Methodology" sections to infer the exact logic.

### B. The Operator Library (for Code Agent)
Hardcode the `operator_library` in `code_generator.py` to prevent hallucinations. Include:
- Arithmetic: `add`, `sub`, `mul`, `div`
- Time Series: `ts_mean`, `ts_std`, `ts_rank`, `ts_return`, `ts_decay`
- Cross Section: `cs_rank`, `cs_zscore`, `cs_neutralize`

### C. Anti-Overfitting Logic (FINSABER compliance)
- **Survivorship Bias**: When testing on S&P 500, ensure you query the `point_in_time_loader` for the *historical* constituents.
- **Data Snooping**: If you generate 100 factors and pick the best 5, you must apply a **multiple hypothesis testing correction** (e.g., Bonferroni) to the Sharpe ratio.

---

## 5. Setup & Execution Commands

Claude Code, please run these shell commands to bootstrap the environment:

```bash
# 1. Environment Setup
python -m venv venv
source venv/bin/activate
pip install -e . # Install in editable mode

# 2. Dependencies (from pyproject.toml)
# Ensure you include: langgraph, openai, pandas, numpy, scikit-learn, torch, transformers, backtrader (or vectorbt)
```

### Running the System
- **Offline Mining**: `python cli.py mine --config configs/master_config.yaml`
- **Backtest**: `python cli.py backtest --factor-pool ./outputs/factors.json`
- **Export to Online**: `python cli.py export --compiled-path ./online/compiled_factors.py`

---

## 6. Instructions for the Human User

> *I am delivering this to my human collaborator.*

1.  **Download Papers**: Please download all PDFs listed in the "Paper Download Channel" section of the previous analysis and put them in the `papers/` folder.
2.  **API Keys**: Create a `.env` file with your OpenAI/DeepSeek and financial data API keys.
3.  **GPUs**: Ensure access to at least 1x A100 for fine-tuning (though initially, we rely on API calls to frontier models to save time).

---

## 7. Verification Checklist (Claude Code must confirm)

After writing the base code, run these validation prompts internally and report back to the user:

- [ ] **PIT Check**: Does the `point_in_time_loader` correctly reject data from `2024-01-01` when the query timestamp is `2023-12-31`?
- [ ] **FinCAD Check**: Does the `look_ahead_detector` reduce the IC of "cheating" factors by > 50%?
- [ ] **Diversity Check**: Are the generated Alpha formulas structurally distinct? (Check AST tree distance).
- [ ] **Cost Check**: Is the estimated monthly LLM API cost under $500? (If not, increase reliance on `code_agent` caching).

---

## 8. Final Prompt to Claude Code

> **Claude Code, your task is to parse this blueprint.**
> Start by reading `papers/FinCAD_arXiv2605.24564.pdf` and `papers/AlphaSchema_arXiv2607.26642.pdf` to understand the 2026 SOTA. Then, write the core classes for `semantic_space.py` and `look_ahead_detector.py`.
> Do not write placeholder code. Write production-grade, type-hinted Python with extensive docstrings explaining the financial logic.
> Once the core is done, write the unit tests in `tests/`.
> 
> **Begin.**

---
**End of Blueprint**