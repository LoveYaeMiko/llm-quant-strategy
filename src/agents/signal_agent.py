"""Signal Agent — generates N independent, complementary hypotheses.

Blueprinted as the researcher: proposes factor hypotheses using the AlphaSchema
semantic space. Independence comes from two levers (review.md §2.2):

* **schema diversity** — every plan is a distinct point in the semantic space;
* **frequent-subtree avoidance** — formulas whose AST subtrees already crowd the
  memory are skipped, so the pool does not collapse onto a few template shapes
  (AlphaJungle).

LIMIT_DOWN blueprint 方案 C adds a third lever: **rejection feedback**. Every
factor the miner rejects (with its verdict, IC / Sharpe / drawdown and the
reason) is recorded and replayed into the next LLM prompt together with hard
constraints — no naive reversal (A-share limit-down continuations kill it), a
60-day minimum lookback, and the crisis-test self-check — so the miner stops
proposing the factor family that was just proven unprofitable.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

from ..bias_control.context_decoder import LLMBackend
from ..config import Config
from ..factors.code_generator import default_formula_for
from ..factors.memory_manager import MemoryManager
from ..factors.residual_memory import ResidualMemory
from ..factors.schema.validator import COMBINATION_TEMPLATES, sample_combination_template
from ..factors.semantic_space import SchemaPlan, SemanticSpace
from .base_agent import AgentContext, AgentResult, BaseAgent

_CODE_FENCE = re.compile(r"```(?:json)?\s*|\s*```")


class SignalAgent(BaseAgent):
    name = "signal"

    def __init__(
        self,
        space: Optional[SemanticSpace] = None,
        memory: Optional[MemoryManager] = None,
        llm: Optional[LLMBackend] = None,
        config: Optional[Config] = None,
        n_hypotheses: int = 10,
        seed: Optional[int] = None,
        *,
        rejection_history_path: Optional[str] = None,
        feedback_enabled: bool = True,
        feedback_rounds: int = 3,
        residual_memory: Optional[ResidualMemory] = None,
    ) -> None:
        super().__init__(llm=llm, config=config)
        self.space = space or SemanticSpace()
        self.memory = memory or MemoryManager()
        self.residual_memory = residual_memory
        self.n_hypotheses = n_hypotheses
        self.feedback_enabled = bool(feedback_enabled)
        self.feedback_rounds = max(1, int(feedback_rounds))
        self.rejection_history_path = rejection_history_path
        self.rejection_history: list[dict] = []
        if rejection_history_path and Path(rejection_history_path).exists():
            try:
                self.rejection_history = json.loads(
                    Path(rejection_history_path).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                self.rejection_history = []
        self.rng = random.Random(seed)
        # every formula the template-slot generator has sampled across rounds.
        # Accepted formulas are NOT in ``rejection_history``, so without this set
        # the bounded pool would keep re-drawing the same winning formula each
        # round — inflating the acceptance count with duplicates (9/20 with only 2
        # unique formulas in the first remedy run) and failing the diversity
        # verification. Blocking all tested formulas forces each slot to explore
        # a NEW formula.
        self._tested_template_formulas: set[str] = set()

    # -- rejection feedback (LIMIT_DOWN blueprint 方案 C) --------------------

    def record_rejection(self, entry: dict) -> None:
        """Persist one rejected factor so the next LLM round can learn from it."""
        self.rejection_history.append(dict(entry))
        if self.rejection_history_path:
            try:
                Path(self.rejection_history_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.rejection_history_path).write_text(
                    json.dumps(self.rejection_history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                pass

    def _build_rejection_feedback(self) -> str:
        """Structured recap of the last ``feedback_rounds`` rejections + hard rules."""
        if not self.rejection_history:
            return "【首次运行】无历史拒绝记录。"
        recent = self.rejection_history[-self.feedback_rounds :]
        lines = [
            "【上一轮挖矿复盘】",
            f"- 累计拒绝因子数：{len(self.rejection_history)}",
            "",
            "**最近拒绝因子及原因：**",
        ]
        for item in recent:
            lines.append(f"  - 因子：`{item.get('formula')}`")
            lines.append(f"    拒绝原因：{item.get('reason', item.get('verdict', 'rejected'))}")
            lines.append(
                f"    IC={float(item.get('ic', 0.0)):.3f}, "
                f"rank_ic={float(item.get('rank_ic', 0.0)):.3f}, "
                f"Sharpe={float(item.get('sharpe', 0.0)):.2f}, "
                f"回撤={float(item.get('max_drawdown', 0.0)):.1%}"
            )
        lines += [
            "",
            "**硬约束指令：**",
            "1. 严禁生成以 TS_Rank/TS_ZScore(Close, N) 做多跌幅最深者的反转逻辑"
            "（A 股跌停连板下，这类因子在股灾中因 -10% 连板收益而巨亏——这正是上面被拒因子的共同死因）。",
            "2. 优先探索方向（全部可用 Close/Open/High/Low/Volume 表达）：低波异象"
            "（做多低波动率）、低换手（做多低成交量）、低价位（做多低股价）、缩量"
            "回调；用 Rank / cs_zscore 做截面标准化，避免原始价格或成交量水平。",
            "3. 单因子回撤必须 ≤ 15%：优先用截面标准化（Rank/cs_zscore）结构；双因子"
            "等权组合 Avg(Neg(Rank(A)), Neg(Rank(B))) 是已证实的低回撤形式。",
            "4. 时间窗口：TS_Return 的 lookback 必须 ≥ 60 日（禁止 5/10 日高频反转）。",
        ]
        return "\n".join(lines)

    def _build_positive_feedback(self) -> str:
        """Replay proven-good regions so the writer steers *toward* them.

        ``_build_rejection_feedback`` covers the negative half of the loop (what
        to avoid); this closes the positive half — the accepted pool's best
        schemas and the residual memory's highest-confidence winning edit motifs
        (AlphaMemo). Without it the writer only ever flees failure and never
        converges on what demonstrably works.
        """
        lines = ["【历史有效方向（值得继续挖掘的语义区域）】"]
        top = self.memory.top_performers(k=5, metric="rank_ic")
        if top:
            lines.append("**已接受的高 IC 因子：**")
            for t in top:
                schema = t.schema or {}
                q = ", ".join(schema.get("qualities", []))
                ev = schema.get("event", "?")
                lines.append(
                    f"  - `{t.formula}` (event={ev}, qualities={q}) "
                    f"rank_ic={float(t.metrics.get('rank_ic', 0.0)):.3f}"
                )
        if self.residual_memory is not None:
            cells = self.residual_memory.top_cells(k=5)
            if cells:
                lines.append("**被证明能提升 IC 的编辑方向：**")
                for c in cells:
                    lines.append(
                        f"  - {c['category']} 下 `{c['motif']}` 平均提升 "
                        f"{c['mean_residual']:+.3f}（置信 {c['confidence']}）"
                    )
        if not top and not (self.residual_memory is not None and self.residual_memory.top_cells(1)):
            return "【首次运行】暂无正向记忆。"
        return "\n".join(lines)

    # -- generation ---------------------------------------------------------

    def _llm_proposed_plans(self, context: AgentContext, n: int) -> list[SchemaPlan]:
        prompt = (
            "You are a quantitative researcher generating independent trading "
            "hypotheses. Propose exactly "
            f"{n} DISTINCT schema plans. Each plan is a JSON object with keys "
            '"event", "context", "qualities" (array), "direction" ("long"/"short"/'
            '"long_short"), "output" ("score"/"signal"/"rank"). Return a JSON '
            "array of such objects, nothing else.\n\n"
            f"Universe horizon: {context.as_of.date()}. Available events/contexts/"
            "qualities are the standard AlphaSchema catalogs; be diverse."
        )
        if self.feedback_enabled:
            prompt += "\n\n" + self._build_rejection_feedback()
            prompt += "\n\n" + self._build_positive_feedback()
        text = self._complete(prompt, context, temperature=0.8, max_tokens=2048)
        text = _CODE_FENCE.sub("", text).strip()
        try:
            raw = json.loads(text)
            plans = [SchemaPlan.from_dict(d) for d in raw if isinstance(d, dict)]
            return [p for p in plans if self.space.validate(p)]
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def generate_hypotheses(
        self, context: AgentContext, n: Optional[int] = None
    ) -> list[SchemaPlan]:
        n = n or self.n_hypotheses
        plans: list[SchemaPlan] = []
        if self.llm is not None:
            plans = self._llm_proposed_plans(context, n)

        while len(plans) < n:
            candidate = self.space.sample(self.rng)
            formula = default_formula_for(candidate)
            if self.memory.has_formula(formula):
                continue
            # frequent-subtree avoidance: skip re-explored structures with a
            # probability proportional to how often they already appear
            penalty = self.memory.avoidance_penalty(formula)
            if penalty > 0 and self.rng.random() < min(penalty, 0.95):
                continue
            plans.append(candidate)
        return plans[:n]

    def generate_template_formulas(
        self, n: int, rng: Optional[random.Random] = None, skip: Optional[set] = None
    ) -> list[str]:
        """Combination-template slot sampling (validation_BLUEPRINT §3.2).

        Returns up to ``n`` distinct dual-factor equal-weight combination formulas,
        skipping any already recorded as a rejection (so a template the risk gate
        just killed is not re-proposed next iteration), any passed in ``skip``
        (so a formula the free slot already produced this iteration is not
        duplicated) AND every formula the generator has sampled before
        (``_tested_template_formulas`` — including accepted ones, so an accepted
        formula is not re-tested verbatim in a later round). Slot 0 is reserved
        for the proven 低波+低换手 direction (COMBINATION_TEMPLATES[0]) — the 0/20
        diagnosis showed it is the only family with alpha under real HS300, so
        every round must test it (with a lookback pair not yet tested). The pool
        is bounded (4 templates × ordered lookback pairs from {60,120,240}), so
        the guard loop returns whatever distinct formulas it can produce.
        """
        rng = rng or self.rng
        out: list[str] = []
        seen: set[str] = set()
        blocked = {str(x.get("formula")) for x in self.rejection_history}
        blocked |= self._tested_template_formulas
        if skip:
            blocked |= set(skip)
        # slot 0: guarantee the proven low-vol + low-turnover direction each round,
        # cycling through its untested lookback pairs. If every pair is already
        # blocked, the fallthrough while-loop fills the slot from the full pool.
        if n > 0:
            for _ in range(n * 20):
                formula = sample_combination_template(rng=rng, template=COMBINATION_TEMPLATES[0])
                if formula in seen or formula in blocked:
                    continue
                seen.add(formula)
                out.append(formula)
                self._tested_template_formulas.add(formula)
                break
        guard = 0
        while len(out) < n and guard < n * 20:
            guard += 1
            formula = sample_combination_template(rng=rng)
            if formula in seen or formula in blocked:
                continue
            seen.add(formula)
            out.append(formula)
            self._tested_template_formulas.add(formula)
        return out

    def run(self, context: AgentContext, n: Optional[int] = None) -> AgentResult:
        plans = self.generate_hypotheses(context, n)
        return AgentResult(
            agent=self.name,
            summary=f"generated {len(plans)} independent hypotheses",
            artifacts={
                "n_hypotheses": len(plans),
                "plans": [p.to_dict() for p in plans],
            },
            context=context,
        )
