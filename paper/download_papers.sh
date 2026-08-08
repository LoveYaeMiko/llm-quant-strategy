#!/usr/bin/env bash
# Parallel downloader for papers/webpages/repos referenced in resource/
# Concurrency 5; skips files already downloaded; logs to download_log.txt
set -u

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPERS="$PAPER_DIR/papers"
WEBPAGES="$PAPER_DIR/webpages"
REPOS="$PAPER_DIR/repos"
LOG="$PAPER_DIR/download_log.txt"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
CONCURRENCY=5

mkdir -p "$PAPERS" "$WEBPAGES" "$REPOS"
: > "$LOG"

# fetch <outfile> <url>  -- resumes partial .tmp via -C - ; verifies %PDF for .pdf targets
fetch() {
  local out="$1" url="$2"
  if [ -s "$out" ]; then
    echo "SKIP  $(basename "$out")  (exists)"
    return 0
  fi
  local args=(-sL --fail --retry 6 --retry-delay 4 --retry-connrefused --max-time 600 -A "$UA")
  if [ -s "$out.tmp" ]; then
    args+=(-C -)   # resume partial download instead of restarting
  fi
  if curl "${args[@]}" -o "$out.tmp" "$url" 2>/dev/null; then
    if [ -s "$out.tmp" ]; then
      case "$out" in
        *.pdf)
          if head -c 5 "$out.tmp" | grep -q '%PDF'; then
            mv "$out.tmp" "$out"
            echo "OK    $(basename "$out")  <-  $url"
            return 0
          else
            rm -f "$out.tmp"
            echo "FAIL  $(basename "$out")  (not a PDF)  <-  $url"
            return 1
          fi
          ;;
        *)
          mv "$out.tmp" "$out"
          echo "OK    $(basename "$out")  <-  $url"
          return 0
          ;;
      esac
    fi
  fi
  echo "FAIL  $(basename "$out")  <-  $url"
  return 1
}

export -f fetch
export PAPER_DIR PAPERS WEBPAGES LOG UA

run_parallel() {
  # stdin: "outfile|url" lines
  xargs -P "$CONCURRENCY" -d '\n' -I{} bash -c '
    line="$1"
    out="${line%%|*}"
    url="${line#*|}"
    fetch "$out" "$url" >> "$LOG"
  ' _ {}
}

# ---------- 1. arXiv papers (PDF) ----------
echo "==== arXiv papers ====" >> "$LOG"
cat <<EOF | run_parallel
$PAPERS/AgenticAITA_2605.12532.pdf|https://arxiv.org/pdf/2605.12532
$PAPERS/ContestTrade_2508.00554v4.pdf|https://arxiv.org/pdf/2508.00554v4
$PAPERS/AlphaSchema_2607.26642.pdf|https://arxiv.org/pdf/2607.26642
$PAPERS/AlphaMemo_2606.20625.pdf|https://arxiv.org/pdf/2606.20625
$PAPERS/QuantaAlpha_2602.16789.pdf|https://arxiv.org/pdf/2602.16789
$PAPERS/TriAgent_2607.19794.pdf|https://arxiv.org/pdf/2607.19794
$PAPERS/FINSABER_2505.07078.pdf|https://arxiv.org/pdf/2505.07078
$PAPERS/FinCAD_2605.24564.pdf|https://arxiv.org/pdf/2605.24564
$PAPERS/EvoQuant_2607.12455.pdf|https://arxiv.org/pdf/2607.12455
$PAPERS/BeyondAgentArch_2606.08285.pdf|https://arxiv.org/pdf/2606.08285
$PAPERS/AgenticTradingSurvey_2605.19337.pdf|https://arxiv.org/pdf/2605.19337
$PAPERS/AlphaCrafter_2605.05580.pdf|https://arxiv.org/pdf/2605.05580
$PAPERS/AutomateStrategyFinding_2409.06289.pdf|https://arxiv.org/pdf/2409.06289v3
$PAPERS/MLMultiFactorBiasCorrection_2507.07107.pdf|https://arxiv.org/pdf/2507.07107v1
$PAPERS/CogAlpha_2511.18850.pdf|https://arxiv.org/pdf/2511.18850v3
EOF

# ---------- 2. Other publishers (PDF) ----------
echo "==== Other publishers ====" >> "$LOG"
cat <<EOF | run_parallel
$PAPERS/CognitiveAlphaMining_ACL2026.pdf|https://aclanthology.org/2026.acl-long.538.pdf
$PAPERS/HybridSentiment_MDPI_AI-07-00138.pdf|https://mdpi-res.com/d_attachment/ai/ai-07-00138/article_deploy/ai-07-00138.pdf
$PAPERS/NavigatingAlphaJungle_AAAI2026.pdf|https://doi.org/10.1609/aaai.v40i2.37069
$PAPERS/Sleipnir_IEEE11394768.pdf|https://ieeexplore.ieee.org/abstract/document/11394768
$PAPERS/EvoAlpha_IEEE11463591.pdf|https://ieeexplore.ieee.org/document/11463591
$PAPERS/FinSentLLM_IEEE11461632.pdf|https://ieeexplore.ieee.org/abstract/document/11461632
$PAPERS/FactorMAD_ACM.pdf|https://dl.acm.org/doi/pdf/10.1145/3768292.3770377
$PAPERS/AlphaAgent_Fmread|https://www.fmread.com/pdfshare/nd6ozy
EOF

# ---------- 3. Webpages (HTML) ----------
echo "==== Webpages ====" >> "$LOG"
cat <<EOF | run_parallel
$WEBPAGES/guolianmingsheng_2026-07-16_ai-touyan.html|https://finance.sina.com.cn/wm/2026-07-16/doc-inihyyvy4515788.shtml
$WEBPAGES/huaan_alpha-factor.html|http://stockfinance.sina.cn/stock/go.php/paper/reportid/833644386223/index.phtml
$WEBPAGES/guangfa_automatic-factor-ai.html|https://stockfinance.sina.cn/stock/go.php/paper/reportid/833691942180/index.phtml
$WEBPAGES/nvidia_multi-agent-signal.html|https://developer.nvidia.com/blog/automating-and-optimizing-financial-signal-discovery-with-multi-agent-systems
$WEBPAGES/dongfang_quantaalpha.html|http://brow.fygsoft.com/data/09107bf9e2075b6fc3f34f0742a3dfde.html
$WEBPAGES/guoxin_asset-allocation.html|https://www.fxbaogao.com/detail/5120107
$WEBPAGES/mentalmomentum_automating-trading-rules.html|https://research.mental-momentum.ai/r/automating-trading-rules-research-using-699apm
$WEBPAGES/mentalmomentum_llm-trading-analysts.html|https://research.mental-momentum.ai/r/large-language-models-trading-alpha-kgicbo
EOF

# ---------- 4. arXiv abstract pages (HTML, for reference) ----------
echo "==== arXiv abstract pages ====" >> "$LOG"
cat <<EOF | run_parallel
$WEBPAGES/arxiv_abs_2605.12532.html|https://arxiv.org/abs/2605.12532
$WEBPAGES/arxiv_abs_2508.00554.html|https://arxiv.org/abs/2508.00554
$WEBPAGES/arxiv_abs_2607.26642.html|https://arxiv.org/abs/2607.26642
$WEBPAGES/arxiv_abs_2606.20625.html|https://arxiv.org/abs/2606.20625
$WEBPAGES/arxiv_abs_2602.16789.html|https://arxiv.org/abs/2602.16789
$WEBPAGES/arxiv_abs_2607.19794.html|https://arxiv.org/abs/2607.19794
$WEBPAGES/arxiv_abs_2505.07078.html|https://arxiv.org/abs/2505.07078
$WEBPAGES/arxiv_abs_2605.24564.html|https://arxiv.org/abs/2605.24564
$WEBPAGES/arxiv_abs_2607.12455.html|https://arxiv.org/abs/2607.12455
$WEBPAGES/arxiv_abs_2606.08285.html|https://arxiv.org/abs/2606.08285
$WEBPAGES/arxiv_abs_2605.19337.html|https://arxiv.org/abs/2605.19337
$WEBPAGES/arxiv_abs_2605.05580.html|https://arxiv.org/abs/2605.05580
$WEBPAGES/arxiv_abs_2409.06289.html|https://arxiv.org/abs/2409.06289
$WEBPAGES/arxiv_abs_2507.07107.html|https://arxiv.org/abs/2507.07107
$WEBPAGES/arxiv_abs_2511.18850.html|https://arxiv.org/abs/2511.18850
EOF

# ---------- 5. GitHub / code repos ----------
echo "==== Code repos ====" >> "$LOG"
clone() {
  local name="$1" url="$2"
  if [ -d "$REPOS/$name/.git" ] || [ -n "$(ls -A "$REPOS/$name" 2>/dev/null)" ]; then
    echo "SKIP  $name (exists)" >> "$LOG"
    return 0
  fi
  if git clone --depth 1 --single-branch --quiet "$url" "$REPOS/$name" 2>>"$LOG"; then
    echo "OK    $name  <-  $url" >> "$LOG"
  else
    echo "FAIL  $name  <-  $url" >> "$LOG"
    rm -rf "$REPOS/$name"
  fi
}
clone TradingAgents "https://github.com/TauricResearch/TradingAgents.git"
clone AlphaMemo "https://github.com/jarrettyu/AlphaMemo.git"
clone AlphaSchema "https://github.com/JingyangYi/AlphaSchema.git"
clone AlphaAgent "https://github.com/RndmVariableQ/AlphaAgent.git"
clone EvoQuant "https://anonymous.4open.science/r/EVOQUANT.git"
clone vibe-trading "https://github.com/HKUDS/vibe-trading.git"

echo "==== Done ====" >> "$LOG"
echo "=== Summary ==="
echo "OK:   $(grep -c '^OK' "$LOG")"
echo "FAIL: $(grep -c '^FAIL' "$LOG")"
echo "SKIP: $(grep -c '^SKIP' "$LOG")"
