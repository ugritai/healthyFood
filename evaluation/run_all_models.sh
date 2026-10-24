#!/bin/bash
# Run Part A evaluation for ALL models on BOTH batches.
#
# Quick test (1 recipe per batch):
#   bash run_all_models.sh --test
#
# Full run (nohup):
#   nohup bash run_all_models.sh > nohup_all.log 2>&1 &
#
# Write somewhere other than results/ (e.g. to compare with a previous run):
#   RESULTS_ROOT=results/rerun bash run_all_models.sh

set -euo pipefail

# Laptop venv if present, otherwise the server's
if [[ -x ../.venv/bin/python ]]; then PYTHON="../.venv/bin/python"; else PYTHON="../healthyfood_env/bin/python"; fi
SOFT_THRESHOLD="0.75"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

# The five models reported in the paper (Sec. 5)
MODELS=(
    "llama3.1:8b-instruct-q8_0"
    "qwen3:8b-q8_0"
    "gemma2:9b-instruct-q8_0"
    "mistral:7b-instruct-q8_0"
    "dolphin-llama3:8b-v2.9-q8_0"
    # Also tried, not reported:
    # "deepseek-r1:8b-q8_0"
    # "olmo2:7b-instruct-q8_0"
    # "cogito:8b-v0.1-llama3"
)

BATCHES=(
    "benchmark.json|$RESULTS_ROOT/batch1|15 recipes"
    "benchmark_batch2.json|$RESULTS_ROOT/batch2|35 recipes"
)

MAX_RECIPES_ARG=""
if [[ "${1:-}" == "--test" ]]; then
    MAX_RECIPES_ARG="--max-recipes 1"
    echo "  [TEST MODE] Running 1 recipe per model per batch."
fi

mkdir -p "$RESULTS_ROOT/batch1" "$RESULTS_ROOT/batch2"

echo "============================================================"
echo "  FoodMedKG - Full LLM Evaluation (both batches)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Soft threshold : $SOFT_THRESHOLD"
echo "  Models         : ${#MODELS[@]}"
echo "  Batches        : ${#BATCHES[@]}"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Model : $MODEL"
    echo "============================================================"

    if ! ollama list | grep -q "^${MODEL} "; then
        echo "  [PULL] Downloading $MODEL ..."
        if ! ollama pull "$MODEL"; then
            echo "  [WARN] Pull failed for $MODEL - skipping."
            continue
        fi
    else
        echo "  [OK]   Already installed."
    fi

    for BATCH_DEF in "${BATCHES[@]}"; do
        BENCHMARK="${BATCH_DEF%%|*}"
        REST="${BATCH_DEF#*|}"
        RESULTS_DIR="${REST%%|*}"
        LABEL="${REST#*|}"

        echo ""
        echo "  ── Batch: $BENCHMARK  ($LABEL)  →  $RESULTS_DIR"

        # shellcheck disable=SC2086
        if $PYTHON evaluate.py \
                --benchmark "$BENCHMARK" \
                --models "$MODEL" \
                --soft-threshold "$SOFT_THRESHOLD" \
                --save-csv "$RESULTS_DIR/" \
                --runs 1 \
                $MAX_RECIPES_ARG; then
            echo "  [DONE] $MODEL / $BENCHMARK - $(date '+%H:%M:%S')"
        else
            echo "  [ERR ] $MODEL / $BENCHMARK failed - continuing."
        fi
    done
done

echo ""
echo "============================================================"
echo "  All models & batches processed - $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
