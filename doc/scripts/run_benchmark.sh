#!/bin/bash
# Full benchmark comparison: baseline vs latent_mas vs slot_mas vs memory_mas
# Usage: bash doc/scripts/run_benchmark.sh <MODEL_PATH> [GPUS] [OUTPUT_DIR]
# Optional env: MEMORY_DIM (default 256) for memory_mas --memory_dim
# Example: bash doc/scripts/run_benchmark.sh Qwen/Qwen3-8B 0,1,2
# Example: bash doc/scripts/run_benchmark.sh /root/autodl-tmp/Qwen3-8B 0 results/repro_xxx/latentmas
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:?Usage: $0 <model_path> [gpu_ids] [output_dir]}"
GPUS="${2:-0,1,2}"
# 例如: export PYTHON="uv run python"
PYTHON="${PYTHON:-python}"
PROMPT="sequential"
DEFAULT_MAX_TOKENS="${MAX_TOKENS:-1024}"
SHORT_MAX_TOKENS="${SHORT_MAX_TOKENS:-512}"
SEED=42
DATE=$(date +%Y%m%d_%H%M)
if [[ -n "${3:-}" ]]; then
    BASE_DIR="$3"
else
    BASE_DIR="results/comparison_${DATE}"
fi

mkdir -p "$BASE_DIR"

MEMORY_DIM="${MEMORY_DIM:-256}"

declare -a BENCHMARKS=(
    "gsm8k 100"
    "arc_challenge 100"
    "gpqa 198"
    "medqa 100"
    "arc_easy 100"
)

declare -a METHODS=(
    "baseline|||baseline"
    "latent_mas|--latent_steps 10 --think||latent_mas"
    "slot_mas|--latent_steps 10 --num_slots 4 --slot_dim 64 --think||slot_mas_s4d64_random"
    "memory_mas|--latent_steps 10 --think --memory_dim ${MEMORY_DIM}||memory_mas_m${MEMORY_DIM}"
)

echo "============================================" | tee "$BASE_DIR/run.log"
echo "Benchmark Comparison — $(date)" | tee -a "$BASE_DIR/run.log"
echo "Model: $MODEL" | tee -a "$BASE_DIR/run.log"
echo "============================================" | tee -a "$BASE_DIR/run.log"

total_runs=$(( ${#BENCHMARKS[@]} * ${#METHODS[@]} ))
run_idx=0

for bench_spec in "${BENCHMARKS[@]}"; do
    task=$(echo "$bench_spec" | awk '{print $1}')
    max_samples=$(echo "$bench_spec" | awk '{print $2}')
    max_tokens="$DEFAULT_MAX_TOKENS"
    case "$task" in
        gsm8k|arc_easy|arc_challenge|medqa)
            max_tokens="$SHORT_MAX_TOKENS"
            ;;
    esac

    echo "" | tee -a "$BASE_DIR/run.log"
    echo ">>> Task: $task (n=$max_samples, max_new_tokens=$max_tokens)" | tee -a "$BASE_DIR/run.log"

    for method_spec in "${METHODS[@]}"; do
        IFS='|' read -r method extra_args _ label <<< "$method_spec"
        run_idx=$((run_idx + 1))

        outfile="$BASE_DIR/${task}_${label}"
        echo "  [$run_idx/$total_runs] $task / $label ..." | tee -a "$BASE_DIR/run.log"

        CUDA_VISIBLE_DEVICES=$GPUS stdbuf -oL $PYTHON -u src/LatentMAS/run.py \
            --method "$method" \
            --model_name "$MODEL" \
            --task "$task" \
            --prompt "$PROMPT" \
            --max_samples "$max_samples" \
            --max_new_tokens "$max_tokens" \
            --generate_bs 1 \
            --seed $SEED \
            $extra_args \
            2>&1 | tee "${outfile}.log"

        tail -1 "${outfile}.log" > "${outfile}.json"
        echo "    -> $(cat ${outfile}.json)" | tee -a "$BASE_DIR/run.log"
    done
done

echo "" | tee -a "$BASE_DIR/run.log"
echo "ALL DONE!" | tee -a "$BASE_DIR/run.log"
