#!/bin/bash
# Full benchmark comparison: baseline vs latent_mas vs slot_mas
# Usage: bash scripts/run_benchmark.sh <MODEL_PATH> [GPUS]
# Example: bash scripts/run_benchmark.sh Qwen/Qwen3-14B 0,1,2
set -e

MODEL="${1:?Usage: $0 <model_path> [gpu_ids]}"
GPUS="${2:-0,1,2}"
PROMPT="sequential"
MAX_TOKENS=2048
SEED=42
DATE=$(date +%Y%m%d_%H%M)
BASE_DIR="results/comparison_${DATE}"

mkdir -p "$BASE_DIR"

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

    echo "" | tee -a "$BASE_DIR/run.log"
    echo ">>> Task: $task (n=$max_samples)" | tee -a "$BASE_DIR/run.log"

    for method_spec in "${METHODS[@]}"; do
        IFS='|' read -r method extra_args _ label <<< "$method_spec"
        run_idx=$((run_idx + 1))

        outfile="$BASE_DIR/${task}_${label}"
        echo "  [$run_idx/$total_runs] $task / $label ..." | tee -a "$BASE_DIR/run.log"

        CUDA_VISIBLE_DEVICES=$GPUS stdbuf -oL python -u src/LatentMAS/run.py \
            --method "$method" \
            --model_name "$MODEL" \
            --task "$task" \
            --prompt "$PROMPT" \
            --max_samples "$max_samples" \
            --max_new_tokens $MAX_TOKENS \
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
