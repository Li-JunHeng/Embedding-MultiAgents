#!/usr/bin/env bash
# AutoDL：复现实验并写入 results/repro_<时间戳>/（可用 nohup 后台跑）
#
# 用法:
#   source doc/scripts/env_autodl.sh   # 可选；会设置 MODEL_LOCAL、HF 缓存等到数据盘
#   MODE=quick GPUS=0 bash doc/scripts/reproduce_autodl.sh
#   MODE=all GPUS=0 nohup bash doc/scripts/reproduce_autodl.sh &
#
# 环境变量:
#   MODE     quick | bench | hidden | all   (默认 all；all = 完整 LatentMAS 基准 + Hidden Profile，不含 quick)
#   MODEL    覆盖本地模型路径 (默认 ${MODEL_LOCAL}，见 env_autodl.sh)
#   GPUS     CUDA_VISIBLE_DEVICES (默认 0)
#   PYTHON   解释器 (默认 uv run python)
#   RUN_ROOT 结果根目录 (默认 仓库下 results)
#   QUICK_MAX_SAMPLES  quick 模式下 gsm8k 条数 (默认 5，与其它方法一致便于对比)
#   MEMORY_DIM         memory_mas 检索压缩维 (默认 256，与 run.py 默认一致)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=env_autodl.sh
if [[ -f "$SCRIPT_DIR/env_autodl.sh" ]]; then
  # shellcheck source=env_autodl.sh
  source "$SCRIPT_DIR/env_autodl.sh"
fi

MODE="${MODE:-all}"
MODEL="${MODEL:-${MODEL_LOCAL:-/root/autodl-tmp/Qwen3-8B}}"
GPUS="${GPUS:-0}"
PYTHON="${PYTHON:-uv run python}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/results}"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUN_ROOT}/repro_${STAMP}"
mkdir -p "$RUN_DIR"

exec > >(tee -a "$RUN_DIR/repro.log") 2>&1
echo "$$" > "$RUN_DIR/repro.pid"

cat > "$RUN_DIR/meta.json" << EOF
{
  "stamp": "$STAMP",
  "mode": "$MODE",
  "model": "$MODEL",
  "gpus": "$GPUS",
  "python": "$PYTHON",
  "run_dir": "$RUN_DIR",
  "repo": "$REPO_ROOT"
}
EOF

echo "=============================================="
echo "Reproduce — $(date -Is)"
echo "MODE=$MODE MODEL=$MODEL GPUS=$GPUS"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$$ (written to repro.pid)"
echo "=============================================="

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHON
cd "$REPO_ROOT"

run_latentmas_quick() {
  echo ""
  QUICK_MAX_SAMPLES="${QUICK_MAX_SAMPLES:-5}"
  MEMORY_DIM="${MEMORY_DIM:-256}"
  echo "=== LatentMAS 快速试跑 (gsm8k, n=${QUICK_MAX_SAMPLES}, baseline + latent_mas + slot_mas + memory_mas) ==="
  local out="$RUN_DIR/latentmas_quick"
  mkdir -p "$out"
  PROMPT="sequential"
  MAX_TOKENS=2048
  SEED=42
  declare -a BENCHMARKS=("gsm8k ${QUICK_MAX_SAMPLES}")
  declare -a METHODS=(
    "baseline|||baseline"
    "latent_mas|--latent_steps 10 --think||latent_mas"
    "slot_mas|--latent_steps 10 --num_slots 4 --slot_dim 64 --think||slot_mas_s4d64_random"
    "memory_mas|--latent_steps 10 --think --memory_dim ${MEMORY_DIM}||memory_mas_m${MEMORY_DIM}"
  )
  total_runs=$(( ${#BENCHMARKS[@]} * ${#METHODS[@]} ))
  run_idx=0
  for bench_spec in "${BENCHMARKS[@]}"; do
    task=$(echo "$bench_spec" | awk '{print $1}')
    max_samples=$(echo "$bench_spec" | awk '{print $2}')
    echo ">>> Task: $task (n=$max_samples)"
    for method_spec in "${METHODS[@]}"; do
      IFS='|' read -r method extra_args _ label <<< "$method_spec"
      run_idx=$((run_idx + 1))
      outfile="$out/${task}_${label}"
      echo "  [$run_idx/$total_runs] $task / $label ..."
      CUDA_VISIBLE_DEVICES=$GPUS stdbuf -oL $PYTHON -u src/LatentMAS/run.py \
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
      echo "    -> $(cat "${outfile}.json")"
    done
  done
  echo "Quick LatentMAS done. Logs under $out"
}

run_latentmas_full() {
  echo ""
  echo "=== LatentMAS 完整对比 (多数据集，见 run_benchmark.sh) ==="
  bash "$SCRIPT_DIR/run_benchmark.sh" "$MODEL" "$GPUS" "$RUN_DIR/latentmas_benchmark"
}

run_hidden_profile() {
  echo ""
  echo "=== Hidden Profile (训练 + baseline，耗时较长) ==="
  local hp_out="$RUN_DIR/hidden_profile"
  mkdir -p "$hp_out"
  (
    cd "$REPO_ROOT/src/latent_communication_llm"
    $PYTHON run_hidden_profile.py \
      --model-path "$MODEL" \
      --output-dir "$hp_out"
  )
  echo "Hidden Profile done: $hp_out/hidden_profile_results.json"
}

case "$MODE" in
  quick)
    run_latentmas_quick
    ;;
  bench)
    run_latentmas_full
    ;;
  hidden)
    run_hidden_profile
    ;;
  all)
    run_latentmas_full
    run_hidden_profile
    ;;
  *)
    echo "Unknown MODE=$MODE (use quick|bench|hidden|all)" >&2
    exit 1
    ;;
esac

echo ""
echo "ALL DONE — $(date -Is)"
echo "Artifacts: $RUN_DIR"
echo "主日志: $RUN_DIR/repro.log"
echo "监控: bash doc/scripts/monitor_repro.sh $RUN_DIR"
