#!/usr/bin/env bash
# Launch memory_mas on AIME 2025 (full set). run.py loads aime2025 with split=train only; --split is ignored.
# Default GPU: physical card 0 (latent_mas script uses card 1). Override: CUDA_VISIBLE_DEVICES=2 bash ...
#
# Usage:
#   bash doc/scripts/aime2025/run_memory_mas.sh
# Monitor:
#   bash doc/scripts/monitor_repro.sh results/repro_<STAMP>

set -euo pipefail

cd /root/autodl-fs/Embedding-MultiAgents
source doc/scripts/env_autodl.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="results/repro_${STAMP}"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/meta.json" <<EOF
{
  "stamp": "$STAMP",
  "mode": "memory_mas_aime2025_full",
  "model": "${MODEL_LOCAL:-/root/autodl-tmp/Qwen3-8B}",
  "gpus": "${CUDA_VISIBLE_DEVICES}",
  "python": "uv run python",
  "method": "memory_mas",
  "task": "aime2025",
  "split": "train",
  "split_note": "run.py uses load_aime2025(split=train); CLI --split does not apply",
  "max_samples": -1,
  "max_new_tokens": 8192,
  "latent_steps": 10,
  "memory_dim": 256,
  "memory_segment_length": 4,
  "memory_top_agents": 2,
  "memory_top_clusters": 4,
  "memory_top_segments": 4,
  "run_dir": "$(pwd)/$RUN_DIR",
  "note": "AIME 2025 hierarchical shared-memory memory_mas; monitor with doc/scripts/monitor_repro.sh"
}
EOF

cat > "$RUN_DIR/launch.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-fs/Embedding-MultiAgents/doc/scripts/env_autodl.sh
cd /root/autodl-fs/Embedding-MultiAgents
exec > >(tee -a /root/autodl-fs/Embedding-MultiAgents/$RUN_DIR/repro.log) 2>&1
echo \$\$ > /root/autodl-fs/Embedding-MultiAgents/$RUN_DIR/repro.pid
echo "=============================================="
echo "memory_mas AIME 2025 (train split, full set) — \$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL=${MODEL_LOCAL:-/root/autodl-tmp/Qwen3-8B} max_samples=-1 (all) latent_steps=10 memory_dim=256 segment_length=4 max_new_tokens=8192"
echo "RUN_DIR=/root/autodl-fs/Embedding-MultiAgents/$RUN_DIR"
echo "=============================================="
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES uv run python -u src/LatentMAS/run.py \\
  --method memory_mas \\
  --model_name "${MODEL_LOCAL:-/root/autodl-tmp/Qwen3-8B}" \\
  --task aime2025 \\
  --prompt sequential \\
  --split train \\
  --max_samples -1 \\
  --max_new_tokens 8192 \\
  --generate_bs 1 \\
  --seed 42 \\
  --latent_steps 10 \\
  --think \\
  --memory_dim 256 \\
  --memory_segment_length 4 \\
  --memory_top_agents 2 \\
  --memory_top_clusters 4 \\
  --memory_top_segments 4

echo ""
echo "ALL DONE — \$(date -Is)"
echo "主日志: /root/autodl-fs/Embedding-MultiAgents/$RUN_DIR/repro.log"
EOF

chmod +x "$RUN_DIR/launch.sh"
nohup "$RUN_DIR/launch.sh" >/tmp/memory_mas_aime2025_launch.log 2>&1 &

echo "RUN_DIR=$RUN_DIR"
echo "监控: bash doc/scripts/monitor_repro.sh $RUN_DIR"
echo "状态: bash doc/scripts/monitor_repro.sh --status $RUN_DIR"
echo "GPU:  bash doc/scripts/monitor_repro.sh --gpu 2"
