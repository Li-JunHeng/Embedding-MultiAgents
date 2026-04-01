#!/bin/bash
# Train slot compressor (recommended pipeline)
# Usage: bash scripts/train_slot_compressor.sh <MODEL_PATH> [TASK] [GPUS]
# Example: bash scripts/train_slot_compressor.sh Qwen/Qwen3-14B gsm8k 0,1,2
set -e

MODEL="${1:?Usage: $0 <model_path> [task] [gpu_ids]}"
TASK="${2:-gsm8k}"
GPUS="${3:-0,1,2}"

echo "=== Training Slot Compressor ==="
echo "Model: $MODEL"
echo "Task: $TASK"
echo "GPUs: $GPUS"

CUDA_VISIBLE_DEVICES=$GPUS python -u src/LatentMAS/train_compressor.py \
    --model_name "$MODEL" \
    --task "$TASK" \
    --collect_samples 300 \
    --num_slots 4 \
    --slot_dim 64 \
    --latent_steps 10 \
    --epochs 100 \
    --think \
    --train_dims "64,32,16" \
    --cache_tag "slot_aligned_v2" \
    --output_dir "results/trained_compressor"

echo ""
echo "Done! Compressor weights saved to results/trained_compressor/"
echo ""
echo "Evaluate with:"
echo "  python src/LatentMAS/run.py --method slot_mas --model_name $MODEL \\"
echo "    --task $TASK --latent_steps 10 --num_slots 4 --slot_dim 64 --think \\"
echo "    --compressor_path results/trained_compressor/compressor_s4_d64.pt"
