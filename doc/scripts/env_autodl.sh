#!/usr/bin/env bash
# AutoDL: 模型在数据盘，缓存与临时文件避免撑满系统盘 /
# Usage: source doc/scripts/env_autodl.sh

export MODEL_LOCAL="${MODEL_LOCAL:-/root/autodl-tmp/Qwen3-8B}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/cache/datasets}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"

# uv：缓存与项目虚拟环境都放在数据盘，与 wheel 缓存同盘可硬链接，避免跨 autodl-fs 全量拷贝导致极慢
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/autodl-tmp/venvs/embedding-multiagents}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

# 可选：完全离线（本地 from_pretrained 仍可用）
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
