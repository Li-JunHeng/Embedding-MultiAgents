#!/usr/bin/env bash
# 一键：数据盘目录 + uv 依赖 + 拉取默认 Qwen 模型到 MODEL_LOCAL（与 env_autodl.sh 一致）
#
# 用法（在仓库根目录）:
#   bash doc/scripts/start_up.sh
#   HF_MODEL_ID=Qwen/Qwen3-8B FORCE=0 bash doc/scripts/start_up.sh
#
# 环境变量:
#   HF_MODEL_ID  Hugging Face 模型 ID（默认 Qwen/Qwen3-8B）
#   MODEL_LOCAL  本地目录（默认见 env_autodl.sh，一般为 /root/autodl-tmp/Qwen3-8B）
#   FORCE        设为 1 时即使已有 config.json 也重新下载（慎用）
#   SKIP_DOWNLOAD 设为 1 时跳过下载，仅 uv sync
#   SKIP_UV       设为 1 时跳过 uv sync
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=env_autodl.sh
source "$SCRIPT_DIR/env_autodl.sh"

HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-8B}"
FORCE="${FORCE:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_UV="${SKIP_UV:-0}"

echo "=== [1/3] 数据盘目录（MODEL_LOCAL / HF_HOME / UV 等）==="
echo "MODEL_LOCAL=$MODEL_LOCAL"
echo "HF_HOME=$HF_HOME"

if [[ "$SKIP_UV" != "1" ]]; then
  echo ""
  echo "=== [2/3] uv sync（安装/同步项目依赖与虚拟环境）==="
  cd "$REPO_ROOT"
  command -v uv >/dev/null 2>&1 || { echo "请先安装 uv: https://github.com/astral-sh/uv" >&2; exit 1; }
  uv sync
else
  echo "跳过 uv sync (SKIP_UV=1)"
fi

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  echo ""
  echo "跳过模型下载 (SKIP_DOWNLOAD=1)"
  echo "完成。"
  exit 0
fi

if [[ -f "$MODEL_LOCAL/config.json" && "$FORCE" != "1" ]]; then
  echo ""
  echo "已存在 $MODEL_LOCAL/config.json，跳过下载。"
  echo "若要强制重新拉取: FORCE=1 bash doc/scripts/start_up.sh"
  echo ""
  echo "环境就绪。下一步可:"
  echo "  source doc/scripts/env_autodl.sh"
  echo "  bash doc/scripts/run_llm_smoke.sh"
  echo "  MODE=quick GPUS=0 bash doc/scripts/reproduce_autodl.sh"
  exit 0
fi

echo ""
echo "=== [3/3] 下载模型 $HF_MODEL_ID -> $MODEL_LOCAL ===="
mkdir -p "$(dirname "$MODEL_LOCAL")"

cd "$REPO_ROOT"
export HF_MODEL_ID
export MODEL_LOCAL
export FORCE
uv run python - << 'PY'
import os
import shutil

from huggingface_hub import snapshot_download

repo = os.environ["HF_MODEL_ID"]
target = os.environ["MODEL_LOCAL"]

# 强制重下时清空目标目录（避免残留混用）
if os.environ.get("FORCE") == "1" and os.path.isdir(target):
    shutil.rmtree(target)

os.makedirs(target, exist_ok=True)
snapshot_download(
    repo_id=repo,
    local_dir=target,
)
print("OK:", target)
PY

echo ""
echo "环境就绪。下一步可:"
echo "  source doc/scripts/env_autodl.sh"
echo "  bash doc/scripts/run_llm_smoke.sh    # 可选：验证加载"
echo "  MODE=quick GPUS=0 bash doc/scripts/reproduce_autodl.sh"
