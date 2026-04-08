#!/usr/bin/env bash
# 快速验证 latent_communication_llm 能从 autodl-tmp 加载 Qwen（与计划中的 --model-path 一致）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=env_autodl.sh
source "$SCRIPT_DIR/env_autodl.sh"
export PATH="/root/miniconda3/bin:${PATH}"
cd "$REPO_ROOT/src/latent_communication_llm"
python -c "
from run_qwen_handoff import load_qwen_model
_, m = load_qwen_model('${MODEL_LOCAL}')
print('hidden_size', getattr(m.config, 'hidden_size', None))
print('OK: latent_communication_llm + MODEL_LOCAL')
"
