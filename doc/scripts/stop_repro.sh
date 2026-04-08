#!/usr/bin/env bash
# 停止某次 reproduce_autodl.sh 启动的进程树（读 repro.pid），再按需删除目录
#
# 用法:
#   bash doc/scripts/stop_repro.sh results/repro_YYYYMMDD_HHMMSS
#   bash doc/scripts/stop_repro.sh results/repro_xxx --rm    # 停进程并删除该轮目录
#
set -euo pipefail

RUN_DIR="${1:?用法: $0 <results/repro_...> [--rm]}"
DO_RM=false
if [[ "${2:-}" == "--rm" ]]; then
  DO_RM=true
fi

PID_FILE="$RUN_DIR/repro.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "未找到 $PID_FILE" >&2
  exit 1
fi
PID=$(cat "$PID_FILE")

kill_descendants() {
  local p=$1
  local c
  for c in $(pgrep -P "$p" 2>/dev/null || true); do
    kill_descendants "$c"
  done
  kill "$p" 2>/dev/null || true
}

echo "先结束子进程再结束主进程 (PID=$PID) ..."
# 先叶子后根：递归杀子进程
kill_descendants "$PID" || true
sleep 1
if ps -p "$PID" > /dev/null 2>&1; then
  echo "主进程仍在，发送 SIGKILL ..."
  kill -9 "$PID" 2>/dev/null || true
fi

# 可能仍有同会话下的 python（例如已从 shell 脱离）；仅提示
if pgrep -af "LatentMAS/run.py|run_hidden_profile|run_benchmark" >/dev/null 2>&1; then
  echo "若仍有相关 python 占用 GPU，请手动: nvidia-smi 查 PID 后 kill，或:"
  echo "  pgrep -af 'LatentMAS/run.py|run_hidden_profile'"
fi

if $DO_RM; then
  echo "删除目录: $RUN_DIR"
  rm -rf "$RUN_DIR"
  echo "已清除。"
else
  echo "已停止。若需删除本轮数据: rm -rf \"$RUN_DIR\""
  echo "或: bash $0 \"$RUN_DIR\" --rm"
fi
