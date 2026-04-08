#!/usr/bin/env bash
# 监控 reproduce_autodl.sh 后台任务
#
# 用法:
#   bash doc/scripts/monitor_repro.sh                    # 跟踪最新 results/repro_* 日志
#   bash doc/scripts/monitor_repro.sh results/repro_xxx  # 指定目录
#   bash doc/scripts/monitor_repro.sh --status results/repro_xxx
#   bash doc/scripts/monitor_repro.sh --gpu 1            # 每 1 秒刷新 nvidia-smi（另开终端）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="${REPO_ROOT}/results"

latest_repro_dir() {
  ls -td "$RESULTS"/repro_* 2>/dev/null | head -1 || true
}

if [[ "${1:-}" == "--gpu" ]]; then
  INTERVAL="${2:-2}"
  echo "每 ${INTERVAL}s 刷新 nvidia-smi（Ctrl+C 退出）"
  watch -n "$INTERVAL" nvidia-smi
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  RUN_DIR="${2:?用法: $0 --status <results/repro_...>}"
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "目录不存在: $RUN_DIR" >&2
    exit 1
  fi
  echo "RUN_DIR=$RUN_DIR"
  if [[ -f "$RUN_DIR/repro.pid" ]]; then
    PID=$(cat "$RUN_DIR/repro.pid")
    echo "repro.pid=$PID"
    if ps -p "$PID" > /dev/null 2>&1; then
      echo "状态: 仍在运行 (PID $PID)"
    else
      echo "状态: 进程已结束或未找到"
    fi
  else
    echo "无 repro.pid"
  fi
  echo "--- meta.json ---"
  cat "$RUN_DIR/meta.json" 2>/dev/null || echo "(无)"
  echo "--- 日志末尾 (repro.log) ---"
  tail -n 40 "$RUN_DIR/repro.log" 2>/dev/null || echo "(无 repro.log)"
  exit 0
fi

if [[ -n "${1:-}" ]] && [[ "$1" != --* ]]; then
  RUN_DIR="$1"
else
  RUN_DIR="$(latest_repro_dir)"
fi

if [[ -z "$RUN_DIR" ]] || [[ ! -d "$RUN_DIR" ]]; then
  echo "未找到 results/repro_* ，请先运行 reproduce_autodl.sh" >&2
  exit 1
fi

LOG="$RUN_DIR/repro.log"
if [[ ! -f "$LOG" ]]; then
  echo "无日志文件: $LOG" >&2
  exit 1
fi

echo "跟踪日志: $LOG"
echo "（另开终端可看 GPU: bash $SCRIPT_DIR/monitor_repro.sh --gpu 2）"
echo "（查看状态: bash $SCRIPT_DIR/monitor_repro.sh --status $RUN_DIR）"
echo "----------------------------------------"
tail -f "$LOG"
