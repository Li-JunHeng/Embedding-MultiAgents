#!/usr/bin/env bash
# 在数据盘 /root/autodl-tmp 安装 nvm + Node LTS（Gitee + npmmirror），并可选写入 PATH 提示。
set -euo pipefail

NVM_DIR="${NVM_DIR:-/root/autodl-tmp/nvm}"
export NVM_NODEJS_ORG_MIRROR="${NVM_NODEJS_ORG_MIRROR:-https://npmmirror.com/mirrors/node}"
export NVM_IOJS_ORG_MIRROR="${NVM_IOJS_ORG_MIRROR:-https://npmmirror.com/mirrors/iojs}"

mkdir -p "$(dirname "$NVM_DIR")"

if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
  echo "==> 克隆 nvm 到 $NVM_DIR (Gitee 镜像)"
  rm -rf "$NVM_DIR"
  git clone --depth 1 https://gitee.com/mirrors/nvm.git "$NVM_DIR"
fi

# shellcheck source=/dev/null
. "$NVM_DIR/nvm.sh"

echo "==> 安装 Node LTS（二进制来自 npmmirror）"
nvm install --lts
nvm alias default lts/*

echo "==> 当前版本: $(command -v node) -> $(node -v), npm $(npm -v)"
echo "若新开终端仍无 node，请执行: source ~/.profile"
echo "或运行一次（需已安装 node 后）把 node 链到系统路径:"
echo "  sudo ln -sf \"\$(command -v node)\" /usr/local/bin/node && sudo ln -sf \"\$(command -v npm)\" /usr/local/bin/npm"
