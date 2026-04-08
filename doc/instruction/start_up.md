# 项目环境一键构建（AutoDL 数据盘）

本文说明如何通过 **`doc/scripts/start_up.sh`** 在 **与当前约定一致** 的布局下初始化环境：依赖、缓存与 **模型目录均在 `/root/autodl-tmp/`**（由 `doc/scripts/env_autodl.sh` 定义，可按机器改写）。

实验运行、监控与清理见 **[start_exp.md](./start_exp.md)**。

---

## 1. 你需要什么

- 已安装 **[uv](https://github.com/astral-sh/uv)**（用于 `uv sync` 与 `uv run`）。
- 能访问 **Hugging Face Hub**（下载模型；若在国内需自行配置镜像或代理）。
- 若模型为 **gated**，需在环境中设置 **`HF_TOKEN`**（与 Hugging Face CLI / hub 一致）。

数据盘路径默认值（可在运行前 `export` 覆盖）：

| 变量 | 默认 | 含义 |
|------|------|------|
| `MODEL_LOCAL` | `/root/autodl-tmp/Qwen3-8B` | 本地模型快照目录 |
| `HF_HOME` | `/root/autodl-tmp/cache/huggingface` | Transformers / Hub 缓存 |
| `HF_DATASETS_CACHE` | `/root/autodl-tmp/cache/datasets` | Datasets 缓存 |
| `UV_CACHE_DIR` | `/root/autodl-tmp/cache/uv` | uv 包缓存 |
| `UV_PROJECT_ENVIRONMENT` | `/root/autodl-tmp/venvs/embedding-multiagents` | 本项目虚拟环境 |
| `TMPDIR` | `/root/autodl-tmp/tmp` | 临时文件 |

以上由 **`source doc/scripts/env_autodl.sh`** 写入环境变量，并在脚本中 **`mkdir -p`** 所需父目录。

---

## 2. 一键命令

在**仓库根目录**执行：

```bash
cd /root/autodl-fs/Embedding-MultiAgents
bash doc/scripts/start_up.sh
```

脚本顺序为：

1. **`source doc/scripts/env_autodl.sh`**：导出上述变量并创建缓存目录。
2. **`uv sync`**：按 `pyproject.toml` / `uv.lock` 安装依赖，虚拟环境位于 `UV_PROJECT_ENVIRONMENT`（数据盘）。
3. **下载默认模型**：从 Hub 拉取 **`Qwen/Qwen3-8B`** 到 **`MODEL_LOCAL`**（`huggingface_hub.snapshot_download`），与手动下载到同一路径的效果一致。

若 **`$MODEL_LOCAL/config.json` 已存在**，则**跳过下载**（避免误重复拉取）。需要强制重新下载时：

```bash
FORCE=1 bash doc/scripts/start_up.sh
```

---

## 3. 常用环境变量（可选）

| 变量 | 说明 |
|------|------|
| `HF_MODEL_ID` | Hub 上的模型 ID，默认 `Qwen/Qwen3-8B` |
| `MODEL_LOCAL` | 本地目录，默认见 `env_autodl.sh` |
| `FORCE` | `1` 时删除已有模型目录并重新下载（慎用） |
| `SKIP_DOWNLOAD` | `1` 时只做目录 + `uv sync`，不下载模型 |
| `SKIP_UV` | `1` 时跳过 `uv sync`（仅当你已装好依赖） |
| `HF_TOKEN` | Gated 模型或提高限额时使用 |

示例：指定模型 ID 与本地路径：

```bash
export HF_MODEL_ID=Qwen/Qwen3-8B
export MODEL_LOCAL=/root/autodl-tmp/Qwen3-8B
bash doc/scripts/start_up.sh
```

---

## 4. 完成后自检（可选）

```bash
source doc/scripts/env_autodl.sh
bash doc/scripts/run_llm_smoke.sh
```

能打印 `hidden_size` 与 `OK` 即表示可从 **`MODEL_LOCAL`** 加载 Qwen。

快速跑一小段实验：

```bash
MODE=quick GPUS=0 bash doc/scripts/reproduce_autodl.sh
```

---

## 5. 与手动流程的对应关系

| 步骤 | `start_up.sh` 中的做法 |
|------|-------------------------|
| 数据盘目录 | 与 `env_autodl.sh` 相同，避免系统盘被缓存占满 |
| Python 依赖 | `uv sync`，环境在 `autodl-tmp/venvs/...` |
| 模型 | `snapshot_download` 到 `MODEL_LOCAL`，等价于下载到 `/root/autodl-tmp/Qwen3-8B` |

若你已在其它路径有完整模型，无需再下：把 **`MODEL_LOCAL`** 指到该目录（并保证含 `config.json` 等），`start_up.sh` 会跳过下载；或直接 **`SKIP_DOWNLOAD=1 bash doc/scripts/start_up.sh`** 只同步依赖。

---

## 6. 相关文件

| 路径 | 作用 |
|------|------|
| `doc/scripts/env_autodl.sh` | 数据盘路径与 `mkdir` |
| `doc/scripts/start_up.sh` | 本页描述的一键入口 |
| `doc/scripts/run_llm_smoke.sh` | 加载模型冒烟测试 |
| `doc/instruction/start_exp.md` | 开启 / 监控 / 删除实验 |
