# AutoDL 实验：启动、监控与清理

环境准备（uv、数据盘路径、下载 Qwen 等）见 **[start_up.md](./start_up.md)**。

本文说明在本仓库中如何**开启复现实验**、**监控进度**、**停止任务并删除本轮实验数据**。默认假设 Qwen 模型与缓存位于数据盘（例如 `/root/autodl-tmp/`），与 `doc/scripts/env_autodl.sh` 一致。

---

## 1. 前置条件

- 工作目录为仓库根目录，例如：

  ```bash
  cd /root/autodl-fs/Embedding-MultiAgents
  ```

- （推荐）加载数据盘环境变量，避免 Hugging Face / uv 缓存撑满系统盘：

  ```bash
  source doc/scripts/env_autodl.sh
  ```

  其中 `MODEL_LOCAL` 默认为 `/root/autodl-tmp/Qwen3-8B`；若你的模型路径不同，可：

  ```bash
  export MODEL_LOCAL=/root/autodl-tmp/Qwen3-8B
  ```

- 依赖已通过 `uv` 安装（见仓库 `pyproject.toml`），复现脚本使用 **`uv run python`** 调用 Python。

---

## 2. 开启实验

主入口脚本：**`doc/scripts/reproduce_autodl.sh`**

每轮运行会在 **`results/repro_<时间戳>/`** 下生成：

| 文件/目录 | 说明 |
|-----------|------|
| `repro.log` | 主日志 |
| `repro.pid` | 主 bash 进程号（用于停止） |
| `meta.json` | 本次 MODE、模型路径、GPU 等元数据 |
| `latentmas_quick/` | MODE=`quick` 时的快速试跑输出 |
| `latentmas_benchmark/` | MODE=`bench` 或 `all` 时的完整 LatentMAS 对比 |
| `hidden_profile/` | MODE=`hidden` 或 `all` 时的 Hidden Profile 结果 |

### 2.1 环境变量（MODE）

| MODE | 含义 |
|------|------|
| `quick` | GSM8K 每方法 5 条样本，3 种方法；用于快速验证环境 |
| `bench` | 完整 `run_benchmark.sh`（多数据集 × 3 方法） |
| `hidden` | Hidden Profile（含特征抽取与训练，耗时较长） |
| `all`（默认） | **bench + hidden**（不含 quick） |

其他常用变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL` | `MODEL_LOCAL` 或 `/root/autodl-tmp/Qwen3-8B` | 本地模型目录 |
| `GPUS` | `0` | `CUDA_VISIBLE_DEVICES` |
| `PYTHON` | `uv run python` | Python 启动方式 |
| `RUN_ROOT` | 仓库下 `results` | 结果根目录 |

### 2.2 前台运行

```bash
source doc/scripts/env_autodl.sh   # 可选
MODE=all GPUS=0 bash doc/scripts/reproduce_autodl.sh
```

### 2.3 后台运行（推荐长时间任务）

```bash
cd /root/autodl-fs/Embedding-MultiAgents
source doc/scripts/env_autodl.sh
nohup env MODE=all GPUS=0 bash doc/scripts/reproduce_autodl.sh &
```

记下终端打印的 **shell PID**，或事后在 **`results/repro_<时间戳>/repro.pid`** 中查看主进程号。

---

## 3. 监控实验

脚本：**`doc/scripts/monitor_repro.sh`**

### 3.1 跟踪最新一轮的主日志

```bash
bash doc/scripts/monitor_repro.sh
```

等价于对 **`results/` 下最新的 `repro_*` 目录**中的 `repro.log` 执行 `tail -f`。

### 3.2 指定某一轮目录

```bash
bash doc/scripts/monitor_repro.sh results/repro_20260407_153000
```

### 3.3 查看状态（不持续跟随）

```bash
bash doc/scripts/monitor_repro.sh --status results/repro_20260407_153000
```

会显示：`repro.pid`、进程是否仍在、`meta.json`、日志末尾若干行。

### 3.4 观察 GPU（另开终端）

```bash
bash doc/scripts/monitor_repro.sh --gpu 2
```

每 2 秒刷新一次 `nvidia-smi`（数字可改）。

---

## 4. 停止实验并删除本轮数据

### 4.1 仅停止进程（保留 `results/repro_*` 目录）

脚本：**`doc/scripts/stop_repro.sh`**

按 **`repro.pid`** 递归结束子进程（如 `python`、子 shell），再结束主 bash：

```bash
bash doc/scripts/stop_repro.sh results/repro_20260407_153000
```

若仍有占用 GPU 的 Python，脚本会提示；可配合：

```bash
nvidia-smi
pgrep -af 'LatentMAS/run.py|run_hidden_profile'
kill <PID>
```

### 4.2 停止并删除本轮全部文件

```bash
bash doc/scripts/stop_repro.sh results/repro_20260407_153000 --rm
```

### 4.3 仅删除目录（已确认无进程占用时）

```bash
rm -rf results/repro_20260407_153000
```

---

## 5. 相关脚本一览

| 脚本 | 作用 |
|------|------|
| `doc/scripts/env_autodl.sh` | 设置 `MODEL_LOCAL`、`HF_HOME`、`UV_PROJECT_ENVIRONMENT` 等 |
| `doc/scripts/start_up.sh` | 一键建环境 + 下载模型（见 [start_up.md](./start_up.md)） |
| `doc/scripts/reproduce_autodl.sh` | 一键复现（MODE 控制范围） |
| `doc/scripts/monitor_repro.sh` | 跟踪日志 / 状态 / GPU |
| `doc/scripts/stop_repro.sh` | 停止进程树，可选 `--rm` 删除本轮目录 |
| `doc/scripts/run_benchmark.sh` | 单独跑完整 LatentMAS 对比（可被 `reproduce_autodl.sh` 调用） |

---

## 6. 注意事项

- **`MODE=all`** 耗时会很长（多数据集推理 + Hidden Profile 训练）。可先用 **`MODE=quick`** 验证再跑全量。
- 停止实验时优先使用 **`stop_repro.sh`**，避免残留子进程占用显存。
- 删除 **`results/repro_*`** 即清除该轮实验产出；**不会**删除 `/root/autodl-tmp/` 中的模型与 Hugging Face 缓存。
