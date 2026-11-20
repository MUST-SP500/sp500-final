# sp500-final · S&P 500 周频择时策略

> **课程**：数据科学开源工具 · Final Project
> **题目**：从数据到决策 —— S&P 500 的防泄漏机器学习择时策略（1996–2025）
> **组别**：G05　**组员**：熊泽睿、许孙东、赵一玮、陈崇轩、郭飞扬、HHL
> **团队**：MUST-SP500 六人小组

在严格避免前视偏差与数据泄漏、并事先注册主模型与主指标的前提下，回答一个问题：
S&P 500 过去 30 年的量价与宏观信息，对**周频**收益方向到底有多少可预测性？那些"打败了买入持有"的结果，
有多少来自泄漏、选择偏差、成本假设与运气？

## 快速开始（3 条命令）

依赖以 `pyproject.toml` + `uv.lock` 为**唯一真源**，不需要手动创建虚拟环境，也不需要 `pip install`。

```bash
uv sync                            # 按 uv.lock 复现环境（自动建 .venv）
uv run pytest -q                   # 应全部通过
uv run python -m src.pipeline --check   # 管道自检
```

跑通后 `uv run python -m src.pipeline --check` 会打印 `src/` 里哪些模块已实现、哪些还在等负责人实现。

## 目录结构

```text
pyproject.toml   # 依赖与项目设置（唯一真源，配 uv.lock 复现环境）
uv.lock          # 锁定每个依赖的确切版本，必须提交
.python-version  # 锁定 Python 3.12
config.yaml      # 全局参数（种子、区间、切分、成本）
src/          # 所有正式代码（notebook 只用于探索，不进交付物）
tests/        # 防泄漏、数据质量、回测对账的自动化测试
docs/         # 预注册、决策日志、数据与特征字典
data/raw/     # 原始快照，只读，不进 git
data/processed/  # 周频面板、splits、预测
reports/      # 图表与最终报告
notebooks/    # 探索用，非交付物
```

## 分工

| 块 | 主责 | 备份 | 交付 |
|---|---|---|---|
| B1 数据与工程 | P1 | P3 | 数据面板 + 爬虫 + 复现 |
| B2 EDA 与可视化 | P2 | P6 | 10 张图 + 结论 |
| B3 特征与防泄漏 | P3 | P1 | 特征 + split + 泄漏测试 |
| B4 传统 ML 与评测 | P4 | P5 | 模型结果 + 决策规则 |
| B5 PCA 与深度学习 | P5 | P4 | DL 结果 + 消融 |
| B6 回测与集成 | P6 | P2 | tearsheet + 报告 |

> 详细人日、闸门与排期见仓库根目录的 `SP500_FinalProject_Plan_6P_v0.3.md`。

## 协作规则

见 [CONTRIBUTING.md](CONTRIBUTING.md)。核心三条：

1. **`main` 只接受 PR**，不能直接 push；
2. **一个任务先开 Issue**，PR 里写 `Closes #编号`；
3. **`pytest -q` 不过，不许合并**。

## 复现标准

```bash
uv sync
uv run python -m src.pipeline --config config.yaml
```

一条命令从原始数据跑到最终图表与指标。随机种子统一在 `config.yaml` 里管理，
不依赖本机绝对路径，原始数据有缓存可离线重跑。

## 预期结果（当前阶段）

| 命令 | 预期输出 | 耗时 |
|---|---|---|
| `uv run pytest -q` | `45 passed, 3 skipped`（3 个 skip 是等 B3/B6 实现的占位测试） | 约 10 秒 |
| `uv run python -m src.pipeline --check` | `[ok] 目录结构完整` + 8 个模块「可导入」 | 约 2 秒 |
| `uv run python -m src.pipeline --config config.yaml` | 目前会抛 `NotImplementedError("Bx：待实现")`，B1–B6 逐个落地后转为在 `data/processed/` 与 `reports/figures/` 生成产物 | — |

依赖若被改动，`uv sync` 会自动对齐环境；`uv.lock` 与 `pyproject.toml` 必须同时提交。
