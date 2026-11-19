# sp500-final · S&P 500 周频择时策略

> **课程**：数据科学开源工具 · Final Project
> **题目**：从数据到决策 —— S&P 500 的防泄漏机器学习择时策略（1996–2025）
> **团队**：MUST-SP500 六人小组

在严格避免前视偏差与数据泄漏、并事先注册主模型与主指标的前提下，回答一个问题：
S&P 500 过去 30 年的量价与宏观信息，对**周频**收益方向到底有多少可预测性？那些"打败了买入持有"的结果，
有多少来自泄漏、选择偏差、成本假设与运气？

## 快速开始（3 条命令）

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # 应全部通过
```

跑通后你会看到 `src/` 里哪些模块已实现、哪些还在等负责人实现：

```bash
python -m src.pipeline --check
```

## 目录结构

```text
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
python -m src.pipeline --config config.yaml
```

一条命令从原始数据跑到最终图表与指标。随机种子统一在 `config.yaml` 里管理，
不依赖本机绝对路径，原始数据有缓存可离线重跑。
