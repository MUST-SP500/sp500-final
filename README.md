# sp500-final · S&P 500 周频择时

> **课程**：数据科学开源工具 · Final Project
> **题目**：从数据到决策 —— S&P 500 的防泄漏机器学习择时策略（1996–2025）
> **团队**：MUST-SP500 六人小组

在严格避免前视偏差与数据泄漏、并事先注册主模型与主指标的前提下，回答一个问题：
S&P 500 过去 30 年的量价与宏观信息，对**周频**收益方向到底有多少可预测性？那些"打败了买入持有"的结果，
有多少来自泄漏、选择偏差、成本假设与运气？

> 本项目**只用 uv**，不用 pip，也不需要手动创建虚拟环境。
> `uv sync` 会按 `.python-version` 自动装好 Python 和全部依赖。

---

## 一、第一次运行（三条命令）

**1. 装 uv**（一台电脑只做一次）

Windows（PowerShell 里粘贴执行）：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

装完**关掉终端重新打开**，输入 `uv --version` 能看到版本号就算成功。

**2. 装依赖**

```bash
uv sync
```

它会读 `pyproject.toml` 装依赖、按 `.python-version` 准备 Python（本机没有会自动下），
并在项目里生成 `.venv/`。默认不装 PyTorch，所以很快；做深度学习的那位同学多跑一条：

```bash
uv sync --extra dl     # 只有 P5 做 PCA/DNN/CNN/Autoencoder 时需要
```

> `uv sync` 会自动生成 `uv.lock`（依赖的精确版本表）。**生成后请提交进 git**，
> 它保证六个人、以及老师，装出来的是同一套版本。

**3. 跑**

```bash
uv run main.py --check
```

预期输出：项目版本、配置文件路径、区间 1996 → 2025、成本 5 bps，以及 8 个阶段各自的模块能不能导入。

> **当前是骨架状态**：15 个模块里，`config.py`、`validate.py`、`plot_style.py` 和 `backtest.run_positions` 已经能用，
> 其余函数还是 `raise NotImplementedError("Bx：待实现")`。所以现在跑全流程会在第一段就停下并告诉你找谁——这是正常的，
> 每块负责人把自己那几个函数填掉，管道就一段一段通了。

---

## 二、常用命令

| 想做什么 | 命令 |
|---|---|
| 自检环境（不跑数据） | `uv run main.py --check` |
| 全流程 | `uv run main.py` |
| 只跑某一段 | `uv run main.py --stage clean` |
| 跑测试 | `uv run pytest` |
| 数据质量报告 + 异常清单 | `uv run python -m src.data_quality` |
| 抓成分股快照 | `uv run python -m src.scrape_wiki` |
| 导出面板 CSV（用 Excel 核对） | `uv run python -m src.export_csv` |
| 一键复现 + 写复现日志 | `uv run python -m src.reproduce` |
| 删数据从零重跑并比对 | `uv run python -m src.reproduce --full-rebuild` |
| 开 notebook | `uv run jupyter lab` |
| 装深度学习依赖（P5） | `uv sync --extra dl` |
| 加一个依赖 | `uv add 包名`（会自动更新 pyproject 和 uv.lock） |
| 改 Python 版本 | `uv python pin 3.12`（同时更新 `.python-version`） |

阶段名：`download` `clean` `features` `labels` `splits` `models` `backtest` `report`。

**B1 数据层只跑前两段**（其余段依赖 B2–B6）：

```bash
uv run python -m src.pipeline --stage download   # 约 3 秒，5 个标的
uv run python -m src.pipeline --stage clean      # 约 0.1 秒，复权 + 周频 + 面板
```

预期结果：周面板 1,566 行 × 12 列（1996-01-05 ~ 2025-12-31，每年 52–53 周），日面板 7,550 行 × 12 列，零缺失。对不上就说明代码或数据版本不一致。完整记录见 `docs/reproduce_log.md`。

**不要做的事**：不要 `pip install`，不要手动 `python -m venv`，不要把 `.venv/` 打包或提交。
混用 pip 和 uv 会让 `uv.lock` 和实际环境对不上，老师那边复现就会失败。

---

## 三、跑完你会得到什么

| 位置 | 内容 |
|---|---|
| `data/raw/` | 原始数据快照 + `manifest.json`（来源、下载时间、行数、校验和） |
| `data/processed/` | `weekly_panel.parquet`（一行一周）、`splits.json`、`predictions.parquet` |
| `reports/figures/` | 所有图（EDA 7 张 + 回测 tearsheet） |
| `reports/` | `metrics.csv`（所有数字的唯一来源）、`data_quality.md`、`eda_findings.md` |
| `output/` | 提交用：上面这些图与指标的副本（打包时放这里） |

> 数据超过 50 MB 就不进压缩包，改成 README 写下载链接 + 附一个 `sample_data.csv`，
> 让别人离线也能把流程跑通。

---

## 四、目录结构

```text
main.py            # 入口，uv run main.py
pyproject.toml     # 依赖声明（唯一来源）
uv.lock            # 依赖锁定，要提交
.python-version    # Python 版本，要提交
config.yaml        # 全局参数：区间、标的、成本、切分
src/               # 全部正式代码（notebook 只用来探索，不算交付）
  data_download.py data_clean.py scrape_wiki.py    # P1 数据
  features.py labels.py splits.py                  # P3 特征与切分
  models_ml.py decision.py evaluate.py             # P4 模型与决策
  models_pca.py models_dl.py                       # P5 降维与深度学习
  backtest.py pipeline.py plot_style.py            # P6 回测与集成
  config.py validate.py                            # 公共：配置、数据校验
tests/             # 防泄漏、数据质量、回测对账的自动化测试
docs/              # 预注册、决策日志、数据字典、特征字典、超参、复现日志
  preregistration.md  decisions.md  data_dictionary.md  feature_dictionary.md
  hyperparams.md      reproduce_log.md
reports/           # 质量报告、EDA 结论、泄漏审计、指标表、报告工作稿
  data_quality.md  eda_findings.md  leakage_audit.md  final_report.md
  figures/         # 7 张 EDA 图 + 回测 tearsheet
notebooks/         # 探索用（01_eda.ipynb），不算交付
slides/            # 答辩 PPT（outline.md 是 16 页大纲）
output/            # 提交用：打包前把 reports 的结果复制到这里
data/raw/          # 原始快照：yahoo/ fred/ wiki/，只读，不进 git
data/processed/    # weekly_panel.parquet、splits.json、predictions.parquet
```

## 文档地图

| 想看什么 | 打开 |
|---|---|
| **每人做什么（函数级、带数据示例）** | [`SP500_分工说明书_函数与数据示例.md`](SP500_分工说明书_函数与数据示例.md) |
| 环境怎么装、怎么跑 | 本文件第一节 |
| 协作规则（分支、PR、DoD） | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 口径为什么这么定 | `docs/preregistration.md` |
| 字段、特征公式 | `docs/data_dictionary.md`、`docs/feature_dictionary.md` |
| 答辩讲什么 | `slides/outline.md` |
| 报告写到哪了 | `reports/final_report.md` |

---

## 五、分工

| 块 | 主责 | 交付 |
|---|---|---|
| B1 数据与工程 | P1 | 周频面板 + 爬虫 + 复现 |
| B2 EDA 与报告 | P2 | 7 张图 + 结论 + 报告定稿 |
| B3 特征与防泄漏 | P3 | 22 特征 + 标签 + 18 折切分 |
| B4 模型与决策 | P4 | 预测 + 三种下注规则 + 指标表 |
| B5 降维与深度学习 | P5 | PCA / DNN / CNN / Autoencoder |
| B6 回测与交付 | P6 | 净值 + tearsheet + PPT + 打包 |

> **每个人具体做什么（写到函数和数据表）**：见 [`SP500_分工说明书_函数与数据示例.md`](SP500_分工说明书_函数与数据示例.md)，先看这份。

---

## 六、提交前自测（必做）

不要在自己项目目录里测，那测不出问题。

```text
1. 新建一个临时目录（比如桌面的 test-unzip），把压缩包解压进去
2. cd test-unzip/组别_项目名
3. uv sync          # 失败 = 依赖记录有问题
4. uv run main.py   # 或者 uv run jupyter lab 里 Run All
5. 核对 output/ 里的图与数字，和你开发时是否一致
```

压缩包里必须有：`pyproject.toml`、`uv.lock`、`.python-version`、`README.md`、`main.py`。
不能有：`.venv/`、`__pycache__/`、`.ipynb_checkpoints/`、任何 `C:\Users\...` 绝对路径。

---

## 七、协作规则

见 [CONTRIBUTING.md](CONTRIBUTING.md)。三条硬的：

1. `main` 只接受 PR，不能直接 push；
2. 一个任务先开 Issue，PR 里写 `Closes #编号`；
3. `uv run pytest` 不过，不许合并。
