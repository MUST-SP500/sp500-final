# 决策日志（Decision Log）

> **只记决定，不记讨论过程。**
> 一条决定 = 日期 + 决定 + 理由 + 影响范围。讨论过程留在会议记录或聊天记录里。

## 模板

```markdown
### YYYY-MM-DD · 一句话标题
- **决定**：
- **理由**：
- **影响**：影响哪些文件 / 哪些块 / 是否需要改预注册
- **提出 / 确认**：
```

---

## 已记录的决定

### 2026-11-19 · 采用 GitHub Organization + 主责制协作
- **决定**：代码托管在 `MUST-SP500` 组织下，`main` 分支受保护，只接受 PR 合并
- **理由**：仓库跟随组织而非个人账号，便于交接与权限管理
- **影响**：所有人需先接受组织邀请；不能直接 push `main`

### 2026-11-19 · 初始化仓库骨架
- **决定**：建立 `src/` `tests/` `docs/` 结构与 CI
- **理由**：先定接口再并行开发，避免最后一周集成冲突
- **影响**：各块负责人只改自己目录下的文件

### 2026-11-19 · 改用 uv 管理依赖与运行环境
- **决定**：环境唯一真源改为 `pyproject.toml` + `uv.lock` + `.python-version`（Python 3.12），删除 `requirements.txt`；运行一律 `uv run`，加依赖一律 `uv add`
- **理由**：课程《Final Project 開發與測試流程指南》要求用 `uv sync` 一条命令复现环境，`uv.lock` 是老师能复现的凭据；`pip install` 装的包不进 lock，会造成环境漂移
- **影响**：全员需删除旧 `.venv` 后执行 `uv sync`；CI 改为 `astral-sh/setup-uv` + `uv sync --frozen`；README / CONTRIBUTING 中的 pip 指令同步改写；**不影响** `src/` 接口、`config.yaml` 参数与预注册
- **提出 / 确认**：熊泽睿 提出，G05 组会确认

### 2026-11-22 · 按分工说明书重构仓库结构
- **决定**：以 `SP500_分工说明书_函数与数据示例.md` 为唯一执行文档；删除两周版计划、按天手册、人话版等 7 份过时文档；新增 `data/raw/{yahoo,fred,wiki}`、`reports/*.md` 模板、`slides/outline.md`、`notebooks/01_eda.ipynb`、`output/`
- **理由**：六份计划并存，组员会照着过时的那份做；文件路径必须与说明书一一对应（`data/raw/yahoo/SPY.csv`、`reports/data_quality.md` 等），否则各块交付无处安放
- **影响**：所有交付路径以说明书为准；`config.yaml` 补齐 5 个标的；`src/` 各模块 docstring 上的"对应计划 4.x"改为"对应说明书第 N 步"；**不影响**预注册口径与切分协议
- **提出 / 确认**：G05 组内确认

### 2026-11-23 · 成分股只抓当前快照，不抓历史变更（ADR-003）
- **决定**：`src/scrape_wiki.py` 只抓 Wikipedia 的**当前成分表**（503 行 × 8 列），`fetch_changes()` / `to_turnover()` 明确抛 `NotImplementedError` 并注明原因
- **理由**：实测该页只有当前成分表。"Selected changes" 历史变更表已不在本页（`pd.read_html` 只解析出 2 张表：成分表 + 导航框），要做需 Wayback Machine 逐月抓取；成分股数据只用于论证幸存者偏差，不进特征与回测，一周工期不值得
- **影响**：报告只能写"当前 503 只、11 个板块分布 + 幸存者偏差说明"，**不能画年度换手图**（原交期里的 `eda_component_turnover.png` 取消，改为板块分布图）；`data/raw/wiki/constituents.csv` + manifest 条目可复现
- **提出 / 确认**：G05 组内确认（2026-11-23）

### 2026-11-26 · B5 深度学习框架定为 PyTorch CPU（ADR-004，关闭待确认 #2）
- **决定**：B5 的 DNN / CNN / Autoencoder 全部用 **PyTorch（CPU 版）**，作为 `pyproject.toml` 的可选依赖组 `dl`，用 `uv sync --extra dl` 安装；默认 `uv sync` 不装 torch
- **理由**：环境已统一到 uv，把 torch 放可选组可以让 CI 与只跑 sklearn 的人不必下几百 MB；本机实测 18 折 DNN 约 24 秒、CNN 约 28 秒，CPU 完全够用；`src/models_dl.py` 用 `_nn()` 惰性导入，未装 torch 时模块仍可 import（冒烟测试与 CI 不受影响）
- **影响**：跑 B5 必须加 `--extra dl`；DL 测试在无 torch 环境自动 skip；新增 `src/report_b5.py`（一键复现结果表）
- **提出 / 确认**：P5（mmskk55），2026-11-26

### 2026-11-26 · B5 结果表口径：多种子均值 + 未扣成本收益（ADR-005）
- **决定**：① 交付物 `predictions_dl.parquet` 固定种子 42；② 结果表另报 42 / 7 / 2024 三颗种子的**均值 ± 标准差**；③ 对比表除 AUC 外增加年化收益与夏普，口径为 `position × next_ret`（对数收益），空仓记 0、不计现金利息、不扣成本；④ AE 只跑 2008 单折，与 7 维 PCA、22 维全特征同表对比
- **理由**：说明书第 2 步第 3 条明确要求"报告多个种子的均值和波动，不要只报最好那次"。实测 CNN 三种子 AUC 为 0.5341 / 0.5178 / 0.5095（±0.0125），只报 0.5341 会让人误以为 CNN 显著优于逻辑回归（0.5220）；P5 不处理成本与现金收益，避免与 P6 的 `run_backtest` 口径打架
- **影响**：`reports/` 下新增 5 个 `b5_*.csv`（落在 P6 负责的目录，已在本 PR 说明，P6 可直接取用绘图）；`predictions_dl.parquet` 列结构不变（7 列，与说明书 P4 表一致）
- **提出 / 确认**：P5（mmskk55），2026-11-26

---

## 待确认事项（组会解决）

| # | 议题 | 截止 | 负责人 |
|---|---|---|---|
| 1 | 数据区间 1996 vs 1993 | G0 | P1 |
| 3 | 报告语言与页数 | G0 | P6 |
| 4 | 成员真实姓名与 GitHub 用户名对应表 | G0 | P6 |

> 原第 2 项「深度学习框架 PyTorch（CPU）」已由 P5 于 2026-11-26 关闭，见 ADR-004。
