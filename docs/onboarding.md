# 组员上手指南｜B1 数据层已就绪

> 面向 G05 全体成员。照做即可拿到清洗好的数据，开始 B2 / B3。
> 预计耗时：首次 5–10 分钟（主要耗在装 uv 和拉依赖）。

---

## 0. 先确认一件事

仓库地址：https://github.com/MUST-SP500/sp500-final

打开能看见代码 = 你已经被拉进组了。打不开就是邀请还没接受，去邮箱点 GitHub 的邀请链接（注意查垃圾邮件）。

---

## 1. 装 uv（只需要做一次）

**Windows**（PowerShell，不是 CMD）：

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux**：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

装完关掉终端重开，输入 `uv --version` 能看到版本号就行。

> 不要用 `pip install`，也不要自己 `python -m venv`。本项目依赖唯一真源是 `pyproject.toml` + `uv.lock`，混用会让别人复现不了。

---

## 2. 第一次拉代码

```bash
git clone https://github.com/MUST-SP500/sp500-final.git
cd sp500-final
```

已经有本地仓库的（之前拉过），只要：

```bash
git switch main
git pull
```

⚠️ **必须是最新的 `main`**。旧版只有原始数据，没有清洗好的面板。

---

## 3. 装依赖

```bash
uv sync
```

第一次会拉 100 多个包，约 1–2 分钟。之后再跑就是秒过。

---

## 4. 生成数据（两条命令，约 3 秒）

```bash
uv run python -m src.pipeline --stage download
uv run python -m src.pipeline --stage clean
```

第一条从雅虎和 FRED 抓 5 个标的，第二条做复权 + 周频聚合 + 面板合并。

**数据文件不进 git**，所以刚 clone 下来 `data/processed/` 是空的，必须跑这两条。

---

## 5. 验收（三条硬指标，对不上就别往下做）

```bash
uv run pytest
```

预期：**100 passed, 3 skipped**（3 个 skip 是 B3/B6 还没写的占位，正常）。

```bash
uv run python -c "import pandas as pd; w=pd.read_parquet('data/processed/weekly_panel.parquet'); print(w.shape, w.date.min().date(), w.date.max().date())"
```

预期：`(1566, 12) 1996-01-05 2025-12-31`

三个都要对上：**1,566 行、12 列、1996-01-05 ~ 2025-12-31，每年 52–53 周**。对不上说明代码或数据版本不一致，别硬着头皮做，先在群里问。

---

## 6. 你要用的数据在哪

| 文件 | 用途 |
|---|---|
| `data/processed/weekly_panel.parquet` | **建模主表**，一行 = 一周（1,566 行 × 12 列） |
| `data/processed/daily_panel.parquet` | **特征工程用**，一行 = 一天（7,550 行 × 12 列） |
| `data/raw/manifest.json` | 这批数据是哪天下的、每个文件的行数和 sha256 |
| `reports/data_quality.md` | 质量报告（缺失、极端值、除息事件、与 ^GSPC 对账） |
| `docs/variable_descriptions.md` | **每个字段什么意思 → 只查这一个文件** |
| `docs/reproduce_log.md` | 复现记录（环境、耗时、比对结果） |

常用命令：

```bash
# 用 Excel 看数据（会在 data/processed/csv/ 生成两个 CSV）
uv run python -m src.export_csv

# 从零重跑并校验一致性（约 4 秒，会删掉现有数据文件）
uv run python -m src.reproduce --full-rebuild
```

### 两条必须遵守的口径

1. **路线 A（已定）**：特征在 `daily_panel` 上算（20/60 日均线、20 日波动率、RSI…），按 ISO 周取该周最后一个交易日，再与 `weekly_panel.date` 对齐。周频只有 1,566 行，直接算 20 周均线会吃掉前两年样本。
2. **不许自己复权、不许自己 `resample('W-FRI')`**。口径唯一真源是 `src/data_clean.py`，要改先改 `docs/variable_descriptions.md` + 改测试，再改代码。

另：`weekly_panel` 里 **没有** `label_ret` / `label_up` / `next_ret` 三列，这三个由 B3 的 `src/labels.py` 追加。

---

## 7. 写代码、推代码的流程

**不许直接推 `main`。** 一律走「分支 → PR → 1 人 review → Squash merge → 删分支」。

```bash
git switch main
git pull                          # 每次开分支前先同步
git switch -c feat/b2-splits      # 分支名：feat/你负责的模块
# ...写代码、写测试...
uv run pytest                     # 必须全绿
git add src/splits.py tests/test_splits.py
git commit -m "feat(b2): 时间序列切分与防泄漏检查"
git push -u origin feat/b2-splits
```

推完终端会打印一个 `Create a pull request` 链接，点开 → 填标题和「改了什么 / 怎么验证」→ 右侧选一位 reviewer → `Create pull request` → CI 变绿、reviewer 点 Approve → `Squash and merge` → `Delete branch`。

PR 标题格式：`feat(b2): ...` / `fix(b3): ...` / `docs(b6): ...`

---

## 8. 常见问题

**Q1：`download` 报 403 / 超时 / 连接被拒绝**
雅虎和 FRED 在国内网络经常连不上。先开 VPN（TUN 模式最稳）再跑。实在不行找 P1 要三个文件：`weekly_panel.parquet`、`daily_panel.parquet`、`manifest.json`，放到对应路径就能用，不用跑 download。

**Q2：`git push` 报 `fatal: could not read Username`**
Windows 凭据管理器里的 GitHub 登录失效了。关掉终端重开，`git push` 会弹浏览器登录，登录一次就好。

**Q3：`uv sync` 报 lock 文件不一致**
说明 `uv.lock` 被改过。先 `git pull` 拿最新版本；还不行就 `git checkout -- uv.lock`，不要手动编辑它。

**Q4：跑完 `pytest` 不是 100 passed**
先 `git pull` 确认在最新 `main` 上。还是不行就是环境问题：删掉 `.venv` 目录，重新 `uv sync`。

**Q5：数据行数对不上（不是 1,566 行）**
九成是代码版本旧了（`git log -1` 看 commit）。数据本身是 1996–2025 固定区间，行数不该变。

---

## 9. 分工速查

| 模块 | 文件 | 负责人 |
|---|---|---|
| B1 数据 | `src/data_download.py`、`src/data_clean.py`、`src/scrape_wiki.py` | 熊泽睿 |
| B2 切分与防泄漏 | `src/splits.py` | 见分工说明书 |
| B3 特征与标签 | `src/features.py`、`src/labels.py` | 见分工说明书 |
| B4 模型 | `src/models_ml.py`、`src/decision.py` | 见分工说明书 |
| B5 降维 / 深度学习 | `src/models_pca.py`、`src/models_dl.py` | 见分工说明书 |
| B6 回测与报告 | `src/backtest.py`、`src/evaluate.py`、`src/pipeline.py` | 见分工说明书 |

完整函数级说明看仓库根目录的 `SP500_分工说明书_函数与数据示例.md`。
