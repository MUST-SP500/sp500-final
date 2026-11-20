# 协作规范（CONTRIBUTING）

> 目标：六个人同时写代码，**不互相踩**、**不最后一周才集成**、**换台电脑也能跑**。

## 1. 环境（第一次 5 分钟）

依赖与 Python 版本由 `pyproject.toml` + `uv.lock` + `.python-version` 锁定，**不要用 `python -m venv` 和 `pip install`**。

```bash
# 0. 装 uv（一次性）；Windows 在 PowerShell 里跑：
#    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync                # 按 uv.lock 复现环境（自动建 .venv，含 dev 依赖）
uv run pytest -q       # 应 45 passed, 3 skipped
```

`uv run pytest` 全绿 = 环境正确。红了先别看别人的代码，先看报错的第一行。

**加依赖只走一条路**：`uv add 包名`（工具类用 `uv add --dev 包名`）。它会同时写 `pyproject.toml` 和 `uv.lock`，这两个文件必须一起提交。用 `pip install` 装的包不会进 lock，别人就复现不出来。

运行任何脚本都用 `uv run`，不要用裸 `python`：

```bash
uv run python -m src.pipeline --check
uv run python -m src.pipeline --config config.yaml
```

## 2. 分支命名

| 类型 | 格式 | 例子 |
|---|---|---|
| 功能 | `feat/b<块号>-<做什么>` | `feat/b1-download` |
| 修复 | `fix/<做什么>` | `fix/week-key-holiday` |
| 文档 | `docs/<做什么>` | `docs/preregistration` |

**一人一分支，分支寿命不超过 2 天。** 长期分支 = 冲突制造机。

## 3. 提交信息

```text
<type>(<块号>): <一句话做什么>

feat(b1): 7 标的并发下载 + 指数退避重试
fix(b3): 修正感恩节周的周键错位
docs(b6): 补回测对账说明
```

type 只用：`feat` / `fix` / `docs` / `test` / `chore` / `refactor`。

## 4. 一个任务的完整流程

```bash
# 1. 认领 Issue（在网页上把自己设为 Assignee，看板拖到 In Progress）
git switch main && git pull
git switch -c feat/b1-download

# 2. 写代码 + 写测试 + 本地跑通
uv run pytest -q

# 3. 提交、推送、开 PR
git add -A
git commit -m "feat(b1): 7 标的并发下载"
git push -u origin feat/b1-download

# 4. PR 正文写 Closes #12，找至少 1 人 review
# 5. 通过后 squash merge，删分支
```

## 5. Review 规则

- 每个 PR **至少 1 人**通过；数据 / 回测相关 PR 需 P1 或 P6 额外看一次。
- **48 小时内必须响应**（哪怕只是"我今晚看"）。
- Review 只看三件事：**对不对、会不会泄漏、有没有测试**。
- 不许自己批自己。

## 6. 目录归属（减少冲突）

| 谁 | 主要改这些 |
|---|---|
| P1 | `src/data_download.py`, `src/data_clean.py`, `src/scrape_wiki.py`, `src/validate.py`, `data/` |
| P2 | `notebooks/01_*.ipynb`, `reports/figures/eda_*`, `src/plot_style.py` |
| P3 | `src/features.py`, `src/labels.py`, `src/splits.py`, `tests/test_leakage.py` |
| P4 | `src/models_ml.py`, `src/evaluate.py`, `src/decision.py` |
| P5 | `src/models_pca.py`, `src/models_dl.py` |
| P6 | `src/backtest.py`, `src/pipeline.py`, `reports/`, `slides/` |

要改别人的文件，**先在 Issue 里说一声**，不要直接改。

## 7. 完成标准（Definition of Done）

一个任务算完成，必须同时满足：

- [ ] 代码在 `src/` 里，不是只在 notebook 里
- [ ] 有测试，且 `uv run pytest -q` 通过
- [ ] PR 已被 review 并合并到 `main`
- [ ] 相关文档（字典 / decisions）已更新
- [ ] 别人 clone 下来能跑通

## 8. 禁止事项

- ❌ 直接 push 到 `main`
- ❌ 提交数据文件（`data/raw/` 已被 gitignore）
- ❌ 提交密钥、token、`.env`
- ❌ 用 notebook 当交付物
- ❌ 在测试集上调参（见 `docs/preregistration.md`）
- ❌ 看到别人分支有冲突就 `--force` 覆盖
