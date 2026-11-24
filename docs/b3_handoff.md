# B3 交接与复核说明

日期：2026-11-24。最新权威入口为 `src/b3.py`；早期分步日志中的 26 列建模表已升级为 42 列。22 个特征及标签公式保持不变。

## 接手者先执行

在本 PR 合并后的最新仓库根目录执行：

```powershell
uv sync --locked
# 先准备好 B1 的 daily_panel.parquet 和 weekly_panel.parquet
uv run python -m src.b3
uv run pytest
```

如果没有 B1 面板，运行 download/clean 或接收 P1 同一版本的两个面板。数据不进 git；只拉代码不会自动获得面板。B3 不重下载、不改写 B1 面板。

`src.features`、`src.labels` 的旧入口保留；labels 入口已转到统一生成过程，防止覆盖完整交接表。对外推荐只用 `src.b3`，可传 `--config path.yaml`。

## 输出契约

| 文件 | 内容 |
|---|---|
| weekly_features.parquet / CSV | 1,566 行，date + 22 特征 |
| weekly_labels.parquet / CSV | date + label_ret、label_up、next_ret |
| model_panel.parquet / CSV | 1,566 行 × 42 列，包含 B1 原始 12 列、22 特征、3 标签、5 元数据列 |
| splits.json | 2008–2025 共 18 折，含闭区间边界和精确 train_dates/test_dates |
| validation_split.json | 1996–2005 训练、2006–2007 一次性验证，已 purge |
| b3_sample_counts.csv | 逐折有效训练/评估样本数 |
| feature_missingness.csv | 22 个特征的全时段缺失比例，包含预热期 |

元数据：execution_date 为下周首个真实交易日；exit_date 为下下周首个真实交易日；holding_trading_days 是持有期间交易日数；history_ready 标记是否已有 252 个历史日间隔；label_available 标记三类标签是否完整。

这些元数据只供对齐、过滤和回测。唯一模型输入清单是 `feature_columns()`；不得使用 select_dtypes 或“除标签外全部列”，否则会把未来日期、标签可获得标记等带入模型。

## P4/P5 的最小调用

```python
import json
import pandas as pd
from src.b3 import fold_frames
from src.config import project_path
from src.features import feature_columns, make_preprocessor

panel = pd.read_parquet(project_path('data/processed/model_panel.parquet'))
folds = json.loads(project_path('data/processed/splits.json').read_text(encoding='utf-8'))
for fold in folds:
    train, test = fold_frames(panel, fold)
    pre = make_preprocessor()  # 每折新建
    X_train = pre.fit_transform(train[feature_columns()])
    X_test = pre.transform(test[feature_columns()])
    y_train = train['label_up'].astype(int)  # 分类
    r_train = train['label_ret']            # 回归，依预注册口径
    # model.fit(X_train, y_train)
    # 将预测绑定 test.index（决策日期），不要按行号拼接
```

`fold_frames` 排除预热和未知标签；其余偶发缺失交给训练中位数填补。默认评估有 938 个样本外周，2025 年只有 51 个可评估周，末两周未来标签不可得。若需对未知未来周产生预测，用 `require_labels=False`，但不可把这些周用于指标或回测收益计算。

一次性选参用 validation_split.json 中唯一一条切分，通过同一个 fold_frames 读取。训练 468 周、验证 104 周。不要用 2008–2025 测试 AUC 调参。

P5：PCA 必须在本折 X_train 上 fit；CNN/AE 需要 [0,1] 输入可用 `make_preprocessor(scaling='minmax')`。早停另划训练尾部验证时，先划分、purge，再仅在内部训练段 fit 预处理和 PCA/AE，不能把早停验证段一起 fit。序列窗口和缺失日期需自行保留时间索引，不可删周后直接拼接成连续序列。

## P6 的对接事项

当前 src/pipeline.py 仍把 None 传给 features/labels/splits，不能据“模块可导入”判断可运行。无需改动 B3 函数来迎合 None；集成时在 clean 后调用 `from src.b3 import run; run(cfg)`，即可生成模型层输入。此复核未改写 P6 所属 pipeline.py 或模型实现。

回测必须将决策日预测对应到 execution_date–exit_date，直接用 next_ret 作为简单收益，不再多 shift 一周。成本、现金收益仍由 B6 处理。holding_trading_days 指交易日，不是日历天；利息折算按预注册口径。不要用下周周末的利率替代执行时已知的利率；本表没有替 B6 计算现金收益。

## 复核修复与未决事项

修复：补回 B1 基础价格/宏观列及成交日期；补独立验证切分；统一入口避免旧 labels 覆盖交接表；增加 CNN/AE 的 MinMax 选项；拒绝空日表、重复行索引、价格缺失以及把日频或删周后的价格误传入标签；新增 ML/PCA 交接测试。

未决：P4/P5 正式模型、P6 回测与主流程仍由相应组员完成；B1 的 ^GSPC 交叉校验待补；预注册时间平移“策略失效”尚需正式回测验证。AUC 对照和单元测试不证明彻底排除供应商数据修订等所有偏差。

提交时仅添加 src、tests、相关 docs/reports，不提交 data、.venv 或预测结果。开发分支为 feat/b3-features-labels-splits；合并前可拉取该分支使用，须 PR 审查合并后同学才能通过 main 获取。
