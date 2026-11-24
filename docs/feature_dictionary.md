# 特征字典（Feature Dictionary）

22 个特征，**全部比率型**。口径冻结后不再改；要改先改本文件，再改 `src/features.py` 的 `FEATURE_GROUPS`。

## 硬性约束

1. **禁止价格绝对水平**：只用比率、收益率、变化率、滚动 z-score。
2. 所有滚动窗口**只向后看**，窗口末端 ≤ 决策时点 `t`。
3. 截断不变性：把原始数据截到 `t` 重算，结果必须与全量计算一致（`tests/test_leakage.py`）。

## 五组特征

| 组 | 特征 | 公式 / 窗口 | 经济含义 | 方向预期 | 个数 |
|---|---|---|---|---|---|
| 动量 | `ret_1m` | `close/close.shift(21)-1` | 短期趋势 | + | 5 |
| | `ret_3m` | `close/close.shift(63)-1` | 中期趋势 | + | |
| | `ret_6m` | `close/close.shift(126)-1` | 中期趋势 | + | |
| | `ret_12m` | `close/close.shift(252)-1` | 长期趋势 | + | |
| | `mom_12_1` | `ret_12m - ret_1m` | 剔除短期反转 | + | |
| 趋势 | `px_ma20` | `close/MA20 - 1` | 短期偏离 | − | 5 |
| | `px_ma50` | `close/MA50 - 1` | 中期偏离 | − | |
| | `px_ma200` | `close/MA200 - 1` | 长期偏离 | − | |
| | `ma50_ma200` | `MA50/MA200 - 1` | 金叉/死叉 | + | |
| | `ma200_slope` | `MA200/MA200.shift(21) - 1` | 长期趋势斜率 | + | |
| 波动 | `vol_20` | `ret.rolling(20).std() * sqrt(252)` | 近期波动 | − | 5 |
| | `vol_60` | `ret.rolling(60).std() * sqrt(252)` | 中期波动 | − | |
| | `downside_vol` | 负收益的 `rolling(60).std()` | 下行风险 | − | |
| | `atr_14` | `ATR14 / close` | 归一化波幅 | − | |
| | `drawdown` | `close / cummax(close) - 1` | 当前回撤深度 | + | |
| 情绪 | `vix_level` | `vix`（已 shift 1） | 恐慌水平 | − | 4 |
| | `vix_chg` | `vix/vix.shift(5) - 1` | 情绪变化 | − | |
| | `dgs10_chg` | `dgs10.diff(5)` | 利率变化 | − | |
| | `term_spread` | `dgs10 - dtb3` | 期限利差 | + | |
| 量能 | `volume_z` | `(volume - MA20)/std20` | 交易拥挤 | − | 3 |
| | `rsi_14` | 标准 RSI(14) / 100 | 超买超卖 | − | |
| | `macd_hist` | `MACD(12,26,9)` 柱 / close | 动量拐点 | + | |

> `drawdown` 为负值，方向预期 "+" 表示"回撤越深未来收益越高"（均值回复假设）。

## 预处理顺序（每折内独立执行）

```text
原始特征 → 缺失处理（按字典） → winsorize（训练集 1%/99%）
         → 标准化 / 归一化（训练集 fit） → 可选 PCA/AE → 模型
```

| 模型 | 缩放方式 |
|---|---|
| 逻辑回归 / Ridge / PCA / DNN | `StandardScaler` |
| CNN / Autoencoder 输入 | `MinMaxScaler`（网络对输入范围敏感） |
| 树模型 | 不敏感，但为了统一 pipeline 仍保留 |

## 待补（各块负责人填）

- [ ] 每个特征的缺失率实测值（B3 填）
- [ ] 特征间 VIF 与相关系数（B2 出图，B3 填表）
- [ ] 与未来收益的 IC（附录用）

## B3 日频特征实现说明（2026-11-24）

- 入口：`uv run python -m src.features`。读取 B1 日表计算，再按周表的 date 精确对齐，输出 `data/processed/weekly_features.parquet` 及 CSV；不改写 B1 面板。
- 返回的特征矩阵与日频输入索引一致，严格采用 `feature_columns()` 的 22 列顺序。
- 日收益使用简单收益，滚动标准差使用 ddof=1。downside_vol 是最近 60 个交易日内负收益的样本标准差（不年化），负收益少于 2 个时保留缺失；不先删除正收益行再 rolling。
- ATR14 和 RSI14 使用 Wilder 平滑，先以 14 个有效观察的均值初始化，再以 alpha=1/14 递推。RSI 归一化到 [0,1]，连续不涨不跌为 0.5。
- MACD 使用 adjust=False 的 EMA(12,26,9)，待对应窗口预热后输出，柱除以 close。
- volume_z 使用 20 日窗口；成交量标准差为零时保留缺失。宏观数据直接使用 B1 已滞后的字段，不重复 shift。
- 预热期与不可定义的特征保留 NaN，不填零，不执行全样本标准化。
- 本地本次输出：1,566 行、date + 22 特征；22 特征均非缺失的决策周 1,514 行，首个完整日期 1997-01-03。这不是模型评测结果。
- 验证：`tests/test_features.py` 覆盖公式、未来数据扰动、多个截断点、价格同比缩放、异常日期和平盘边界；本次全套测试 105 passed、3 skipped。既有 skip 尚待后续标签、防泄漏和 B6 工作完成，不代表审计全部通过。

## B3 标签实现记录（2026-11-24）

- `uv run python -m src.labels` 读取周频复权 open，输出 weekly_labels.parquet，并将已有 weekly_features.parquet 与标签按 date 一对一合并为 model_panel.parquet 和 CSV。
- label_ret = ln(open[w+2] / open[w+1])，next_ret = open[w+2] / open[w+1] - 1；label_up 为可空整数，收益严格大于零记 1，否则记 0。输入价格缺失导致的未知标签及最后两周标签均保留缺失，不前向填充。
- 本次建模表为 1,566 行 × 26 列（date + 22 特征 + 3 标签）；特征与标签全部可用的行有 1,512，决策日期为 1997-01-03 至 2025-12-19。文件保留全部行，不预先删除缺失，不拟合预处理，也不挑选测试结果。
- tests/test_labels.py 验证人工周价格、涨跌与零收益、未知未来与中间缺失、非法价格、索引和简单/对数收益关系；已启用既有标签对齐与特征截断测试。完整 B3 泄漏审计仍待切分、折内预处理与对照实验完成。

## B3 时间切分实现记录（2026-11-24）

- 入口：`uv run python -m src.splits`。基于 B1 完整周决策日期生成 `data/processed/splits.json`，共 18 折（2008–2025），不按特征缺失先删周，避免改变 purge 的周序关系。
- 每折使用当年 01-01 至 12-31 的周决策日期作为测试；训练从配置 train_start 开始，去掉测试年前末端 purge_weeks 个实际决策周，再保证训练截止日期距测试年开始至少 14 天。
- 对每个训练决策周检查 w+2 周末严格早于测试年开始。w+2 周末是卖出开盘时间的保守上界，避免只检查决策日期却漏掉标签时间。
- JSON 保留 fold/train_start/train_end/test_start/test_end/purge_weeks，并附 train_dates/test_dates 精确日期列表。区间边界为含端点；P4/P5 应按日期列表选样本，再排除未知标签及不满足预热要求的行。每折训练时独立拟合预处理。
- 2008 折：训练截至 2007-12-14，624 个候选训练周，52 个测试决策周；2025 折：训练截至 2024-12-13，1,511 个候选训练周，53 个测试决策周。数量未剔除预热或未知标签，不等于可评测样本数。
- 2006–2007 仍只用于一次性选参；之后冻结参数。它们可以作为 2008 起扩展窗口的历史训练数据，但不能拿测试年份重新调参。
- tests/test_splits.py 覆盖 18 年完整覆盖、无重复测试周、标签可获得时间、purge 幂等性与不修改输入、JSON 往返及异常日期/缺失年份。

## B3 折内预处理规则与验证（2026-11-24）

- `make_preprocessor(winsor=0.01)` 返回 sklearn Pipeline：TrainingMedianImputer → TrainingQuantileClipper → StandardScaler。三个步骤均只用当前训练折 fit；测试集只能 transform。
- 先剔除滚动窗口预热期，不以中位数填造尚无足够历史的特征。预热后偶发缺失可用训练中位数填补；训练整列缺失、无穷值、输入字段顺序或数量不一致会报错。
- 先用训练中位数填补，再从填补后的训练数据估计每列 1%/99% 分位数；测试值按冻结阈值截尾，再按训练截尾后均值与方差标准化。常量列由 StandardScaler 安全处理。
- tests/test_preprocessing.py 验证训练参数与手算一致，测试集极端值及缺失不会更新任一拟合参数，支持 sklearn clone，并检测未拟合调用、非法阈值、全缺失列及字段错位。
- 已在真实建模表上逐折运行 18 次独立 fit/transform（仅作工程验证，未训练预测模型或调参），确认所有输出有限、每折 22 列、测试 transform 不修改训练 scaler。结果见 data/processed/preprocessing_checks.csv。
- 全套测试：121 passed、1 skipped；模型层端到端负向/未来信息注入对照与最终泄漏审计尚未完成。

## 最终交接补充

完整模型表升级为 42 列（保留 B1 列，追加执行日期、退出日期、持有交易日数、历史充分性与标签可用性）。特征选择始终只用 feature_columns() 返回的 22 列。最新复现命令与验证集调用见 docs/b3_handoff.md，统一入口为 src.b3。make_preprocessor 默认 StandardScaler，P5 可显式传 scaling='minmax'；两者都在训练折拟合。
