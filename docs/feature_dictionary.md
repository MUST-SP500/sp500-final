# 特征字典（Feature Dictionary）

22 个特征，**全部比率型**。冻结于 G1（D4）。

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
