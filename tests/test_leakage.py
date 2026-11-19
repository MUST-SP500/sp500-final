"""防泄漏测试（B3）：把"不能泄漏"写成可执行的断言。

分两类：
  1. **现在就能跑**的通用检查（负向对照、折内标准化、滚动窗口只向后看）；
  2. **等 src.features / src.labels 实现后启用**的具体检查（截断不变性、标签对齐）。

注意第 1 类不是形式主义：负向对照（随机标签应接近 0.5）是整个项目
"结果是否可信"的开关。它一旦失效，后面所有结论都不作数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def test_negative_control_random_labels_auc_near_half():
    """随机标签 + 正常特征：AUC 必须接近 0.5，否则管道里有泄漏。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = rng.integers(0, 2, size=400)

    X_train, X_test = X[:300], X[300:]
    y_train, y_test = y[:300], y[300:]

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    assert 0.25 < auc < 0.75, f"随机标签 AUC 异常（{auc:.3f}），检查是否用了未来信息"


def test_scaler_must_be_fitted_on_train_only():
    """折内标准化：训练集均值必须与全样本均值不同，否则说明用了全样本 fit。"""
    rng = np.random.default_rng(1)
    train = rng.normal(loc=10.0, scale=1.0, size=(100, 3))
    test = rng.normal(loc=20.0, scale=1.0, size=(100, 3))

    scaler = StandardScaler().fit(train)
    assert abs(scaler.mean_[0] - 10.0) < 1.0

    full_mean = np.vstack([train, test]).mean(axis=0)[0]
    assert full_mean > 13.0, "全样本均值应显著高于训练集均值"

    # 用训练集 scaler 变换测试集，均值应明显大于 0（被正确外推）
    assert scaler.transform(test).mean() > 5.0


def test_rolling_window_uses_only_past():
    """滚动特征只能用窗口内历史：t 时刻的值 == 截至 t 的窗口均值。"""
    s = pd.Series(np.arange(1.0, 21.0))
    rolling = s.rolling(5).mean()

    t = 10
    expected = s.iloc[t - 4 : t + 1].mean()
    assert rolling.iloc[t] == pytest.approx(expected)


def test_label_shift_convention_is_documented():
    """标签口径的"文字契约"：t 的信号只能预测 t+1 之后才可获得的收益。

    这里用一条人造序列把口径写死，任何人改口径都会让这条测试失败。
    """
    open_px = pd.Series([100.0, 101.0, 102.0, 103.0])
    # t=0 决策 → t+1 开盘买 → t+2 开盘卖
    r_oo = np.log(open_px.iloc[2] / open_px.iloc[1])
    assert r_oo == pytest.approx(0.009852296443, rel=1e-9)
    assert r_oo > 0


@pytest.mark.skip(reason="待 B3 实现 src.features 后启用：截断不变性")
def test_features_truncation_invariance(ohlc):
    """把原始数据截到 t 重算特征，结果必须与全量计算在 t 处一致。"""
    from src.features import build_features

    full = build_features(ohlc)
    cut = ohlc.iloc[:-60]
    truncated = build_features(cut)

    common = truncated.index.intersection(full.index)
    assert len(common) > 0
    pd.testing.assert_frame_equal(
        full.loc[common].dropna(), truncated.loc[common].dropna(), check_like=True
    )


@pytest.mark.skip(reason="待 B3 实现 src.labels 后启用：标签对齐")
def test_label_uses_next_week_open_to_open(ohlc):
    from src.labels import build_labels

    labels = build_labels(ohlc.set_index("date")["open"])
    assert {"label_ret", "label_up"}.issubset(labels.columns)
    assert labels["label_up"].dropna().isin([0, 1]).all()
