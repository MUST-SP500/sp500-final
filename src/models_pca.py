"""PCA（B5）：折内降维 + 逻辑回归对比。

负责人：P5（B5）
对应：SP500_分工说明书 P5 第 1 步

红线：PCA 只能在训练折内部 fit，绝不能用全样本（否则测试年的信息漏进降维方向）。
图表（碎石图 / 载荷热力图）由 B2 出，本模块只出结果表。

用法：uv run python -m src.models_pca
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import load_config, project_path
from src.features import feature_columns, make_preprocessor

POSITION_THRESHOLD = 0.5  # 与 B4 的 R1 固定阈值一致，只为让 position 列可直接回测


def fit_pca(X_train, variance_target: float = 0.90):
    """在训练折上 fit PCA，返回 (pca, 保留成分数)。

    输入应是 make_preprocessor() 标准化后的矩阵；
    variance_target=0.90 表示保留到累计解释方差 ≥ 90% 为止。
    """
    pca = PCA(n_components=variance_target, svd_solver="full")
    pca.fit(X_train)
    return pca, int(pca.n_components_)


def pca_logreg(panel: pd.DataFrame, folds: list[dict], cfg: dict) -> pd.DataFrame:
    """18 折滚动：PCA+逻辑回归 与 全特征逻辑回归，同一份切分、同一套口径。

    前 7 列与 B4 的 predictions.parquet 完全一致（model / fold / decision_date /
    execution_date / y_true / y_pred / position），P6 可直接回测；
    末尾多一列 n_components，只作诊断，回答"为什么压成 7 维"。
    """
    feature_cols = feature_columns()
    seed = int(cfg["project"]["seed"])
    all_predictions = []

    for fold in folds:
        train_rows, test_rows = _fold_rows(panel, fold, feature_cols)
        if train_rows.empty or test_rows.empty:
            continue

        # 预处理只在训练折上 fit，测试年仅 transform
        preprocessor = make_preprocessor()
        X_train = preprocessor.fit_transform(train_rows[feature_cols])
        X_test = preprocessor.transform(test_rows[feature_cols])

        pca_model, n_components = fit_pca(X_train)

        # 两个模型吃同一份标准化数据，唯一区别是有没有降维
        variants = {
            "pca_logreg": (pca_model.transform(X_train), pca_model.transform(X_test)),
            "lr_full": (X_train, X_test),
        }

        for model_name, (X_fit, X_pred) in variants.items():
            classifier = LogisticRegression(max_iter=1000, random_state=seed)
            classifier.fit(X_fit, train_rows["label_up"])
            probability = classifier.predict_proba(X_pred)[:, 1]
            all_predictions.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "fold": fold["fold"],
                        "decision_date": test_rows.index,
                        "execution_date": _execution_dates(test_rows),
                        "y_true": test_rows["label_up"].to_numpy(),
                        "y_pred": probability,
                        "position": (probability >= POSITION_THRESHOLD).astype(int),
                        "n_components": n_components if model_name == "pca_logreg" else len(feature_cols),
                    }
                )
            )

    return pd.concat(all_predictions, ignore_index=True)


def auc_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """按模型汇总 AUC，一张小表直接抄进报告。"""
    rows = [
        {
            "model": model_name,
            "n_samples": len(part),
            "n_components": int(part["n_components"].iloc[0]),
            "auc": roc_auc_score(part["y_true"], part["y_pred"]),
        }
        for model_name, part in predictions.groupby("model", sort=False)
    ]
    return pd.DataFrame(rows)


def strategy_metrics(position, next_ret) -> dict:
    """把仓位序列变成"未扣成本"的年化收益与夏普，供 B5 各对比表统一口径。

    next_ret 是对数收益（ln(O[t+2]/O[t+1])），空仓记 0 收益、不计现金利息；
    正式回测（成本 / 日频 MaxDD / 现金收益）以 P6 的 run_backtest 为准。
    """
    pos = np.asarray(position, dtype=float)
    ret = np.asarray(next_ret, dtype=float)
    keep = ~np.isnan(ret)
    strategy = pos[keep] * ret[keep]
    n_weeks = len(strategy)
    if n_weeks == 0:
        return {"n_weeks": 0, "ann_return": np.nan, "sharpe": np.nan}

    total = float(np.exp(strategy.sum()) - 1.0)
    years = n_weeks / 52.0
    sd = float(strategy.std(ddof=1))
    return {
        "n_weeks": n_weeks,
        "ann_return": (1.0 + total) ** (1.0 / years) - 1.0,
        "sharpe": float(strategy.mean() / sd * np.sqrt(52.0)) if sd > 0 else np.nan,
    }


def _fold_rows(panel: pd.DataFrame, fold: dict, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按折取出训练/测试行，丢掉特征不全或没有标签的行。"""
    needed = [*feature_cols, "label_up"]
    train_rows = panel.reindex(pd.DatetimeIndex(fold["train_dates"])).dropna(subset=needed)
    test_rows = panel.reindex(pd.DatetimeIndex(fold["test_dates"])).dropna(subset=needed)
    return train_rows, test_rows


def _execution_dates(test_rows: pd.DataFrame) -> pd.Series:
    """执行日：面板里有就用，没有就留空。"""
    if "execution_date" in test_rows.columns:
        return test_rows["execution_date"]
    return pd.Series(pd.NaT, index=test_rows.index)


def _load_inputs():
    """读面板与切分，给命令行用。"""
    cfg = load_config()
    panel = pd.read_parquet(project_path(cfg["data"]["processed_dir"], "model_panel.parquet"))
    panel = panel.assign(date=pd.to_datetime(panel["date"])).set_index("date").sort_index()
    with open(project_path(cfg["data"]["splits"]), encoding="utf-8") as f:
        folds = json.load(f)
    folds = folds.get("folds", folds) if isinstance(folds, dict) else folds
    return panel, folds, cfg


if __name__ == "__main__":
    panel_data, fold_list, config = _load_inputs()
    predictions = pca_logreg(panel_data, fold_list, config)

    print(auc_table(predictions).to_string(index=False))
    pca_dims = predictions.loc[predictions["model"] == "pca_logreg", "n_components"]
    print(f"\n每折保留成分数：{pca_dims.mean():.1f}（均值）| 预测行数：{len(predictions)}")

    out_path = project_path("data", "processed", "pca_predictions.parquet")
    predictions.to_parquet(out_path, index=False)
    print(f"已写出 {out_path}")
