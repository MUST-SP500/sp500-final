"""B5 报告生成：四套模型的结果汇总，全部从这里一键复现。

负责人：P5（B5）
对应：SP500_分工说明书 P5 第 1–5 步 + 第 7 步（过拟合诊断）

用法：uv run python -m src.report_b5

产出（reports/ 下四个 csv）：
    b5_model_results.csv       18 折主对比：AUC / 年化收益 / 夏普（未扣成本）
    b5_ae_pca_2008.csv         2008 单折：AE 瓶颈 vs PCA vs 全特征
    b5_overfit_diagnostics.csv 2008 折训练/验证/测试 AUC、参数量、早停轮数
    b5_pca_variance.csv        PC1..PC10 解释方差与累计（回答"为什么是 7 维"）
    b5_seed_stability.csv      种子 42 / 7 / 2024 的 AUC，用于报均值和波动

口径说明：收益 = position × next_ret（对数收益），空仓记 0、不计现金利息、
不扣成本；正式回测以 P6 的 run_backtest 为准。
"""

from __future__ import annotations

import copy
import json

import pandas as pd

from src.config import load_config, project_path
from src.features import feature_columns, make_preprocessor
from src.models_pca import _fold_rows, fit_pca, pca_logreg, strategy_metrics

MAIN_SEED = 42
SEED_LIST = [42, 7, 2024]  # 说明书第 2 步要求"报告多个种子的均值和波动"
VARIANCE_TARGET = 0.90


def load_inputs():
    """读面板、切分与配置。"""
    cfg = load_config()
    panel = pd.read_parquet(project_path(cfg["data"]["processed_dir"], "model_panel.parquet"))
    panel = panel.assign(date=pd.to_datetime(panel["date"])).set_index("date").sort_index()
    with open(project_path(cfg["data"]["splits"]), encoding="utf-8") as f:
        folds = json.load(f)
    folds = folds.get("folds", folds) if isinstance(folds, dict) else folds
    return panel, folds, cfg


def _next_ret(prediction: pd.DataFrame, panel: pd.DataFrame):
    """按决策日把 next_ret 对齐到预测表的每一行。"""
    return panel["next_ret"].reindex(pd.DatetimeIndex(prediction["decision_date"])).to_numpy()


def _summary_row(name: str, prediction: pd.DataFrame, panel: pd.DataFrame, n_features: int) -> dict:
    metrics = strategy_metrics(prediction["position"].to_numpy(), _next_ret(prediction, panel))
    from sklearn.metrics import roc_auc_score

    return {
        "model": name,
        "n_features": n_features,
        "n_rows": len(prediction),
        **{key: round(value, 4) for key, value in metrics.items()},
        "auc": round(float(roc_auc_score(prediction["y_true"], prediction["y_pred"])), 4),
    }


def run_headline_models(panel, folds, cfg) -> dict[str, pd.DataFrame]:
    """主对比用的五套预测：LR 全特征、PCA+LR、DNN、CNN、CNN+MinMax。"""
    from src.models_dl import run_dl_folds

    pca_predictions = pca_logreg(panel, folds, cfg)
    return {
        "lr_full": pca_predictions[pca_predictions["model"] == "lr_full"].reset_index(drop=True),
        "pca_logreg": pca_predictions[pca_predictions["model"] == "pca_logreg"].reset_index(drop=True),
        "dnn_standard": run_dl_folds(panel, folds, cfg, model_kind="dnn"),
        "cnn_standard": run_dl_folds(panel, folds, cfg, model_kind="cnn"),
        "cnn_minmax": run_dl_folds(panel, folds, cfg, model_kind="cnn", scaling="minmax"),
    }


def model_table(predictions: dict[str, pd.DataFrame], panel, n_groups: int) -> pd.DataFrame:
    """18 折主对比表；PCA 的特征数取每折均值（7–8 之间）。"""
    rows = []
    for name, prediction in predictions.items():
        if name == "pca_logreg":
            n_features = int(round(prediction["n_components"].mean()))
        else:
            n_features = n_groups
        rows.append(_summary_row(name, prediction, panel, n_features))
    return pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)


def ae_table_2008(panel, folds, cfg) -> pd.DataFrame:
    """2008 单折：AE 6 维 / PCA 7 维 / 全 22 维，同一个逻辑回归。"""
    from src.models_dl import autoencoder_vs_pca

    table = autoencoder_vs_pca(panel, folds[0], cfg)
    for column in ("auc", "ann_return", "sharpe"):
        table[column] = table[column].astype(float).round(4)
    return table.sort_values("auc", ascending=False, ignore_index=True)


def overfit_diagnostics(panel, folds, cfg) -> pd.DataFrame:
    """2008 折的过拟合体检：训练 / 验证 / 测试三段 AUC + 参数量。"""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from src.models_dl import (
        _fold_rows,
        _split_validation,
        _to_model_input,
        build_cnn,
        build_dnn,
        train_and_predict,
    )

    cols = feature_columns()
    seed = int(cfg["project"].get("seed", MAIN_SEED))
    fold = folds[0]
    train_rows, test_rows = _fold_rows(panel, fold, cols)
    core_rows, val_rows = _split_validation(train_rows)

    preprocessor = make_preprocessor()
    preprocessor.fit(core_rows[cols])

    rows = []
    for kind, builder in (("dnn", build_dnn), ("cnn", build_cnn)):
        X_core, _, y_core = _to_model_input(core_rows, preprocessor, cols, kind, 20)
        X_val, _, y_val = _to_model_input(val_rows, preprocessor, cols, kind, 20)
        X_test, _, y_test = _to_model_input(test_rows, preprocessor, cols, kind, 20)
        if min(len(X_core), len(X_val), len(X_test)) == 0:
            continue

        model = builder(len(cols), seed) if kind == "dnn" else builder(20, len(cols), seed)
        train_and_predict(model, X_core, y_core, X_val, y_val, X_test, seed=seed)

        def _auc(X, y) -> float:
            with torch.no_grad():
                probability = torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).view(-1).numpy()
            return float(roc_auc_score(y, probability))

        auc_train, auc_val, auc_test = _auc(X_core, y_core), _auc(X_val, y_val), _auc(X_test, y_test)
        rows.append(
            {
                "model": kind,
                "params": int(sum(p.numel() for p in model.parameters())),
                "fold": fold["fold"],
                "n_train": len(X_core),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "auc_train": round(auc_train, 4),
                "auc_val": round(auc_val, 4),
                "auc_test": round(auc_test, 4),
                "gap_train_test": round(auc_train - auc_test, 4),
                "epochs_used": len(model.loss_history_["train"]),
            }
        )
    return pd.DataFrame(rows)


def pca_variance_table(panel, folds, cfg, top: int = 10) -> pd.DataFrame:
    """在 2008 折训练段上 fit 一次 PCA，列出前 top 个主成分的解释方差。"""
    cols = feature_columns()
    train_rows, _ = _fold_rows(panel, folds[0], cols)
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(train_rows[cols])
    pca_model, _ = fit_pca(X_train, variance_target=VARIANCE_TARGET)
    ratio = pca_model.explained_variance_ratio_[:top]
    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(ratio))],
            "explained": [round(float(v), 4) for v in ratio],
            "cumulative": [round(float(v), 4) for v in ratio.cumsum()],
        }
    )


def seed_stability(panel, folds, cfg, reused: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """同一份切分、同一套超参，只换种子，看 AUC 波动有多大。"""
    from sklearn.metrics import roc_auc_score

    from src.models_dl import run_dl_folds

    rows = []
    for seed in SEED_LIST:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["project"]["seed"] = seed
        for kind, key in (("dnn", "dnn_standard"), ("cnn", "cnn_standard")):
            prediction = reused[key] if seed == MAIN_SEED and key in reused else run_dl_folds(
                panel, folds, cfg_seed, model_kind=kind
            )
            rows.append(
                {
                    "seed": seed,
                    "model": kind,
                    "n_rows": len(prediction),
                    "auc": round(float(roc_auc_score(prediction["y_true"], prediction["y_pred"])), 4),
                }
            )
        print(f"  种子 {seed} 完成")
    return pd.DataFrame(rows)


def main() -> None:
    panel, folds, cfg = load_inputs()
    cols = feature_columns()

    print("1/5 跑主对比（LR / PCA / DNN / CNN / CNN-MinMax）…")
    predictions = run_headline_models(panel, folds, cfg)

    print("2/5 跑 2008 单折 AE 对比…")
    ae = ae_table_2008(panel, folds, cfg)

    print("3/5 跑过拟合诊断（2008 折）…")
    diagnostics = overfit_diagnostics(panel, folds, cfg)

    print("4/5 算 PCA 解释方差…")
    variance = pca_variance_table(panel, folds, cfg)

    print("5/5 跑种子稳定性（42 / 7 / 2024）…")
    stability = seed_stability(panel, folds, cfg, predictions)

    results = model_table(predictions, panel, len(cols))
    outputs = {
        "b5_model_results.csv": results,
        "b5_ae_pca_2008.csv": ae,
        "b5_overfit_diagnostics.csv": diagnostics,
        "b5_pca_variance.csv": variance,
        "b5_seed_stability.csv": stability,
    }
    for filename, frame in outputs.items():
        path = project_path("reports", filename)
        frame.to_csv(path, index=False)
        print(f"已写出 {path}")

    print("\n=== 18 折主对比（未扣成本）===")
    print(results.to_string(index=False))
    print("\n=== 2008 单折：AE vs PCA vs 全特征 ===")
    print(ae.to_string(index=False))
    summary = stability.groupby("model")["auc"].agg(["mean", "std", "min", "max"]).round(4)
    print("\n=== 种子稳定性（42/7/2024）===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
