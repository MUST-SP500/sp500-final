"""传统机器学习（B4）：基线 + LR / Ridge / RF。

负责人：P4（B4）
对应：SP500_分工说明书 P4 第 1–6 步（基线 / 模型工厂 / 一次性选参 / 18 折滚动 / 评测与性能）

约束：
  - 全部使用 src.splits 生成的同一份 splits.json
  - 预处理（winsorize / scaler）只在训练折内 fit
  - 超参与阈值只在 2006–2007 上选一次，之后冻结
  - 主模型：L2 逻辑回归（预注册，不许中途替换）
  - HGB 本版不做（一周工期）；Ridge 只用来给 R4 提供回归预测
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from src.b3 import fold_frames
from src.backtest import run_positions
from src.config import load_config, project_path, set_seed
from src.decision import rule_fixed, rule_kelly, rule_quantile
from src.evaluate import (
    classification_metrics,
    paired_bootstrap_diff,
    regression_metrics,
    strategy_metrics,
)
from src.features import feature_columns, make_preprocessor


def _cash_return_series(df: pd.DataFrame) -> pd.Series:
    """将 DTB3 贴现率转为债券等价收益率，并按当周实际交易日数折算。"""
    dtb3 = pd.Series(df["dtb3"], index=df.index, dtype=float).fillna(0.0) / 100.0
    bey = (365.0 * dtb3) / (360.0 - 91.0 * dtb3)
    if "holding_trading_days" in df.columns:
        days = pd.Series(df["holding_trading_days"], index=df.index, dtype=float).fillna(5.0)
    elif "n_trading_days" in df.columns:
        days = pd.Series(df["n_trading_days"], index=df.index, dtype=float).fillna(5.0)
    else:
        days = pd.Series(5.0, index=df.index, dtype=float)
    return bey * (days / 252.0)


def _baseline_positions(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """生成 4 条基线的目标仓位序列：买入持有、始终做多、恒定 60%、SMA200。"""
    idx = panel.index
    ones = pd.Series(1.0, index=idx, dtype=float)
    const_60 = pd.Series(0.6, index=idx, dtype=float)
    if "px_ma200" in panel.columns:
        sma200 = (pd.Series(panel["px_ma200"], index=idx, dtype=float) > 0.0).astype(float)
    else:
        ma200 = panel["close"].rolling(200).mean()
        sma200 = (panel["close"] > ma200).astype(float)
    return {
        "buy_and_hold": ones,
        "always_long": ones.copy(),
        "const_60": const_60,
        "sma200": sma200,
    }


def make_baselines(panel: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """买入持有、恒定 60% 仓位、SMA200、始终做多 —— 必须与最终模型同口径同成本。"""
    cfg = load_config() if cfg is None else cfg
    df = panel.set_index("date") if "date" in panel.columns else panel.copy()
    if "label_available" in df.columns:
        df = df.loc[df["label_available"].astype(bool)]
    df = df.dropna(subset=["next_ret"])

    bt_cfg = cfg["backtest"]
    rf = _cash_return_series(df)
    pos_map = _baseline_positions(df)

    rows = []
    for name, pos in pos_map.items():
        bt = run_positions(
            positions=pos,
            asset_returns=df["next_ret"],
            cash_returns=rf,
            cost_bps=float(bt_cfg["cost_bps"]),
            initial_position=float(bt_cfg["initial_position"]),
            initial_equity=float(bt_cfg["initial_equity"]),
        )
        m = strategy_metrics(bt["equity"], bt, freq=52, rf_returns=rf)
        rows.append({"strategy": name, **m})
    return pd.DataFrame(rows)


def build_model(name: str, seed: int = 42, **kwargs):
    """返回未训练的 sklearn 模型（logreg / ridge / rf）。"""
    key = name.lower()
    if key == "logreg":
        return LogisticRegression(
            C=float(kwargs.get("C", 0.1)),
            max_iter=1000,
            random_state=seed,
        )
    if key == "ridge":
        return Ridge(
            alpha=float(kwargs.get("alpha", 10.0)),
            random_state=seed,
        )
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=int(kwargs.get("n_estimators", 200)),
            max_depth=kwargs.get("max_depth", 5),
            min_samples_leaf=int(kwargs.get("min_samples_leaf", 10)),
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"不支持的模型: {name}（本版仅支持 logreg / ridge / rf）")


def _count_params(model) -> int | None:
    if hasattr(model, "coef_"):
        n = int(np.asarray(model.coef_).size)
        if hasattr(model, "intercept_"):
            n += int(np.asarray(model.intercept_).size)
        return n
    return None


def select_hyperparams(
    panel: pd.DataFrame,
    val_fold: dict,
    cfg: dict,
) -> tuple[dict, list[dict]]:
    """只在 2006–2007 验证集上一次性选参并冻结。"""
    seed = int(cfg["project"]["seed"])
    bt_cfg = cfg["backtest"]
    feat_cols = feature_columns()

    train, val = fold_frames(panel, val_fold)
    pre = make_preprocessor()
    X_train = pre.fit_transform(train[feat_cols])
    X_val = pre.transform(val[feat_cols])
    y_train = train["label_up"].astype(int).to_numpy()
    y_val = val["label_up"].astype(int).to_numpy()
    r_train = train["label_ret"].astype(float).to_numpy()
    r_val = val["label_ret"].astype(float).to_numpy()

    today_str = date.today().isoformat()
    records: list[dict] = []

    # 1. L2 逻辑回归：选 C
    c_grid = [0.01, 0.1, 1.0, 10.0]
    best_c, best_auc = c_grid[0], -np.inf
    for c_val in c_grid:
        clf = build_model("logreg", seed=seed, C=c_val)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, proba))
        if auc > best_auc:
            best_auc, best_c = auc, c_val
    records.append(
        {
            "model": "logreg",
            "param": "C（L2 正则强度）",
            "grid": "0.01 / 0.1 / 1.0 / 10.0",
            "selected": str(best_c),
            "metric": f"Val AUC = {best_auc:.4f}",
            "date": today_str,
        }
    )
    records.append(
        {
            "model": "logreg",
            "param": "R1 阈值",
            "grid": "预注册固定 0.50",
            "selected": "0.50",
            "metric": "预注册主规则（不看测试集）",
            "date": today_str,
        }
    )

    # 2. Ridge 回归：选 alpha
    alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha, best_rmse = alpha_grid[0], np.inf
    best_mu_val = None
    for a_val in alpha_grid:
        reg = build_model("ridge", seed=seed, alpha=a_val)
        reg.fit(X_train, r_train)
        pred = reg.predict(X_val)
        rmse = float(root_mean_squared_error(r_val, pred))
        if rmse < best_rmse:
            best_rmse, best_alpha, best_mu_val = rmse, a_val, pred
    records.append(
        {
            "model": "ridge",
            "param": "alpha（L2 正则强度）",
            "grid": "0.1 / 1 / 10 / 100 / 1000",
            "selected": str(best_alpha),
            "metric": f"Val RMSE = {best_rmse:.5f}",
            "date": today_str,
        }
    )

    # 3. R4 Kelly 缩放系数 kappa：在验证集按 5 bps 净 Sharpe 选定一次
    kappa_grid = [0.05, 0.1, 0.25, 0.5, 1.0]
    sigma_val = (val["vol_20"].astype(float) / np.sqrt(52.0)).clip(lower=1e-4)
    mu_val_s = pd.Series(best_mu_val, index=val.index, dtype=float)
    rf_val = _cash_return_series(val)
    best_kappa, best_sharpe = kappa_grid[0], -np.inf
    for k_val in kappa_grid:
        pos_val = rule_kelly(mu_val_s, sigma_val, kappa=k_val)
        bt_val = run_positions(
            positions=pos_val,
            asset_returns=val["next_ret"],
            cash_returns=rf_val,
            cost_bps=float(bt_cfg["cost_bps"]),
            initial_position=float(bt_cfg["initial_position"]),
            initial_equity=float(bt_cfg["initial_equity"]),
        )
        m_val = strategy_metrics(bt_val["equity"], bt_val, freq=52, rf_returns=rf_val)
        sh = m_val["sharpe"] if np.isfinite(m_val["sharpe"]) else -np.inf
        if sh > best_sharpe:
            best_sharpe, best_kappa = sh, k_val
    records.append(
        {
            "model": "ridge (R4)",
            "param": "kappa（Kelly 缩放系数）",
            "grid": "0.05 / 0.1 / 0.25 / 0.5 / 1.0",
            "selected": str(best_kappa),
            "metric": f"Val Net Sharpe = {best_sharpe:.4f}",
            "date": today_str,
        }
    )

    # 4. 随机森林：选 max_depth
    depth_grid = [3, 5, 8]
    best_depth, best_rf_auc = depth_grid[0], -np.inf
    for d_val in depth_grid:
        rf_clf = build_model("rf", seed=seed, max_depth=d_val)
        rf_clf.fit(X_train, y_train)
        proba = rf_clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, proba))
        if auc > best_rf_auc:
            best_rf_auc, best_depth = auc, d_val
    records.append(
        {
            "model": "rf",
            "param": "max_depth（树深）",
            "grid": "3 / 5 / 8（n_estimators=200, min_samples_leaf=10）",
            "selected": str(best_depth),
            "metric": f"Val AUC = {best_rf_auc:.4f}",
            "date": today_str,
        }
    )

    frozen = {
        "logreg": {"C": best_c},
        "ridge": {"alpha": best_alpha, "kappa": best_kappa},
        "rf": {"max_depth": best_depth, "n_estimators": 200, "min_samples_leaf": 10},
    }
    return frozen, records


def write_hyperparams_doc(records: list[dict], path) -> None:
    """将一次性选定的冻结超参写入 docs/hyperparams.md。"""
    lines = [
        "# 冻结超参数记录表（B4 · 传统机器学习）",
        "",
        "> 纪律：所有超参数与缩放系数仅在 `2006–2007` 验证集（`validation_split.json`，训练 468 周 / 验证 104 周）上选定一次并冻结，`2008–2025` 共 18 折滚动测试全程不再修改。",
        "",
        "| 模型 | 参数 | 搜索范围 | 选中值 | 验证集指标 | 选定日期 |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['model']} | {r['param']} | {r['grid']} | {r['selected']} | {r['metric']} | {r['date']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def walk_forward(
    panel: pd.DataFrame,
    folds: list[dict],
    model_name: str,
    cfg: dict,
    params: dict | None = None,
    perf_out: dict | None = None,
) -> pd.DataFrame:
    """滚动训练+预测，返回 predictions 表（每折独立预处理，严格防泄露）。"""
    seed = int(cfg["project"]["seed"])
    feat_cols = feature_columns()
    model_key = model_name.lower()
    model_params = dict(params or {})
    kappa = float(model_params.pop("kappa", 0.25))

    frames: list[pd.DataFrame] = []
    fit_times: list[float] = []
    pred_times: list[float] = []
    peak_mems_mb: list[float] = []
    total_pred_rows = 0
    param_count = None

    for fold in folds:
        train, test = fold_frames(panel, fold)
        pre = make_preprocessor()
        X_train = pre.fit_transform(train[feat_cols])
        X_test = pre.transform(test[feat_cols])

        model = build_model(model_key, seed=seed, **model_params)

        tracemalloc.start()
        t0 = time.perf_counter()
        if model_key == "ridge":
            model.fit(X_train, train["label_ret"].astype(float).to_numpy())
        else:
            model.fit(X_train, train["label_up"].astype(int).to_numpy())
        t1 = time.perf_counter()

        if model_key == "ridge":
            y_pred_vals = model.predict(X_test)
        else:
            y_pred_vals = model.predict_proba(X_test)[:, 1]
        t2 = time.perf_counter()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        fit_times.append(t1 - t0)
        pred_times.append((t2 - t1) * 1000.0)
        peak_mems_mb.append(peak_bytes / (1024.0 * 1024.0))
        total_pred_rows += len(test)
        if param_count is None:
            param_count = _count_params(model)

        pred_s = pd.Series(y_pred_vals, index=test.index, dtype=float)
        if model_key == "ridge":
            sigma_test = (test["vol_20"].astype(float) / np.sqrt(52.0)).clip(lower=1e-4)
            pos_main = rule_kelly(pred_s, sigma_test, kappa=kappa)
            pos_r2 = pd.Series(np.nan, index=test.index, dtype=float)
        else:
            train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=train.index, dtype=float)
            pos_main = rule_fixed(pred_s, threshold=0.5)
            pos_r2 = rule_quantile(pred_s, train_proba, q=0.60)

        out = pd.DataFrame(
            {
                "model": model_key,
                "fold": int(fold["fold"]),
                "decision_date": test.index,
                "execution_date": test["execution_date"].to_numpy(),
                "exit_date": test["exit_date"].to_numpy(),
                "y_true": test["label_up"].astype(int).to_numpy(),
                "y_pred": pred_s.to_numpy(),
                "position": pos_main.to_numpy(),
                "position_r2": pos_r2.to_numpy(),
                "next_ret": test["next_ret"].astype(float).to_numpy(),
                "label_ret": test["label_ret"].astype(float).to_numpy(),
                "holding_trading_days": test["holding_trading_days"].astype(float).to_numpy(),
                "dtb3": test["dtb3"].astype(float).to_numpy(),
            }
        )
        frames.append(out)

    if perf_out is not None and fit_times:
        total_pred_s = sum(pred_times) / 1000.0
        perf_out.update(
            {
                "model": model_key,
                "train_sec_per_fold": round(float(np.mean(fit_times)), 4),
                "predict_ms_per_fold": round(float(np.mean(pred_times)), 4),
                "throughput_rows_per_sec": round(
                    total_pred_rows / total_pred_s if total_pred_s > 0 else np.nan, 1
                ),
                "peak_memory_mb": round(float(np.max(peak_mems_mb)), 3),
                "n_params": param_count if param_count is not None else "—",
                "device": "CPU",
            }
        )

    return pd.concat(frames, ignore_index=True)


def run(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """B4 一键入口：一次性选参冻结 → 18 折滚动 → 基线与指标对比 → 写出交付文件。"""
    cfg = load_config() if cfg is None else cfg
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    bt_cfg = cfg["backtest"]

    panel = pd.read_parquet(project_path("data/processed/model_panel.parquet"))
    folds = json.loads(project_path(cfg["data"]["splits"]).read_text(encoding="utf-8"))
    val_folds = json.loads(
        project_path("data/processed/validation_split.json").read_text(encoding="utf-8")
    )
    val_fold = val_folds[0] if isinstance(val_folds, list) else val_folds

    # 1. 2006–2007 一次性选参并写入 docs/hyperparams.md
    frozen_params, hp_records = select_hyperparams(panel, val_fold, cfg)
    hp_path = project_path("docs/hyperparams.md")
    write_hyperparams_doc(hp_records, hp_path)

    # 2. 18 折滚动预测（logreg / ridge / rf）并记录耗时与内存
    perf_rows: list[dict] = []
    pred_dict: dict[str, pd.DataFrame] = {}
    for m_name in ("logreg", "ridge", "rf"):
        perf_info: dict = {}
        pred_df = walk_forward(
            panel=panel,
            folds=folds,
            model_name=m_name,
            cfg=cfg,
            params=frozen_params[m_name],
            perf_out=perf_info,
        )
        pred_dict[m_name] = pred_df
        perf_rows.append(perf_info)

    all_preds = pd.concat(pred_dict.values(), ignore_index=True)
    pred_path = project_path(cfg["data"]["predictions"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    all_preds.to_parquet(pred_path, index=False)

    perf_df = pd.DataFrame(perf_rows)
    perf_path = project_path("reports/perf.csv")
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    perf_df.to_csv(perf_path, index=False)

    # 3. 同口径回测：3 种决策规则 + RF + 4 条基线
    oos_dates = pd.DatetimeIndex(pred_dict["logreg"]["decision_date"])
    indexed_panel = panel.set_index("date") if "date" in panel.columns else panel
    oos_panel = indexed_panel.loc[oos_dates].copy()
    rf_oos = _cash_return_series(oos_panel)
    asset_oos = oos_panel["next_ret"].astype(float)

    strategy_positions: dict[str, pd.Series] = {
        "logreg_R1": pd.Series(pred_dict["logreg"]["position"].to_numpy(), index=oos_dates),
        "logreg_R2": pd.Series(pred_dict["logreg"]["position_r2"].to_numpy(), index=oos_dates),
        "ridge_R4": pd.Series(pred_dict["ridge"]["position"].to_numpy(), index=oos_dates),
        "rf_R1": pd.Series(pred_dict["rf"]["position"].to_numpy(), index=oos_dates),
        **_baseline_positions(oos_panel),
    }

    bt_results: dict[str, pd.DataFrame] = {}
    excess_returns: dict[str, pd.Series] = {}
    for s_name, pos in strategy_positions.items():
        bt = run_positions(
            positions=pos,
            asset_returns=asset_oos,
            cash_returns=rf_oos,
            cost_bps=float(bt_cfg["cost_bps"]),
            initial_position=float(bt_cfg["initial_position"]),
            initial_equity=float(bt_cfg["initial_equity"]),
        )
        bt_results[s_name] = bt
        excess_returns[s_name] = bt["net_return"] - rf_oos

    # 4. 汇总预测层 + 策略层 + 配对 Bootstrap 95% CI
    lr_cls = classification_metrics(
        pred_dict["logreg"]["y_true"],
        (pred_dict["logreg"]["position"] > 0.5).astype(int),
        pred_dict["logreg"]["y_pred"],
    )
    rf_cls = classification_metrics(
        pred_dict["rf"]["y_true"],
        (pred_dict["rf"]["position"] > 0.5).astype(int),
        pred_dict["rf"]["y_pred"],
    )
    ridge_reg = regression_metrics(
        pred_dict["ridge"]["label_ret"],
        pred_dict["ridge"]["y_pred"],
    )

    result_rows: list[dict] = []
    bh_excess = excess_returns["buy_and_hold"]
    c60_excess = excess_returns["const_60"]

    for s_name, bt in bt_results.items():
        sm = strategy_metrics(bt["equity"], bt, freq=52, rf_returns=rf_oos)
        row: dict = {"strategy": s_name, **sm}

        if s_name in ("logreg_R1", "logreg_R2"):
            row.update(
                {
                    "auc": lr_cls["auc"],
                    "balanced_accuracy": lr_cls["balanced_accuracy"],
                    "f1": lr_cls["f1"],
                    "brier": lr_cls["brier"],
                }
            )
        elif s_name == "rf_R1":
            row.update(
                {
                    "auc": rf_cls["auc"],
                    "balanced_accuracy": rf_cls["balanced_accuracy"],
                    "f1": rf_cls["f1"],
                    "brier": rf_cls["brier"],
                }
            )
        elif s_name == "always_long":
            row.update(
                {
                    "auc": lr_cls["always_long_auc"],
                    "balanced_accuracy": lr_cls["always_long_balanced_accuracy"],
                    "f1": lr_cls["always_long_f1"],
                    "brier": lr_cls["always_long_brier"],
                }
            )
        elif s_name == "ridge_R4":
            row.update(ridge_reg)

        if s_name != "buy_and_hold":
            lo_bh, hi_bh = paired_bootstrap_diff(
                excess_returns[s_name], bh_excess, block=8, n_boot=5000, seed=seed
            )
            row["sharpe_diff_vs_bh"] = sm["sharpe"] - strategy_metrics(
                bt_results["buy_and_hold"]["equity"],
                bt_results["buy_and_hold"],
                freq=52,
                rf_returns=rf_oos,
            )["sharpe"]
            row["sharpe_diff_vs_bh_ci_low"] = lo_bh
            row["sharpe_diff_vs_bh_ci_high"] = hi_bh

        if s_name != "const_60":
            lo_60, hi_60 = paired_bootstrap_diff(
                excess_returns[s_name], c60_excess, block=8, n_boot=5000, seed=seed
            )
            row["sharpe_diff_vs_c60"] = sm["sharpe"] - strategy_metrics(
                bt_results["const_60"]["equity"],
                bt_results["const_60"],
                freq=52,
                rf_returns=rf_oos,
            )["sharpe"]
            row["sharpe_diff_vs_c60_ci_low"] = lo_60
            row["sharpe_diff_vs_c60_ci_high"] = hi_60

        result_rows.append(row)

    results_df = pd.DataFrame(result_rows)
    res_path = project_path("reports/model_results.csv")
    results_df.to_csv(res_path, index=False)

    print(f"B4 完成：样本外 {len(oos_dates)} 周（2008–2025），已生成：")
    print(f"  - {pred_path}")
    print(f"  - {res_path}")
    print(f"  - {perf_path}")
    print(f"  - {hp_path}")
    print("\n=== 六线主对比 + RF（5 bps 单边成本）===")
    cols = ["strategy", "cagr", "ann_vol", "sharpe", "max_dd_weekly", "ann_turnover", "avg_position"]
    print(results_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 训练与推理性能（CPU）===")
    print(perf_df.to_string(index=False))

    return {"predictions": all_preds, "results": results_df, "perf": perf_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="B4 传统机器学习建模与评测入口")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()