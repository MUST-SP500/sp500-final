"""深度学习（B5）：DNN / 1D CNN / Autoencoder。

负责人：P5（B5）
对应：SP500_分工说明书 P5 第 2–5 步

约束：
  - 早停只用本折训练段尾部 2 年，绝不用测试年
  - 全折只用 1 个固定种子（一周工期，不做多种子方差；真的需要再单独说）
  - 参数量 <= 5,000（样本只有约 1,500 周，模型不能大）
  - 单次训练 > 5 分钟的超参组合直接移出搜索空间
  - 滚动窗口的最后一个观测值必须 <= 决策时点 t，且不跨 train/test 边界

用法：
    uv run python -m src.models_dl                    # DNN，18 折，落 predictions_dl.parquet（主交付物）
    uv run python -m src.models_dl --kind cnn         # 换 CNN，另存 predictions_dl_cnn_standard.parquet
    uv run python -m src.models_dl --scaling minmax   # 缩放消融，另存 predictions_dl_cnn_minmax.parquet
    uv run python -m src.models_dl --kind ae          # Autoencoder 单折对比（不落盘）
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import load_config, project_path
from src.features import feature_columns, make_preprocessor

if TYPE_CHECKING:  # 只为类型标注，运行时不导入（torch 是可选依赖）
    import torch
    from torch import nn

# 训练口径集中在常量里，改这里就等于改全项目
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 10           # 验证损失连续 10 轮不降就停
VALIDATION_YEARS = 2    # 早停验证段 = 训练段尾部 2 年
PURGE_WEEKS = 2         # 核心训练段与验证段之间空 2 周
THRESHOLD = 0.5         # R1 固定阈值，用于填 position 列


def _nn():
    """用到深度学习时才导入 torch：没装 dl 组也能 import 本模块（冒烟测试要求）。"""
    import torch
    from torch import nn

    return torch, nn


def build_dnn(n_features: int, seed: int = 42) -> nn.Module:
    """22 → 64 → 32 → 1；Dropout 0.3 + BatchNorm，参数量约 3,700（远小于 5,000）。"""
    torch, nn = _nn()
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(n_features, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
    )


def build_cnn(window: int = 20, n_features: int = 22, seed: int = 42) -> nn.Module:
    """把"20 周 × 22 指标"当一张小图：Conv1D(16) → Conv1D(32) → 全局最大池化 → 1。

    输入形状必须是 (样本, 特征, 周)，由 _to_model_input 负责转置。
    """
    torch, nn = _nn()
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv1d(n_features, 16, kernel_size=3, padding=1),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.AdaptiveMaxPool1d(1),
        nn.Flatten(),
        nn.Linear(32, 1),
    )


def train_and_predict(model: nn.Module, X_train, y_train, X_val, y_val, X_test, epochs: int = 100, seed: int = 42):
    """训练 + 早停（只看验证段）+ 返回测试集的上涨概率。

    损失曲线存在 model.loss_history_，供 B5 第 7 步画学习曲线。
    进函数先固定随机流：同一份输入、同一个 seed，无论之前跑过什么，结果都一样。
    """
    torch, nn = _nn()
    torch.manual_seed(seed)
    X_train_t, y_train_t = _to_tensors(X_train, y_train)
    X_val_t, y_val_t = _to_tensors(X_val, y_val)
    X_test_t = _to_tensor(X_test)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss, best_state, wait = float("inf"), None, 0
    history = {"train": [], "val": []}

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X_train_t))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            if len(batch) < 2:  # BatchNorm 不接受单样本批
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t[batch]), y_train_t[batch])
            loss.backward()
            optimizer.step()

        train_loss = _mean_loss(model, loss_fn, X_train_t, y_train_t)
        val_loss = _mean_loss(model, loss_fn, X_val_t, y_val_t)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss, wait = val_loss, 0
            best_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.loss_history_ = history

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_test_t)).view(-1).numpy()


def make_windows(features: pd.DataFrame, window: int = 20):
    """(周 × 特征) → (样本 × window × 特征)，每个样本的最后一行就是决策周。

    只往后看的历史，绝不含未来行；返回 (windows, decision_dates, is_complete)。
    is_complete=False 表示窗口里还有缺失值（例如 1996 年的暖机期），训练前要丢掉。
    """
    values = features.to_numpy(dtype=float)
    n_samples = len(values) - window + 1
    if n_samples <= 0:
        raise ValueError(f"样本只有 {len(values)} 行，凑不出 {window} 周窗口")

    windows = np.stack([values[i : i + window] for i in range(n_samples)])
    decision_dates = features.index[window - 1 :]
    is_complete = ~np.isnan(windows).any(axis=(1, 2))
    return windows, decision_dates, is_complete


def autoencoder_bottleneck(X_train, X_val, bottleneck: int = 8, epochs: int = 200, seed: int = 42):
    """22 → 16 → bottleneck → 16 → 22，只重构训练段。

    返回 (encoder, 训练集瓶颈特征, 验证集瓶颈特征)：拿那几维瓶颈喂逻辑回归，
    就能和 PCA 做"线性 vs 非线性"的对比。说明书里只要求跑 2008 一折。
    """
    torch, nn = _nn()
    torch.manual_seed(seed)
    n_features = np.asarray(X_train).shape[1]
    encoder = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, bottleneck), nn.ReLU())
    decoder = nn.Sequential(nn.Linear(bottleneck, 16), nn.ReLU(), nn.Linear(16, n_features))
    autoencoder = nn.Sequential(encoder, decoder)

    X_train_t = _to_tensor(X_train)
    X_val_t = _to_tensor(X_val)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    best_val_loss, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        autoencoder.train()
        optimizer.zero_grad()
        loss = loss_fn(autoencoder(X_train_t), X_train_t)
        loss.backward()
        optimizer.step()

        val_loss = _mean_loss(autoencoder, loss_fn, X_val_t, X_val_t)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, wait = val_loss, 0
            best_state = {name: tensor.clone() for name, tensor in autoencoder.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE * 2:  # 重构任务更平缓，耐心翻倍
                break

    if best_state is not None:
        autoencoder.load_state_dict(best_state)

    encoder.eval()
    with torch.no_grad():
        z_train = encoder(X_train_t).numpy()
        z_val = encoder(X_val_t).numpy()
    return encoder, z_train, z_val


def autoencoder_vs_pca(panel: pd.DataFrame, fold: dict, cfg: dict, bottleneck: int = 6) -> pd.DataFrame:
    """单折对比（说明书只要求 2008 折）：AE 瓶颈 / PCA / 全特征 三种输入喂同一个逻辑回归。

    返回一张小表，回答"非线性压缩比线性 PCA 好在哪"。
    """
    from sklearn.linear_model import LogisticRegression

    from src.models_pca import fit_pca, strategy_metrics

    feature_cols = feature_columns()
    seed = int(cfg["project"]["seed"])
    train_rows, test_rows = _fold_rows(panel, fold, feature_cols)
    core_rows, val_rows = _split_validation(train_rows)

    preprocessor = make_preprocessor()
    preprocessor.fit(core_rows[feature_cols])
    X_core = preprocessor.transform(core_rows[feature_cols])
    X_val = preprocessor.transform(val_rows[feature_cols])
    X_test = preprocessor.transform(test_rows[feature_cols])
    y_core = core_rows["label_up"]
    y_test = test_rows["label_up"]

    encoder, z_core, _ = autoencoder_bottleneck(X_core, X_val, bottleneck=bottleneck, seed=seed)
    z_test = encoder(_to_tensor(X_test)).detach().numpy()
    pca_model, pca_dims = fit_pca(X_core)

    variants = {
        f"autoencoder_{bottleneck}d": (z_core, z_test),
        f"pca_{pca_dims}d": (pca_model.transform(X_core), pca_model.transform(X_test)),
        f"full_{len(feature_cols)}d": (X_core, X_test),
    }

    rows = []
    next_ret = test_rows["next_ret"].to_numpy() if "next_ret" in test_rows.columns else np.full(len(test_rows), np.nan)
    for name, (X_fit, X_pred) in variants.items():
        classifier = LogisticRegression(max_iter=1000, random_state=seed).fit(X_fit, y_core)
        probability = classifier.predict_proba(X_pred)[:, 1]
        rows.append(
            {
                "input": name,
                "n_features": int(X_fit.shape[1]),
                "auc": roc_auc_score(y_test, probability),
                **strategy_metrics((probability >= THRESHOLD).astype(int), next_ret),
                "fold": fold["fold"],
            }
        )
    return pd.DataFrame(rows)


def run_dl_folds(
    panel: pd.DataFrame,
    folds: list[dict],
    cfg: dict,
    model_kind: str = "dnn",
    window: int = 20,
    epochs: int = 100,
    scaling: str = "standard",
) -> pd.DataFrame:
    """18 折滚动训练 DNN 或 CNN，返回与 B4 同列结构的预测表。

    每折的流程固定：训练段切出"尾部 2 年"做早停验证 → 预处理只在核心训练段 fit
    → 训练 → 预测该测试年。测试年一行都不进训练，20 周窗口也不跨段。
    """
    feature_cols = feature_columns()
    seed = int(cfg["project"]["seed"])
    model_name = f"{model_kind}_{scaling}"
    all_predictions = []

    for fold in folds:
        train_rows, test_rows = _fold_rows(panel, fold, feature_cols)
        if train_rows.empty or test_rows.empty:
            continue
        core_rows, val_rows = _split_validation(train_rows)
        if core_rows.empty or val_rows.empty:
            continue
        # CNN 要 window 周才凑一个样本，任一段太短就整折跳过（宁缺不造）
        if model_kind == "cnn" and min(len(core_rows), len(val_rows), len(test_rows)) < window:
            continue

        # 预处理只在核心训练段 fit，验证段与测试段只能 transform
        preprocessor = make_preprocessor(scaling=scaling)
        preprocessor.fit(core_rows[feature_cols])

        X_core, _, y_core = _to_model_input(core_rows, preprocessor, feature_cols, model_kind, window)
        X_val, _, y_val = _to_model_input(val_rows, preprocessor, feature_cols, model_kind, window)
        X_test, test_dates, y_test = _to_model_input(test_rows, preprocessor, feature_cols, model_kind, window)
        if min(len(X_core), len(X_val), len(X_test)) == 0:  # 窗口全含缺失时一段都凑不出来
            continue

        model = build_dnn(len(feature_cols), seed) if model_kind == "dnn" else build_cnn(window, len(feature_cols), seed)
        probability = train_and_predict(model, X_core, y_core, X_val, y_val, X_test, epochs=epochs, seed=seed)

        all_predictions.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "fold": fold["fold"],
                    "decision_date": test_dates,
                    "execution_date": _execution_dates(test_rows, test_dates),
                    "y_true": y_test,
                    "y_pred": probability,
                    "position": (probability >= THRESHOLD).astype(int),
                }
            )
        )

    return pd.concat(all_predictions, ignore_index=True)


def _to_model_input(rows: pd.DataFrame, preprocessor, feature_cols: list[str], model_kind: str, window: int):
    """把一段行变成模型输入。DNN 是 (样本, 22)；CNN 是 (样本, 22, 20)。"""
    scaled = pd.DataFrame(preprocessor.transform(rows[feature_cols]), index=rows.index, columns=feature_cols)
    if model_kind == "dnn":
        return scaled.to_numpy(), rows.index, rows["label_up"].to_numpy()

    windows, decision_dates, is_complete = make_windows(scaled, window)
    labels = rows.loc[decision_dates, "label_up"].to_numpy()
    windows, decision_dates, labels = windows[is_complete], decision_dates[is_complete], labels[is_complete]
    return windows.transpose(0, 2, 1), decision_dates, labels


def _split_validation(train_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练段尾部 2 年做早停验证，并在中间留 2 周 purge，防止标签跨窗重叠。"""
    tail_start = train_rows.index.max() - pd.DateOffset(years=VALIDATION_YEARS)
    val_rows = train_rows[train_rows.index >= tail_start]
    core_rows = train_rows[train_rows.index < tail_start]
    if core_rows.empty:
        return train_rows.iloc[0:0], train_rows

    purge_cutoff = core_rows.index.max() - pd.DateOffset(weeks=PURGE_WEEKS)
    return core_rows[core_rows.index <= purge_cutoff], val_rows


def _fold_rows(panel: pd.DataFrame, fold: dict, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按折取训练/测试行，丢掉特征不全或没有标签的行。"""
    needed = [*feature_cols, "label_up"]
    train_rows = panel.reindex(pd.DatetimeIndex(fold["train_dates"])).dropna(subset=needed)
    test_rows = panel.reindex(pd.DatetimeIndex(fold["test_dates"])).dropna(subset=needed)
    return train_rows, test_rows


def _execution_dates(test_rows: pd.DataFrame, decision_dates) -> pd.Series:
    """执行日：面板里有就取对应行，没有就整列 NaT（保持 datetime 类型）。"""
    if "execution_date" in test_rows.columns:
        return test_rows.loc[decision_dates, "execution_date"]
    return pd.Series(pd.NaT, index=pd.DatetimeIndex(decision_dates))


def _output_path(kind: str, scaling: str) -> str:
    """主口径（DNN + 标准分）固定写 predictions_dl.parquet，消融档位带后缀另存。"""
    if (kind, scaling) == ("dnn", "standard"):
        name = "predictions_dl.parquet"
    else:
        name = f"predictions_dl_{kind}_{scaling}.parquet"
    return str(project_path("data", "processed", name))


def _to_tensor(array) -> torch.Tensor:
    torch, _ = _nn()
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def _to_tensors(features, labels) -> tuple[torch.Tensor, torch.Tensor]:
    torch, _ = _nn()
    return _to_tensor(features), torch.as_tensor(np.asarray(labels), dtype=torch.float32).view(-1, 1)


def _mean_loss(model: nn.Module, loss_fn, features: torch.Tensor, labels: torch.Tensor) -> float:
    """整段数据上的平均损失（评估用，不参与梯度）。"""
    torch, _ = _nn()
    if len(features) == 0:
        return float("inf")
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(features), labels).item())


def _load_inputs():
    """读面板与切分，给命令行用。"""
    cfg = load_config()
    panel = pd.read_parquet(project_path(cfg["data"]["processed_dir"], "model_panel.parquet"))
    panel = panel.assign(date=pd.to_datetime(panel["date"])).set_index("date").sort_index()
    with open(project_path(cfg["data"]["splits"]), encoding="utf-8") as f:
        folds = json.load(f)
    folds = folds.get("folds", folds) if isinstance(folds, dict) else folds
    return panel, folds, cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B5 深度学习滚动预测")
    parser.add_argument("--kind", default="dnn", choices=["dnn", "cnn", "ae"])
    parser.add_argument("--scaling", default="standard", choices=["standard", "minmax"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    panel, folds, cfg = _load_inputs()

    if args.kind == "ae":  # 只跑 2008 一折，和 PCA 做定性对比
        table = autoencoder_vs_pca(panel, folds[0], cfg)
        print(table.to_string(index=False))
        return 0

    predictions = run_dl_folds(panel, folds, cfg, model_kind=args.kind, epochs=args.epochs, scaling=args.scaling)

    for name, part in predictions.groupby("model"):
        print(f"{name}: {len(part)} 行 | AUC {roc_auc_score(part['y_true'], part['y_pred']):.4f}")

    if not args.no_save:
        out_path = _output_path(args.kind, args.scaling)
        predictions.to_parquet(out_path, index=False)
        print(f"已写出 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
