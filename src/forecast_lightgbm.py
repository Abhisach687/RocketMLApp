"""
Train LightGBM forecast model (progress bar, no warnings).
Logs MAE + wMAPE and checks the ≤ 10 % gate.
"""
from __future__ import annotations
import warnings, yaml, joblib, lightgbm as lgb
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return np.abs(y_true - y_pred).sum() / denom if denom else np.nan


def _load_cfg(path="config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf‑8") as f:
        return yaml.safe_load(f)


class TQDMCallback:
    def __init__(self, total_rounds: int):
        self.pbar = tqdm(total=total_rounds, unit="iter", desc="Training", leave=False)

    def __call__(self, env):
        self.pbar.update(1)
        if env.iteration + 1 == self.pbar.total:
            self.pbar.close()


# ──────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, float]:
    fcfg       = cfg["models"]["forecast"]
    val_days   = cfg["models"]["val_split_days"]
    drop_zero  = fcfg["drop_zero_target"]
    use_weight = fcfg["use_sample_weight"]

    df = df[df["target_next7"].notnull()]
    if drop_zero:
        df = df[df["target_next7"] > 0]

    df["date"]   = pd.to_datetime(df["date"])
    split_date   = df["date"].max() - pd.Timedelta(days=val_days)
    train_df     = df[df["date"] < split_date]
    val_df       = df[df["date"] >= split_date]
    feats        = [c for c in df.columns if c not in ("itemid", "date", "target_next7")]

    X_train, y_train = train_df[feats], train_df["target_next7"]
    X_val,   y_val   = val_df[feats],   val_df["target_next7"]
    weights          = (train_df["sales_sum_7d"].fillna(0.1).clip(lower=0.1)
                        if use_weight else None)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
    dval   = lgb.Dataset(X_val,   label=y_val)

    params = {
        "objective":         fcfg["objective"],
        "metric":            fcfg["metric"],
        "learning_rate":     fcfg["learning_rate"],
        "num_leaves":        fcfg["num_leaves"],
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "verbosity":         -1,
        "seed":              42,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=fcfg["num_boost_round"],
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(fcfg["early_stopping_rounds"], verbose=False),
                TQDMCallback(fcfg["num_boost_round"]),
            ],
        )

    y_hat   = model.predict(X_val, num_iteration=model.best_iteration)
    metrics = {
        "val_mae":   float(mean_absolute_error(y_val, y_hat)),
        "val_wmape": float(wmape(y_val.values, y_hat)),
        "best_iter": int(model.best_iteration),
    }
    metrics["pass_wmape_gate"] = metrics["val_wmape"] <= 0.10

    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics_forecast_final.yaml", "w") as f:
        yaml.dump(metrics, f)

    joblib.dump(model, fcfg["weighted_model_path"])
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg   = _load_cfg()
    feats = Path(cfg["features"]["out_dir"]) / cfg["features"]["processed_forecast_path"]
    res   = train(pd.read_parquet(feats), cfg)
    print("Validation wMAPE:", res["val_wmape"], "| Pass Gate ≤10%:", res["pass_wmape_gate"])
