"""
Optuna tuning for LightGBM forecast model with TQDM progress bar.
• Wide search space   • Early stopping callback   • Config auto‑update
"""
from __future__ import annotations
import warnings, yaml, optuna, lightgbm as lgb, joblib
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


def _save_cfg(cfg: Dict[str, Any], path="config.yaml") -> None:
    with open(path, "w", encoding="utf‑8") as f:
        yaml.dump(cfg, f)


# ──────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial,
              train_df: pd.DataFrame,
              val_df:   pd.DataFrame,
              feats:    list[str],
              use_weight: bool) -> float:

    weights = (train_df["sales_sum_7d"].fillna(0.1).clip(lower=0.1)
               if use_weight else None)

    dtrain = lgb.Dataset(train_df[feats], label=train_df["target_next7"], weight=weights)
    dval   = lgb.Dataset(val_df[feats],   label=val_df["target_next7"])

    params = {
        "objective":        "regression",
        "metric":           "mae",
        "boosting_type":    "gbdt",
        "verbosity":        -1,
        "seed":             42,
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 8, 256, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 150),
        "lambda_l1":        trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2":        trial.suggest_float("lambda_l2", 0.0, 10.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 15),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

    pred = model.predict(val_df[feats], num_iteration=model.best_iteration)
    return wmape(val_df["target_next7"].values, pred)


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg   = _load_cfg()
    fcfg  = cfg["models"]["forecast"]

    feats_path = Path(cfg["features"]["out_dir"]) / cfg["features"]["processed_forecast_path"]
    df = pd.read_parquet(feats_path)
    if fcfg["drop_zero_target"]:
        df = df[df["target_next7"] > 0]

    df["date"] = pd.to_datetime(df["date"])
    split_date = df["date"].max() - pd.Timedelta(days=cfg["models"]["val_split_days"])
    train_df   = df[df["date"] < split_date]
    val_df     = df[df["date"] >= split_date]
    feats      = [c for c in df.columns if c not in ("itemid", "date", "target_next7")]

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda tr: objective(tr, train_df, val_df, feats, fcfg["use_sample_weight"]),
        n_trials=fcfg["optuna_trials"],
        timeout=fcfg["optuna_timeout_minutes"] * 60,
        show_progress_bar=True,
    )

    print("Best wMAPE:", study.best_value)

    # Retrain with best params
    best_params = study.best_params | {
        "objective": "regression",
        "metric":    "mae",
        "verbosity": -1,
        "seed":      42,
    }
    dtrain = lgb.Dataset(train_df[feats], label=train_df["target_next7"],
                         weight=train_df["sales_sum_7d"].fillna(0.1).clip(lower=0.1)
                         if fcfg["use_sample_weight"] else None)
    best_model = lgb.train(best_params, dtrain,
                           num_boost_round=study.best_trial.number + 50)

    # Final eval on validation
    pred       = best_model.predict(val_df[feats])
    val_mae    = mean_absolute_error(val_df["target_next7"], pred)
    val_wmape  = wmape(val_df["target_next7"].values, pred)

    metrics = {
        "val_mae":          float(val_mae),
        "val_wmape":        float(val_wmape),
        "pass_wmape_gate":  bool(val_wmape <= 0.10),
        "best_params":      study.best_params,
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics_forecast_tuned.yaml", "w") as f:
        yaml.dump(metrics, f)

    # Save model + update cfg
    joblib.dump(best_model, fcfg["tuned_model_weighted_path"])
    cfg["models"]["forecast"].update(study.best_params)
    _save_cfg(cfg)


if __name__ == "__main__":
    main()
