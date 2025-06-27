"""
Train LightGBM forecast model, log rich Markdown
and create validation visualisations.
"""
from __future__ import annotations
import warnings, yaml, joblib, lightgbm as lgb
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return np.abs(y_true - y_pred).sum() / denom if denom else np.nan


def _load_cfg(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf‑8") as f:
        return yaml.safe_load(f)


class _TQDM:
    def __init__(self, total: int):  self.pbar = tqdm(total=total, unit="iter", desc="Training", leave=False)
    def __call__(self, env):         self.pbar.update(1);  self.pbar.close() if env.iteration + 1 == self.pbar.total else None


# ──────────────────────────────────────────────────────────────────────────────
def _visualise(val_df: pd.DataFrame,
               y_pred: np.ndarray,
               model:  lgb.Booster,
               out_dir: Path) -> None:
    """Save scatter, error hist & feature importance."""
    y_true = val_df["target_next7"].values
    abs_err = np.abs(y_true - y_pred)

    # Scatter
    plt.figure(figsize=(4,4))
    plt.scatter(y_true, y_pred, s=4, alpha=.4)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(out_dir / "val_pred_vs_true.png", dpi=120)
    plt.close()

    # Error histogram
    plt.figure(figsize=(4,3))
    plt.hist(abs_err, bins=40, edgecolor="k")
    plt.xlabel("|Error|"); plt.ylabel("Freq"); plt.title("Absolute‑Error Histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "val_error_hist.png", dpi=120)
    plt.close()

    # Feature importance (gain)
    imp = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    imp_df = (pd.DataFrame({"feature": names, "gain": imp})
                .sort_values("gain", ascending=False)
                .head(20))
    plt.figure(figsize=(6,4))
    plt.barh(imp_df["feature"][::-1], imp_df["gain"][::-1])
    plt.title("Top‑20 Feature Importance (gain)")
    plt.tight_layout()
    plt.savefig(out_dir / "val_feature_importance.png", dpi=120)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, float]:
    fcfg, vdays = cfg["models"]["forecast"], cfg["models"]["val_split_days"]
    df = df[df["target_next7"].notnull()]
    if fcfg["drop_zero_target"]:
        df = df[df["target_next7"] > 0]

    df["date"] = pd.to_datetime(df["date"])
    split = df["date"].max() - pd.Timedelta(days=vdays)
    tr, val = df[df["date"] < split], df[df["date"] >= split]
    Xtr, ytr = tr.drop(columns=["itemid","date","target_next7"]), tr["target_next7"]
    Xva, yva = val.drop(columns=["itemid","date","target_next7"]), val["target_next7"]
    w = tr["sales_sum_7d"].fillna(0.1).clip(lower=0.1) if fcfg["use_sample_weight"] else None

    dtr, dva = lgb.Dataset(Xtr, ytr, weight=w), lgb.Dataset(Xva, yva)

    params = {"objective":fcfg["objective"], "metric":fcfg["metric"],
              "learning_rate":fcfg["learning_rate"], "num_leaves":fcfg["num_leaves"],
              "feature_fraction":.8, "bagging_fraction":.8, "bagging_freq":5,
              "verbosity":-1, "seed":42}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = lgb.train(params, dtr,
                          num_boost_round=fcfg["num_boost_round"],
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(fcfg["early_stopping_rounds"], verbose=False),
                                     _TQDM(fcfg["num_boost_round"])])

    y_hat   = model.predict(Xva, num_iteration=model.best_iteration)
    mae     = mean_absolute_error(yva, y_hat)
    wmape   = wmape(yva.values, y_hat)
    passed  = wmape <= 0.10

    # ── write markdown report ────────────────────────────────────────────────
    rpt = Path("reports");  rpt.mkdir(exist_ok=True)
    md  = rpt / "metrics_forecast_final.md"
    md.write_text(
        f"""# Forecast – Validation Report\n
| Metric | Value |
| ------ | ----- |
| MAE | {mae:.5f} |
| wMAPE | {wmape:.5%} |
| Best Iteration | {model.best_iteration} |
| Pass ≤ 10 % wMAPE? | {'✅' if passed else '❌'} |\n"""
    , encoding="utf‑8")

    # ── visuals ──────────────────────────────────────────────────────────────
    _visualise(val, y_hat, model, rpt)

    # ── save model ───────────────────────────────────────────────────────────
    Path(fcfg["weighted_model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, fcfg["weighted_model_path"])
    return {"mae": mae, "wmape": wmape, "pass_gate": passed}


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg  = _load_cfg()
    feats= Path(cfg["features"]["out_dir"]) / cfg["features"]["processed_forecast_path"]
    res  = train(pd.read_parquet(feats), cfg)
    print(f"wMAPE={res['wmape']:.4%}  |  Pass Gate={res['pass_gate']}")
