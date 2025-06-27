"""
– Forecast model training
• Filters out non‑LightGBM params
• Uses Booster.feature_name()
• Suppresses SettingWithCopyWarning
• Keeps wMAPE early‑stopping + Markdown & PNG outputs
"""
from __future__ import annotations
import warnings, yaml, joblib, lightgbm as lgb, matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
def wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    d = np.abs(y).sum()
    return np.abs(y - yhat).sum() / d if d else np.nan


def _cfg(p="config.yaml") -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _Bar:  # tqdm callback
    def __init__(self, total): self.t = tqdm(total=total, desc="Train", unit="iter", leave=False)
    def __call__(self, env):   self.t.update(1); self.t.close() if env.iteration+1==self.t.total else None


# ─────────────────────────────────────────────────────────────────────────────
def _visuals(val: pd.DataFrame, pred: np.ndarray, mdl: lgb.Booster, rpt: Path):
    y = val["target_next7"].values
    # Pred vs Actual
    plt.figure(figsize=(4,4)); plt.scatter(y, pred, s=4, alpha=.3)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.tight_layout()
    plt.savefig(rpt/"lightgbm_val_pred_vs_true.png", dpi=120); plt.close()
    # Error hist
    plt.figure(figsize=(4,3)); plt.hist(np.abs(y-pred), bins=40, edgecolor="k")
    plt.xlabel("|Error|"); plt.title("Absolute‑Error"); plt.tight_layout()
    plt.savefig(rpt/"lightgbm_val_error_hist.png", dpi=120); plt.close()
    # Feature importance
    gain = mdl.feature_importance("gain"); names = mdl.feature_name()
    top  = (pd.DataFrame({"feature":names,"gain":gain})
              .sort_values("gain", ascending=False).head(20))
    plt.figure(figsize=(6,4))
    plt.barh(top["feature"][::-1], top["gain"][::-1])
    plt.title("Top‑20 Feature Importance"); plt.tight_layout()
    plt.savefig(rpt/"lightgbm_val_feature_importance.png", dpi=120); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, cfg: Dict[str, Any]) -> bool:
    fcfg, vdays = cfg["models"]["forecast"], cfg["models"]["val_split_days"]
    if fcfg["drop_zero_target"]: df = df[df["target_next7"]>0]

    # avoid SettingWithCopyWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
        df["date"] = pd.to_datetime(df["date"])

    split = df["date"].max() - pd.Timedelta(days=vdays)
    tr, va = df[df["date"]<split], df[df["date"]>=split]
    Xtr = tr.drop(columns=["itemid","date","target_next7"]); ytr = tr["target_next7"]
    Xva = va.drop(columns=["itemid","date","target_next7"]); yva = va["target_next7"]
    w   = tr["sales_sum_7d"].clip(lower=0.1) if fcfg["use_sample_weight"] else None

    dtr = lgb.Dataset(Xtr, label=ytr, weight=w); dva = lgb.Dataset(Xva, label=yva)

    # keep only legal LightGBM params
    allowed = {"objective","metric","learning_rate","num_leaves",
               "feature_fraction","bagging_fraction","bagging_freq",
               "min_data_in_leaf","lambda_l1","lambda_l2"}
    params  = {k:fcfg[k] for k in allowed if k in fcfg}
    params.update({"objective":"poisson", "metric":"mae", "verbosity":-1, "seed":42})

    def _wmape_eval(preds, data):
        return "wMAPE", wmape(data.get_label(), preds), False  # lower better

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mdl = lgb.train(params, dtr,
                        num_boost_round=4000,
                        valid_sets=[dva],
                        feval=_wmape_eval,
                        callbacks=[lgb.early_stopping(150, verbose=False),
                                   _Bar(4000)])

    preds   = mdl.predict(Xva, num_iteration=mdl.best_iteration)
    mae     = mean_absolute_error(yva, preds)
    wm_err  = wmape(yva.values, preds); passed = wm_err <= 0.10

    rpt = Path("reports"); rpt.mkdir(exist_ok=True)
    _visuals(va, preds, mdl, rpt)
    (rpt/"metrics_forecast_final.md").write_text(
f"""# Baseline Forecast Report

| Metric | Value |
| ------ | ----- |
| MAE | {mae:.5f} |
| wMAPE | {wm_err:.2%} |
| Best iteration | {mdl.best_iteration} |
| Pass ≤ 10 %? | {'✅' if passed else '❌'} |
""", encoding="utf-8")

    Path(fcfg["weighted_model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl, fcfg["weighted_model_path"])
    return passed


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg  = _cfg()
    feats= Path(cfg["features"]["out_dir"]) / cfg["features"]["processed_forecast_path"]
    ok   = train(pd.read_parquet(feats), cfg)
    print("wMAPE gate:", "PASSED" if ok else "NOT met")
