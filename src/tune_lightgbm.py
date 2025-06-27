"""
Optuna tuning → Markdown report + visuals + config update.
"""
from __future__ import annotations
import warnings, yaml, optuna, joblib, lightgbm as lgb
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
def wmape(y, yhat): denom = np.abs(y).sum();  return np.abs(y - yhat).sum() / denom if denom else np.nan
def _cfg(path="config.yaml"):  return yaml.safe_load(open(path, "r", encoding="utf‑8"))
def _save(cfg, path="config.yaml"): yaml.dump(cfg, open(path,"w",encoding="utf‑8"))

# ──────────────────────────────────────────────────────────────────────────────
def objective(tr, vl, feats, use_w, trial):
    w = tr["sales_sum_7d"].fillna(0.1).clip(lower=0.1) if use_w else None
    dtr = lgb.Dataset(tr[feats], label=tr["target_next7"], weight=w)
    dva = lgb.Dataset(vl[feats], label=vl["target_next7"])
    p = {"objective":"regression","metric":"mae","verbosity":-1,"seed":42,
         "learning_rate":trial.suggest_float("learning_rate",.01,.3,log=True),
         "num_leaves":trial.suggest_int("num_leaves",8,256,log=True),
         "min_data_in_leaf":trial.suggest_int("min_data_in_leaf",5,150),
         "lambda_l1":trial.suggest_float("lambda_l1",0,10),
         "lambda_l2":trial.suggest_float("lambda_l2",0,10),
         "feature_fraction":trial.suggest_float("feature_fraction",.5,1),
         "bagging_fraction":trial.suggest_float("bagging_fraction",.5,1),
         "bagging_freq":trial.suggest_int("bagging_freq",1,15)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        m = lgb.train(p, dtr, 1000, [dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = m.predict(vl[feats], num_iteration=m.best_iteration)
    return wmape(vl["target_next7"].values, preds)

# ──────────────────────────────────────────────────────────────────────────────
def _visual(vl, preds, model, rpt: Path):
    # Only error histogram to avoid duplicating earlier plots
    err = np.abs(vl["target_next7"] - preds)
    plt.figure(figsize=(4,3))
    plt.hist(err, bins=40, edgecolor="k"); plt.title("Optuna: |Error| Hist"); plt.tight_layout()
    plt.savefig(rpt / "optuna_error_hist.png", dpi=120); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
def main():
    cfg   = _cfg();  fcfg = cfg["models"]["forecast"]
    feats_p = Path(cfg["features"]["out_dir"]) / cfg["features"]["processed_forecast_path"]
    df   = pd.read_parquet(feats_p)
    if fcfg["drop_zero_target"]: df = df[df["target_next7"]>0]
    df["date"] = pd.to_datetime(df["date"])
    split = df["date"].max() - pd.Timedelta(days=cfg["models"]["val_split_days"])
    tr, vl = df[df["date"]<split], df[df["date"]>=split]
    feats  = [c for c in df.columns if c not in ("itemid","date","target_next7")]

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda t: objective(tr, vl, feats, fcfg["use_sample_weight"], t),
                   n_trials=fcfg["optuna_trials"],
                   timeout=fcfg["optuna_timeout_minutes"]*60,
                   show_progress_bar=True)

    best = study.best_params | {"objective":"regression","metric":"mae","verbosity":-1,"seed":42}
    dtr  = lgb.Dataset(tr[feats], label=tr["target_next7"],
                       weight=tr["sales_sum_7d"].fillna(0.1).clip(lower=0.1)
                       if fcfg["use_sample_weight"] else None)
    mdl  = lgb.train(best, dtr, study.best_trial.number+50)
    pred = mdl.predict(vl[feats]);  mae = mean_absolute_error(vl["target_next7"], pred)
    wmp  = wmape(vl["target_next7"].values, pred);  pass_gate = wmp<=0.10

    rpt = Path("reports"); rpt.mkdir(exist_ok=True)
    rpt_md = rpt / "metrics_forecast_tuned.md"
    rpt_md.write_text(
        f"""# Optuna Tuning Report\n
**Best wMAPE:** {wmp:.5%} ({'✅ pass' if pass_gate else '❌ fail'})\n
| Metric | Value |\n|---|---|\n| MAE | {mae:.5f} |\n| wMAPE | {wmp:.5%} |\n| Trials | {len(study.trials)} |\n| Pass Gate ≤10 % | {'Yes' if pass_gate else 'No'} |\n\n
## Best Parameters\n```yaml\n{yaml.dump(study.best_params)}\n```\n"""
    , encoding="utf‑8")

    _visual(vl, pred, mdl, rpt)

    joblib.dump(mdl, fcfg["tuned_model_weighted_path"])
    cfg["models"]["forecast"].update(study.best_params); _save(cfg)

    print("Best wMAPE:", wmp, "| pass gate:", pass_gate)


if __name__ == "__main__":
    main()
