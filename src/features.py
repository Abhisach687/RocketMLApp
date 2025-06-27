"""
Enhanced feature‑engineering for the RetailRocket dataset
(FORECAST + RECOMMENDATION) with visible TQDM progress bars.

Outputs
-------
data/processed/forecast_features.parquet
data/processed/reco_sequences.parquet
artefacts/item2idx.json
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
def _load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf‑8") as f:
        return yaml.safe_load(f)


def _rolling_sum(df: pd.DataFrame, window: int, col: str) -> pd.Series:
    """Item‑level rolling sum with min_periods=1."""
    return (
        df.sort_values(["itemid", "date"])
          .groupby("itemid")[col]
          .rolling(window, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )


def _smooth(series: pd.Series, cat_series: pd.Series, alpha: float = 10.0) -> pd.Series:
    cat_mean = series.groupby(cat_series).transform("mean")
    return (series + alpha * cat_mean) / (1.0 + alpha)


# ──────────────────────────────────────────────────────────────────────────────
def build_forecast_features(events: pd.DataFrame,
                            props:  pd.DataFrame,
                            cfg:    Dict[str, Any]) -> pd.DataFrame:
    """Return enhanced item‑daily feature frame."""
    df = events.copy()
    df["date"] = df["timestamp"].dt.normalize()

    # ── Daily counts ─────────────────────────────────────────────────────────
    daily = (
        df.pivot_table(index=["itemid", "date"],
                       columns="event",
                       values="visitorid",
                       aggfunc="count",
                       fill_value=0)
          .rename(columns={"view": "views",
                           "addtocart": "adds",
                           "transaction": "sales"})
          .reset_index()
    )
    for c in ["views", "adds", "sales"]:
        daily[c] = daily.get(c, 0)

    # ── Price ────────────────────────────────────────────────────────────────
    price_df = props.loc[props["property"] == "price", ["itemid", "value"]]\
                    .rename(columns={"value": "price"})
    price_df["price"] = pd.to_numeric(price_df["price"], errors="coerce")
    daily = daily.merge(price_df.drop_duplicates("itemid"), on="itemid", how="left")
    daily["price"] = daily["price"].ffill()

    # ── Category ─────────────────────────────────────────────────────────────
    cat_df = props.loc[props["property"] == "categoryid", ["itemid", "value"]]\
                  .rename(columns={"value": "categoryid"})\
                  .drop_duplicates("itemid")
    daily = daily.merge(cat_df, on="itemid", how="left")

    # ── Rolling windows ──────────────────────────────────────────────────────
    roll_windows = cfg["features"].get("rolling_windows", [3, 7, 14, 30, 60])
    for w in tqdm(roll_windows, desc="Rolling windows"):
        for col in ["views", "adds", "sales"]:
            daily[f"{col}_sum_{w}d"] = _rolling_sum(daily, w, col)

    # ── Lags ─────────────────────────────────────────────────────────────────
    lag_days = cfg["features"].get("lag_days", [1, 7, 14])
    daily = daily.sort_values(["itemid", "date"])
    for lag in lag_days:
        daily[f"sales_lag_{lag}d"] = daily.groupby("itemid")["sales"].shift(lag).fillna(0)
    # Same weekday last week
    daily["sales_lag_dow"] = (
        daily.groupby("itemid")["sales"].shift(7)
             .where(daily["date"].dt.dayofweek ==
                    (daily["date"] - pd.Timedelta(days=7)).dt.dayofweek, 0)
             .fillna(0)
    )

    # ── Ratios & smoothed ratios ─────────────────────────────────────────────
    daily["ctr_7d"]     = daily["adds_sum_7d"]  / daily["views_sum_7d"].replace(0, np.nan)
    daily["buyrate_7d"] = daily["sales_sum_7d"] / daily["views_sum_7d"].replace(0, np.nan)
    daily["ctr_sm_7d"]     = _smooth(daily["ctr_7d"].fillna(0),     daily["categoryid"])
    daily["buyrate_sm_7d"] = _smooth(daily["buyrate_7d"].fillna(0), daily["categoryid"])

    # ── Category aggregates ──────────────────────────────────────────────────
    cat_sales = (
        daily.groupby(["categoryid", "date"])["sales_sum_7d"]
             .sum()
             .rename("cat_sales_7d")
             .reset_index()
    )
    daily = daily.merge(cat_sales, on=["categoryid", "date"], how="left")
    daily["sales_sm_7d"] = _smooth(
        daily["sales_sum_7d"],
        daily["categoryid"],
        alpha=cfg["features"].get("smooth_alpha", 5)
    )

    # ── Price change ─────────────────────────────────────────────────────────
    daily["price_lag_7d"]     = daily.groupby("itemid")["price"].shift(7)
    daily["price_pct_chg_7d"] = (daily["price"] - daily["price_lag_7d"]) \
                                / daily["price_lag_7d"].replace(0, np.nan)

    # ── Calendar ─────────────────────────────────────────────────────────────
    daily["dow"]         = daily["date"].dt.dayofweek
    daily["is_weekend"]  = daily["dow"].isin([5, 6]).astype(int)
    daily["weekofyear"]  = daily["date"].dt.isocalendar().week.astype(int)
    daily["month"]       = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayof_year
    daily = pd.concat([daily,
                       pd.get_dummies(daily["dow"], prefix="dow", dtype="int8")],
                      axis=1)

    # ── Target (next‑7‑day sales) ────────────────────────────────────────────
    tgt = daily[["itemid", "date", "sales"]].copy()
    tgt["date"] = tgt["date"] - pd.Timedelta(days=7)
    tgt         = tgt.rename(columns={"sales": "target_next7"})
    daily       = daily.merge(tgt, on=["itemid", "date"], how="left") \
                       .fillna({"target_next7": 0})

    return daily.drop(columns=["views", "adds", "sales", "price_lag_7d", "dow"])


# ──────────────────────────────────────────────────────────────────────────────
def build_reco_sequences(events: pd.DataFrame,
                         props:  pd.DataFrame,
                         cats:   pd.DataFrame,
                         cfg:    Dict[str, Any]) -> tuple[pd.DataFrame, Dict[int, int]]:

    df = events.copy()
    uniques = df["itemid"].unique()
    item2idx = {int(i): ix + 1 for ix, i in enumerate(uniques)}
    df["item_idx"] = df["itemid"].map(item2idx)

    cat_map = props.loc[props["property"] == "categoryid",
                        ["itemid", "value"]].rename(columns={"value": "categoryid"})
    cat2idx = {int(c): ix + 1 for ix, c in enumerate(cats["categoryid"].unique())}
    df = df.merge(cat_map, on="itemid", how="left")
    df["cat_idx"] = df["categoryid"].map(cat2idx).fillna(0).astype(int)

    df = df.sort_values(["visitorid", "timestamp"])
    grouped = df.groupby("visitorid")
    max_len = cfg["features"]["max_seq_length"]

    item_seq = grouped["item_idx"].apply(lambda x: list(x.tail(max_len)))
    cat_seq  = grouped["cat_idx"].apply(lambda x: list(x.tail(max_len)))
    time_seq = grouped["timestamp"].apply(
        lambda x: list((x - x.iloc[0]).dt.total_seconds() // 60)
    )

    reco_df = pd.DataFrame({
        "visitorid": item_seq.index,
        "item_seq":  item_seq.values,
        "cat_seq":   cat_seq.values,
        "time_seq":  time_seq.values,
    })
    return reco_df, item2idx


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("--cfg",     default="config.yaml")
    parser.add_argument("--in_dir",  default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--art_dir", default=None)
    args = parser.parse_args()

    cfg      = _load_cfg(Path(args.cfg))
    in_dir   = Path(args.in_dir or cfg["features"]["in_dir"])
    out_dir  = Path(args.out_dir or cfg["features"]["out_dir"])
    art_dir  = Path(args.art_dir or cfg["features"].get("artefacts_dir", "artefacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    logging.info("Loading data…")
    events = pd.read_parquet(in_dir / "events_clean.parquet")
    props  = pd.read_parquet(in_dir / "item_properties.parquet")
    cats   = pd.read_parquet(in_dir / "category_tree.parquet")

    logging.info("Building forecast features…")
    ff = build_forecast_features(events, props, cfg)
    ff.to_parquet(out_dir / "forecast_features.parquet", index=False)

    logging.info("Building reco sequences…")
    reco_df, item2idx = build_reco_sequences(events, props, cats, cfg)
    reco_df.to_parquet(out_dir / "reco_sequences.parquet", index=False)
    with open(art_dir / "item2idx.json", "w", encoding="utf‑8") as f:
        json.dump(item2idx, f)

    logging.info("✔ Feature engineering done")


if __name__ == "__main__":
    main()
