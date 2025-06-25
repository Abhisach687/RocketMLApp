"""
Feature-engineering stage for the Retailrocket project, including:
  - Forecasting features: daily counts, rolling windows, lags (by shift), ratios,
    category smoothing, calendar effects, price features, category-level aggregates
  - Recommendation sequences: item & category indices, recency & time-of-day features

Reads:
    data/interim/events_clean.parquet
    data/interim/item_properties.parquet
    data/interim/category_tree.parquet
    config.yaml

Writes:
    data/processed/forecast_features.parquet   – per (itemid, date)
    data/processed/reco_sequences.parquet      – per visitor sequence
    artefacts/item2idx.json                    – mapping for recommender
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import yaml


def _load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rolling_sum(df: pd.DataFrame, window: int, col: str) -> pd.Series:
    return (
        df.sort_values(["itemid","date"])
          .groupby("itemid")[col]
          .rolling(window, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )


def _smooth(series: pd.Series, cat_series: pd.Series, alpha: float = 10.0) -> pd.Series:
    cat_mean = series.groupby(cat_series).transform("mean")
    return (series + alpha * cat_mean) / (1 + alpha)


def build_forecast_features(events: pd.DataFrame, props: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = events.copy()
    df['date'] = df['timestamp'].dt.normalize()

    # daily pivot counts
    daily = (
        df.pivot_table(
            index=['itemid','date'],
            columns='event',
            values='visitorid',
            aggfunc='count',
            fill_value=0
        )
        .rename(columns={'view':'views','addtocart':'adds','transaction':'sales'})
        .reset_index()
    )
    for col in ['views','adds','sales']:
        daily[col] = daily.get(col, 0)

    # price
    price_df = (
        props[props['property']=='price'][['itemid','value']]
        .rename(columns={'value':'price'})
    )
    price_df['price'] = pd.to_numeric(price_df['price'], errors='coerce')
    daily = daily.merge(
        price_df.drop_duplicates('itemid'),
        on='itemid', how='left'
    )
    daily['price'] = daily['price'].ffill()

    # category
    cat_df = (
        props[props['property']=='categoryid'][['itemid','value']]
        .rename(columns={'value':'categoryid'})
        .drop_duplicates('itemid')
    )
    daily = daily.merge(cat_df, on='itemid', how='left')

    # rolling sums
    for w in cfg['features'].get('rolling_windows',[7,14,30]):
        for col in ['views','adds','sales']:
            daily[f'{col}_sum_{w}d'] = _rolling_sum(daily, w, col)

    # lag features
    daily = daily.sort_values(['itemid','date'])
    for lag in cfg['features'].get('lag_days',[1,7,14]):
        daily[f'sales_lag_{lag}d'] = (
            daily.groupby('itemid')['sales'].shift(lag).fillna(0)
        )
        daily[f'views_lag_{lag}d'] = (
            daily.groupby('itemid')['views'].shift(lag).fillna(0)
        )

    # ratios
    daily['ctr_7d'] = daily['adds_sum_7d'] / daily['views_sum_7d'].replace(0,np.nan)
    daily['buyrate_7d'] = daily['sales_sum_7d'] / daily['views_sum_7d'].replace(0,np.nan)

    # category-level
    cat_sales = (
        daily.groupby(['categoryid','date'])['sales_sum_7d']
             .sum().rename('cat_sales_7d')
    )
    daily = daily.merge(
        cat_sales.reset_index(), on=['categoryid','date'], how='left'
    )
    daily['sales_sm_7d'] = _smooth(daily['sales_sum_7d'], daily['categoryid'], alpha=cfg['features'].get('smooth_alpha',5))

    # price change
    daily['price_lag_7d'] = daily.groupby('itemid')['price'].shift(7)
    daily['price_pct_chg_7d'] = (
        daily['price'] - daily['price_lag_7d']
    ) / daily['price_lag_7d'].replace(0, np.nan)

    # calendar
    daily['dow'] = daily['date'].dt.dayofweek
    daily['is_weekend'] = daily['dow'].isin([5,6]).astype(int)
    daily = pd.concat([daily, pd.get_dummies(daily['dow'], prefix='dow')], axis=1)

    # target
    tgt = daily[['itemid','date','sales']].copy()
    tgt['date'] = tgt['date'] - pd.Timedelta(days=7)
    tgt = tgt.rename(columns={'sales':'target_next7'})
    daily = daily.merge(tgt, on=['itemid','date'], how='left').fillna({'target_next7':0})

    return daily.drop(columns=['views','adds','sales','price_lag_7d','dow'])


def build_reco_sequences(events: pd.DataFrame, props: pd.DataFrame, cats: pd.DataFrame, cfg: Dict[str,Any]) -> tuple[pd.DataFrame,Dict[int,int]]:
    df = events.copy()
    # item index
    uniques = df['itemid'].unique()
    item2idx = {int(i): idx+1 for idx,i in enumerate(uniques)}
    df['item_idx'] = df['itemid'].map(item2idx)

    # category index
    cat_map = props[props['property']=='categoryid'][['itemid','value']].rename(columns={'value':'categoryid'})
    uniques_cat = cats['categoryid'].unique()
    cat2idx = {int(c): idx+1 for idx,c in enumerate(uniques_cat)}
    df = df.merge(cat_map, on='itemid', how='left')
    df['cat_idx'] = df['categoryid'].map(cat2idx).fillna(0).astype(int)

    # build sequences without GroupBy.apply warning
    df_sorted = df.sort_values(['visitorid','timestamp'])
    grouped = df_sorted.groupby('visitorid')
    item_seq = grouped['item_idx'].apply(lambda x: list(x.tail(cfg['features']['max_seq_length'])))
    cat_seq  = grouped['cat_idx'].apply(lambda x: list(x.tail(cfg['features']['max_seq_length'])))
    time_seq = grouped['timestamp'].apply(lambda x: list((x - x.iloc[0]).dt.total_seconds()/60))

    reco_df = pd.DataFrame({
        'visitorid': item_seq.index,
        'item_seq': item_seq.values,
        'cat_seq': cat_seq.values,
        'time_seq': time_seq.values,
    })
    return reco_df, item2idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature eng for Retailrocket dataset")
    parser.add_argument("--cfg", default="config.yaml")
    parser.add_argument("--in_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--art_dir", default=None)
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.cfg))
    in_dir = Path(args.in_dir or cfg['features']['in_dir'])
    out_dir = Path(args.out_dir or cfg['features']['out_dir'])
    art_dir = Path(args.art_dir or cfg['features'].get('artefacts_dir','artefacts'))
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info("Loading data…")
    events = pd.read_parquet(in_dir/'events_clean.parquet')
    props  = pd.read_parquet(in_dir/'item_properties.parquet')
    cats   = pd.read_parquet(in_dir/'category_tree.parquet')

    logging.info("Building forecast features…")
    ff = build_forecast_features(events, props, cfg)
    ff.to_parquet(out_dir/'forecast_features.parquet', index=False)

    logging.info("Building recommendation sequences…")
    reco_df, item2idx = build_reco_sequences(events, props, cats, cfg)
    reco_df.to_parquet(out_dir/'reco_sequences.parquet', index=False)
    with open(art_dir/'item2idx.json','w',encoding='utf-8') as f:
        json.dump(item2idx,f)

    logging.info("Done.")

if __name__=='__main__':
    main()  