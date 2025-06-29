"""
tune_GRU4Rec.py  –  GPU-aware Optuna sweep for GRU4Rec
======================================================

* Maximises NDCG@20, prunes early
* Uses CUDA automatically if present (batches pinned & non-blocking)
* Outputs unchanged (model, SQLite study, PNG, Markdown)
"""

from __future__ import annotations
import argparse, json, os, platform
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .GRU4REC_baseline import (GRU4Rec, _SeqDataset, _collate,
                               _evaluate, _load_cfg, _set_seed)


# ────────── helpers ──────────
def _load_seqs(cfg: Dict) -> Tuple[List[List[int]], int]:
    p = Path(cfg["features"]["out_dir"]) / cfg["features"].get(
        "processed_reco_path", "reco_sequences.parquet")
    df = pd.read_parquet(p)
    seqs = df["item_seq"].tolist()
    return seqs, int(max(max(s) for s in seqs))

def _split(seqs, seed=42):
    rng = np.random.RandomState(seed)
    m = rng.rand(len(seqs)) < 0.8
    return [s for keep, s in zip(m, seqs) if keep], [s for keep, s in zip(m, seqs) if not keep]

def _train_one(model, loader, opt, device):
    ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    for i, (seq, tgt) in enumerate(loader):
        seq, tgt = seq.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        opt.zero_grad(); loss = ce(model(seq), tgt)
        loss.backward(); opt.step()
        # Print progress every 100 batches to detect hanging
        if i % 100 == 0:
            tqdm.write(f"    Batch {i}, Loss: {loss.item():.4f}")


# ────────── Optuna objective ──────────
def _objective(trial: optuna.Trial, cfg: Dict,
               seqs: List[List[int]], n_items: int,
               device: torch.device) -> float:
    tqdm.write(f"Starting trial {trial.number}")
    
    emb   = trial.suggest_categorical("emb",   [32, 64, 96])
    hid   = trial.suggest_categorical("hid",   [32, 64, 96, 128])
    layers= trial.suggest_int        ("layers", 1, 2)
    drop  = trial.suggest_float      ("drop",  0.0, 0.5)
    lr    = trial.suggest_float      ("lr",    1e-4, 3e-3, log=True)
    wd    = trial.suggest_float      ("wd",    1e-6, 1e-4, log=True)
    batch = trial.suggest_categorical("batch", [256, 384, 512])
    epochs = 10
    
    tqdm.write(f"Trial {trial.number} params: emb={emb}, hid={hid}, batch={batch}, lr={lr:.2e}")
    
    tr_seqs, va_seqs = _split(seqs)
    max_len = cfg["models"]["reco"].get("max_seq_len", 50)
    
    # Reduce num_workers to avoid potential deadlocks
    tr_loader = DataLoader(_SeqDataset(tr_seqs, max_len), batch_size=batch,
                           shuffle=True, num_workers=2,
                           pin_memory=(device.type == "cuda"), collate_fn=_collate)
    va_loader = DataLoader(_SeqDataset(va_seqs, max_len), batch_size=1024,
                           shuffle=False, num_workers=2,
                           pin_memory=(device.type == "cuda"), collate_fn=_collate)

    model = GRU4Rec(n_items, emb, hid, layers, drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best = 0.0
    for ep in range(1, epochs + 1):
        tqdm.write(f"  Trial {trial.number}, Epoch {ep}")
        _train_one(model, tr_loader, opt, device)
        _, ndcg = _evaluate(model, va_loader, device)
        tqdm.write(f"  Trial {trial.number}, Epoch {ep}: NDCG={ndcg:.4f}")
        trial.report(ndcg, ep)
        if trial.should_prune(): 
            tqdm.write(f"  Trial {trial.number} pruned at epoch {ep}")
            raise optuna.TrialPruned()
        best = max(best, ndcg)
    
    tqdm.write(f"Trial {trial.number} completed with best NDCG={best:.4f}")
    return best


# ─────────────── main ───────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--trials", type=int, default=80)
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.cfg)); _set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"🚀 Using device: {device} "
               f"({torch.cuda.get_device_name() if device.type=='cuda' else platform.processor()})")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    seqs, n_items = _load_seqs(cfg)

    Path("artefacts").mkdir(exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url="sqlite:///artefacts/optuna_gru4rec.db",
        engine_kwargs={"connect_args": {"check_same_thread": False}})
    study = optuna.create_study(
        study_name="gru4rec_tuning",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda t: _objective(t, cfg, seqs, n_items, device),
                   n_trials=args.trials, show_progress_bar=True)

    # PNG
    Path("reports").mkdir(exist_ok=True)
    df = study.trials_dataframe(attrs=("number", "value"))
    plt.figure(figsize=(6,4))
    plt.plot(df["number"], df["value"])
    plt.xlabel("Trial"); plt.ylabel("Best NDCG@20 so far")
    plt.title("Optuna GRU4Rec Progress"); plt.tight_layout()
    plt.savefig("reports/tunedgru4rec_study_curve.png", dpi=120); plt.close()

    # final refit
    p = study.best_params
    max_len = cfg["models"]["reco"].get("max_seq_len", 50)
    tr_loader = DataLoader(_SeqDataset(seqs, max_len), batch_size=p["batch"],
                           shuffle=True, num_workers=os.cpu_count(),
                           pin_memory=(device.type == "cuda"), collate_fn=_collate)

    model = GRU4Rec(n_items, p["emb"], p["hid"], p["layers"], p["drop"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["wd"])
    for _ in tqdm(range(20), desc="Final fit", unit="epoch"):
        _train_one(model, tr_loader, opt, device)

    torch.save({"state_dict": model.state_dict(),
                "n_items": n_items, "best_params": p, "cfg": cfg},
               "artefacts/gru4rec_tuned.pt")

    # metrics
    full_loader = DataLoader(_SeqDataset(seqs, max_len), batch_size=1024,
                             shuffle=False, num_workers=os.cpu_count(),
                             pin_memory=(device.type == "cuda"), collate_fn=_collate)
    hr, ndcg = _evaluate(model, full_loader, device)
    Path("reports/metrics_reco_tuned.md").write_text(
f"""# Tuned GRU4Rec Report

| Metric      | Value |
|-------------|-------|
| HitRate@20  | {hr:.3f} |
| NDCG@20     | {ndcg:.3f} |
| Trials      | {len(study.trials)} |
| Best trial  | {study.best_trial.number} |
| Device      | {device} |
| Pass Gates? | {'✅' if ndcg>=0.14 and hr>=0.35 else '❌'} |

```json
{json.dumps(p, indent=2)}
```""", encoding="utf-8")

    print("Model saved ➜ artefacts/gru4rec_tuned.pt")

if __name__ == "__main__":
    main()
