#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from npe_demography.config import load_config, ensure_dir
from npe_demography.nsf import FlowConfig, NeuralSplineFlow


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NSF NPE model.")
    p.add_argument("--config", required=True, help="Path to config YAML")
    p.add_argument("--simulations", default="simulations/sim_data.npz", help="Input .npz from simulate.py")
    p.add_argument("--out", default="models/nsf_model.pt", help="Output model path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def _standardize(a: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (a - mean) / std, {"mean": mean, "std": std}


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)

    data = np.load(args.simulations, allow_pickle=True)
    X = data["X"].astype(np.float32)
    Theta = data["Theta"].astype(np.float32)
    theta_keys = [str(x) for x in data["theta_keys"].tolist()]
    pop_order = [str(x) for x in data["pop_order"].tolist()]

    # scaling
    x_scaler = None
    t_scaler = None
    if cfg["npe"].get("standardize_x", True):
        X, x_scaler = _standardize(X)
    if cfg["npe"].get("standardize_theta", True):
        Theta, t_scaler = _standardize(Theta)

    # train/val split with safety checks
    rng = np.random.default_rng(int(cfg.get("project", {}).get("seed", 42)))
    n = X.shape[0]
    
    # Ensure we have enough samples for meaningful training
    min_required = 100  # Minimum for stable training
    if n < min_required:
        raise ValueError(
            f"Insufficient simulations: got {n}, need at least {min_required}\n"
            f"Increase n_sims in config file."
        )
    
    idx = rng.permutation(n)
    
    # Adaptive validation split: 10% of data, capped at 5000, minimum 50
    val_frac = float(cfg["npe"].get("val_frac", 0.1))
    max_val = int(cfg["npe"].get("max_val", 5000))
    min_val = int(cfg["npe"].get("min_val", 50))

    n_val = int(val_frac * n)
    n_val = max(min_val, min(max_val, n_val))
    n_val = min(n_val, n - min_val)  # ensure at least min_val training samples remain


    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    
    print(f"Train/val split: {len(tr_idx)} training, {len(val_idx)} validation")

    Xtr = torch.from_numpy(X[tr_idx])
    Ttr = torch.from_numpy(Theta[tr_idx])
    Xva = torch.from_numpy(X[val_idx])
    Tva = torch.from_numpy(Theta[val_idx])

    ds_tr = TensorDataset(Xtr, Ttr)
    ds_va = TensorDataset(Xva, Tva)

    bs = int(cfg["npe"].get("batch_size", 64))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False)

    device = torch.device(args.device)

    flow_cfg = cfg["npe"].get("flow", {})
    flow_config = FlowConfig(
        hidden_sizes=list(flow_cfg.get("hidden_sizes", cfg["npe"].get("hidden_sizes", [128, 128]))),
        num_layers=int(flow_cfg.get("num_layers", 4)),
        num_bins=int(flow_cfg.get("num_bins", 8)),
        tail_bound=float(flow_cfg.get("tail_bound", 3.0)),
        min_bin_width=float(flow_cfg.get("min_bin_width", 1e-3)),
        min_bin_height=float(flow_cfg.get("min_bin_height", 1e-3)),
        min_derivative=float(flow_cfg.get("min_derivative", 1e-3)),
        # NEW LINE: Reads "dropout" from config (default 0.0)
        dropout=float(flow_cfg.get("dropout", cfg["npe"].get("dropout", 0.2))),
    )

    model = NeuralSplineFlow(
        theta_dim=Theta.shape[1],
        context_dim=X.shape[1],
        config=flow_config,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["npe"].get("lr", 5e-4)),
        weight_decay=float(cfg["npe"].get("weight_decay", 1e-5)),
    )

    max_epochs = int(cfg["npe"].get("max_epochs", 200))
    patience = int(cfg["npe"].get("early_stop_patience", 25))

    best = float("inf")
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, tb in dl_tr:
            xb = xb.to(device)
            tb = tb.to(device)
            log_prob = model.log_prob(tb, xb)
            loss = -log_prob.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += float(loss.item()) * xb.shape[0]
        tr_loss /= len(ds_tr)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, tb in dl_va:
                xb = xb.to(device)
                tb = tb.to(device)
                log_prob = model.log_prob(tb, xb)
                loss = -log_prob.mean()
                va_loss += float(loss.item()) * xb.shape[0]
        va_loss /= len(ds_va)

        print(f"epoch {epoch:03d}  train_nll={tr_loss:.4f}  val_nll={va_loss:.4f}")

        if va_loss < best - 1e-4:
            best = va_loss
            bad = 0
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "x_dim": X.shape[1],
                    "theta_dim": Theta.shape[1],
                    "model_type": "nsf",
                    "flow_config": {
                        "hidden_sizes": list(flow_config.hidden_sizes),
                        "num_layers": flow_config.num_layers,
                        "num_bins": flow_config.num_bins,
                        "tail_bound": flow_config.tail_bound,
                        "min_bin_width": flow_config.min_bin_width,
                        "min_bin_height": flow_config.min_bin_height,
                        "min_derivative": flow_config.min_derivative,
                    },
                    "theta_keys": theta_keys,
                    "pop_order": pop_order,
                    "x_scaler": x_scaler,
                    "theta_scaler": t_scaler,
                    "config_path": str(args.config),
                    "n_posterior_samples": int(cfg["npe"].get("n_posterior_samples", 50000)),
                },
                args.out,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch} (best val_nll={best:.4f}).")
                break

    print(f"Saved best model to {args.out}")


if __name__ == "__main__":
    main()
