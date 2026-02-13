#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from npe_demography.config import load_config
from npe_demography.sbi_backend import train_npe_sbi


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

    # --- CRITICAL FIX START: Filter out NaNs and Infs ---
    # This prevents the "AssertionError: No valid data entries left" crash
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(Theta).all(axis=1)
    n_total = len(X)
    n_valid = valid_mask.sum()

    if n_valid < n_total:
        n_dropped = n_total - n_valid
        print(f"WARNING: Dropping {n_dropped} simulations containing NaNs/Infs ({100 * n_dropped / n_total:.1f}%)")
        X = X[valid_mask]
        Theta = Theta[valid_mask]

    if n_valid == 0:
        raise ValueError("All simulations contained NaNs/Infs! Check your simulation parameters/priors.")
    # --- CRITICAL FIX END ---

    # scaling
    x_scaler = None
    t_scaler = None
    if cfg["npe"].get("standardize_x", True):
        X, x_scaler = _standardize(X)
    if cfg["npe"].get("standardize_theta", True):
        Theta, t_scaler = _standardize(Theta)

    # Ensure we have enough samples for meaningful training
    n = X.shape[0]
    min_required = 100  # Minimum for stable training
    if n < min_required:
        raise ValueError(
            f"Insufficient simulations: got {n}, need at least {min_required}\n"
            f"Increase n_sims in config file."
        )
    
    device = torch.device(args.device)
    flow_cfg = cfg["npe"].get("flow", {})
    flow_config = {
        "hidden_sizes": list(flow_cfg.get("hidden_sizes", cfg["npe"].get("hidden_sizes", [128, 128]))),
        "num_layers": int(flow_cfg.get("num_layers", 4)),
        "num_bins": int(flow_cfg.get("num_bins", 8)),
        "tail_bound": float(flow_cfg.get("tail_bound", 3.0)),
        "min_bin_width": float(flow_cfg.get("min_bin_width", 1e-3)),
        "min_bin_height": float(flow_cfg.get("min_bin_height", 1e-3)),
        "min_derivative": float(flow_cfg.get("min_derivative", 1e-3)),
        "dropout": float(flow_cfg.get("dropout", cfg["npe"].get("dropout", 0.2))),
    }

    _, density_estimator = train_npe_sbi(
        theta=torch.from_numpy(Theta).to(device),
        x=torch.from_numpy(X).to(device),
        flow_config=flow_config,
        train_config=cfg["npe"],
        device=str(device),
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": density_estimator.state_dict(),
            "x_dim": X.shape[1],
            "theta_dim": Theta.shape[1],
            "model_type": "sbi_nsf",
            "flow_config": flow_config,
            "theta_keys": theta_keys,
            "pop_order": pop_order,
            "x_scaler": x_scaler,
            "theta_scaler": t_scaler,
            "config_path": str(args.config),
            "n_posterior_samples": int(cfg["npe"].get("n_posterior_samples", 50000)),
            "prior_bounds": (-10.0, 10.0),
        },
        args.out,
    )

    print(f"Saved best model to {args.out}")


if __name__ == "__main__":
    main()
