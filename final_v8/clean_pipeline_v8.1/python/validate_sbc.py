#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from npe_demography.nsf import FlowConfig, NeuralSplineFlow
from npe_demography.transforms import inverse_transform_theta_vector


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulation-Based Calibration using trained NSF model.")
    p.add_argument("--model", required=True, help="Trained NSF model checkpoint")
    p.add_argument("--simulations", required=True, help="Simulation dataset (.npz)")
    p.add_argument("--out", default="results/sbc_results.npz", help="Output .npz path")
    p.add_argument("--n-sbc", type=int, default=200, help="Number of SBC draws")
    p.add_argument("--n-post", type=int, default=2000, help="Posterior samples per SBC draw")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def _load_flow(ckpt: dict, device: torch.device) -> NeuralSplineFlow:
    flow_cfg = ckpt.get("flow_config", {})
    flow_config = FlowConfig(
        hidden_sizes=list(flow_cfg.get("hidden_sizes", [256, 256])),
        num_layers=int(flow_cfg.get("num_layers", 6)),
        num_bins=int(flow_cfg.get("num_bins", 8)),
        tail_bound=float(flow_cfg.get("tail_bound", 3.0)),
        min_bin_width=float(flow_cfg.get("min_bin_width", 1e-3)),
        min_bin_height=float(flow_cfg.get("min_bin_height", 1e-3)),
        min_derivative=float(flow_cfg.get("min_derivative", 1e-3)),
    )
    model = NeuralSplineFlow(
        theta_dim=int(ckpt["theta_dim"]),
        context_dim=int(ckpt["x_dim"]),
        config=flow_config,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _apply_scaler(x: np.ndarray, scaler: dict | None) -> np.ndarray:
    if scaler is None:
        return x
    return (x - scaler["mean"]) / scaler["std"]


def _invert_scaler(x: np.ndarray, scaler: dict | None) -> np.ndarray:
    if scaler is None:
        return x
    return x * scaler["std"] + scaler["mean"]


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if ckpt.get("model_type") != "nsf":
        raise ValueError("Checkpoint is not a Neural Spline Flow model.")

    data = np.load(args.simulations, allow_pickle=True)
    X = data["X"].astype(np.float32)
    Theta = data["Theta"].astype(np.float32)
    theta_keys = tuple(str(x) for x in data["theta_keys"].tolist())
    if not theta_keys:
        raise ValueError("Simulation dataset missing theta_keys; cannot run SBC.")

    rng = np.random.default_rng(42)
    n_sbc = min(args.n_sbc, X.shape[0])
    idx = rng.choice(X.shape[0], size=n_sbc, replace=False)

    model = _load_flow(ckpt, device)
    x_scaler = ckpt.get("x_scaler")
    theta_scaler = ckpt.get("theta_scaler")

    ranks = np.zeros((n_sbc, Theta.shape[1]), dtype=np.int32)

    for i, sim_idx in enumerate(idx):
        x_obs = X[sim_idx]
        theta_true = Theta[sim_idx]

        x_obs_scaled = _apply_scaler(x_obs, x_scaler)
        x_t = torch.from_numpy(x_obs_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            theta_post = model.sample(args.n_post, x_t).cpu().numpy().astype(np.float32)

        theta_post = _invert_scaler(theta_post, theta_scaler)
        theta_true_unscaled = theta_true

        theta_post_bio = inverse_transform_theta_vector(theta_post, theta_keys)
        theta_true_bio = inverse_transform_theta_vector(theta_true_unscaled, theta_keys)

        ranks[i] = np.sum(theta_post_bio < theta_true_bio, axis=0)

        if (i + 1) % 25 == 0:
            print(f"SBC progress: {i + 1}/{n_sbc}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        ranks=ranks,
        n_post=args.n_post,
        theta_keys=np.array(theta_keys, dtype=object),
    )
    print(f"SBC results saved to {out_path}")


if __name__ == "__main__":
    main()
