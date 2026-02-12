#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from npe_demography.sbi_backend import load_sbi_posterior
from npe_demography.config import load_config
from npe_demography.transforms import inverse_transform_theta_vector
from npe_demography.priors import build_size_anchors


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulation-Based Calibration using trained NSF model.")
    p.add_argument("--model", required=True, help="Trained NSF model checkpoint")
    p.add_argument("--simulations", required=True, help="Simulation dataset (.npz)")
    p.add_argument("--config", required=True, help="Config YAML (for size anchors)")
    p.add_argument("--out", default="results/sbc_results.npz", help="Output .npz path")
    p.add_argument("--n-sbc", type=int, default=200, help="Number of SBC draws")
    p.add_argument("--n-post", type=int, default=2000, help="Posterior samples per SBC draw")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


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
    if ckpt.get("model_type") != "sbi_nsf":
        raise ValueError("Checkpoint is not an sbi NSF model.")

    data = np.load(args.simulations, allow_pickle=True)
    X = data["X"].astype(np.float32)
    Theta = data["Theta"].astype(np.float32)
    theta_keys = tuple(str(x) for x in data["theta_keys"].tolist())
    size_anchors = build_size_anchors(load_config(args.config), theta_keys)
    if not theta_keys:
        raise ValueError("Simulation dataset missing theta_keys; cannot run SBC.")

    rng = np.random.default_rng(42)
    n_sbc = min(args.n_sbc, X.shape[0])
    idx = rng.choice(X.shape[0], size=n_sbc, replace=False)

    model = load_sbi_posterior(ckpt, device)
    x_scaler = ckpt.get("x_scaler")
    theta_scaler = ckpt.get("theta_scaler")

    ranks = np.zeros((n_sbc, Theta.shape[1]), dtype=np.int32)

    for i, sim_idx in enumerate(idx):
        x_obs = X[sim_idx]
        theta_true = Theta[sim_idx]

        x_obs_scaled = _apply_scaler(x_obs, x_scaler)
        x_t = torch.from_numpy(x_obs_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            theta_post = model.sample((args.n_post,), x=x_t).cpu().numpy().astype(np.float32)

        theta_post = _invert_scaler(theta_post, theta_scaler)
        theta_true_unscaled = theta_true

        theta_post_bio = inverse_transform_theta_vector(theta_post, theta_keys, size_anchors=size_anchors)
        theta_true_bio = inverse_transform_theta_vector(theta_true_unscaled, theta_keys, size_anchors=size_anchors)

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
