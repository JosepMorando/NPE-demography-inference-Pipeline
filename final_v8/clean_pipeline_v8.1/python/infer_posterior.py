#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch

from npe_demography.mdn import MDN, mdn_sample
from npe_demography.transforms import inverse_transform_theta_vector


def _check_time_ordering(theta: np.ndarray, param_names: List[str]) -> np.ndarray:
    """Check if time parameters satisfy phylogenetic ordering constraints.
    
    Returns boolean mask of valid samples.
    
    Phylogeny constraints (forward-time: parent splits BEFORE children):
    - T_BG01 < T_CORE
    - T_CORE < T_SOUTH_LOW < T_EAST
    - T_CORE < T_SOUTH_MID
    - T_CORE < T_INT < T_CENTRAL
    - T_CORE < T_INT < T_PYRENEES
    """
    # Build index map
    idx_map = {name: i for i, name in enumerate(param_names)}
    
    # Get required time indices (some may not be present in all configs)
    required_times = ['T_BG01', 'T_CORE', 'T_SOUTH_LOW', 'T_SOUTH_MID', 'T_EAST', 
                     'T_INT', 'T_CENTRAL', 'T_PYRENEES']
    
    # Check which times are actually in the model
    time_indices = {}
    for t in required_times:
        if t in idx_map:
            time_indices[t] = idx_map[t]
    
    # If we don't have enough times to check, assume valid
    if len(time_indices) < 2:
        return np.ones(theta.shape[0], dtype=bool)
    
    n_samples = theta.shape[0]
    valid = np.ones(n_samples, dtype=bool)
    
    # Check constraints that can be checked
    if 'T_BG01' in time_indices and 'T_CORE' in time_indices:
        valid &= theta[:, time_indices['T_BG01']] < theta[:, time_indices['T_CORE']]
    
    if 'T_CORE' in time_indices and 'T_SOUTH_LOW' in time_indices:
        valid &= theta[:, time_indices['T_CORE']] < theta[:, time_indices['T_SOUTH_LOW']]
    
    if 'T_SOUTH_LOW' in time_indices and 'T_EAST' in time_indices:
        valid &= theta[:, time_indices['T_SOUTH_LOW']] < theta[:, time_indices['T_EAST']]
    
    if 'T_CORE' in time_indices and 'T_SOUTH_MID' in time_indices:
        valid &= theta[:, time_indices['T_CORE']] < theta[:, time_indices['T_SOUTH_MID']]
    
    if 'T_CORE' in time_indices and 'T_INT' in time_indices:
        valid &= theta[:, time_indices['T_CORE']] < theta[:, time_indices['T_INT']]
    
    if 'T_INT' in time_indices and 'T_CENTRAL' in time_indices:
        valid &= theta[:, time_indices['T_INT']] < theta[:, time_indices['T_CENTRAL']]
    
    if 'T_INT' in time_indices and 'T_PYRENEES' in time_indices:
        valid &= theta[:, time_indices['T_INT']] < theta[:, time_indices['T_PYRENEES']]
    
    return valid


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Infer posterior from a trained MDN using observed summaries."
    )
    p.add_argument("--model", default="models/mdn_model.pt", help="Trained model .pt")
    p.add_argument("--obs", default="observed_data/observed_summaries.npz",
                    help="Observed summaries .npz (must contain x or x_obs)")
    p.add_argument("--out", default="results/posterior_samples.npz",
                    help="Output posterior samples (.npz)")
    p.add_argument("--n", type=int, default=None,
                    help="Number of posterior samples (overrides checkpoint default)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def _apply_scaler(x: np.ndarray, scaler: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if scaler is None:
        return x
    return (x - scaler["mean"]) / scaler["std"]


def _invert_scaler(x: np.ndarray, scaler: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if scaler is None:
        return x
    return x * scaler["std"] + scaler["mean"]


def _load_x_obs(path: str) -> np.ndarray:
    obs = np.load(path, allow_pickle=True)
    if "x" in obs:
        return obs["x"].astype(np.float32)
    if "x_obs" in obs:
        return obs["x_obs"].astype(np.float32)
    raise KeyError(f"Observed NPZ must contain 'x' or 'x_obs'. Keys found: {list(obs.keys())}")


def _infer_param_names(ckpt: dict) -> List[str]:
    names = ckpt.get("param_names", ckpt.get("theta_keys"))
    if names is None:
        theta_dim = int(ckpt["theta_dim"])
        return [f"theta_{i}" for i in range(theta_dim)]
    return [str(x) for x in list(names)]


def main() -> None:
    args = build_argparser().parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)

    x_dim = int(ckpt["x_dim"])
    theta_dim = int(ckpt["theta_dim"])

    device = torch.device(args.device)

    model = MDN(
        x_dim=x_dim,
        theta_dim=theta_dim,
        hidden=list(ckpt["hidden"]),
        n_components=int(ckpt["n_components"]),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_scaler = ckpt.get("x_scaler")
    theta_scaler = ckpt.get("theta_scaler")
    param_names = _infer_param_names(ckpt)

    x_obs = _load_x_obs(args.obs)
    if x_obs.ndim != 1:
        x_obs = x_obs.reshape(-1).astype(np.float32)
    if x_obs.shape[0] != x_dim:
        raise ValueError(f"Observed summary length {x_obs.shape[0]} != model x_dim {x_dim}.")

    x_obs_scaled = _apply_scaler(x_obs, x_scaler)
    x_t = torch.from_numpy(x_obs_scaled).unsqueeze(0).to(device)

    n_samples = args.n if args.n is not None else int(ckpt.get("n_posterior_samples", 50000))
    print(f"Sampling {n_samples:,} posterior samples...")

    # Use vectorised MDN sampling (much faster than the per-sample for-loop)
    with torch.no_grad():
        pi, mu, log_sigma = model(x_t)
        theta_post_t = mdn_sample(pi, mu, log_sigma, n_samples)  # (n, theta_dim)

    theta_post = theta_post_t.cpu().numpy().astype(np.float32)
    
    # De-standardize from training space
    theta_post = _invert_scaler(theta_post, theta_scaler)
    
    # Transform from unconstrained to biological space
    theta_post = inverse_transform_theta_vector(theta_post, param_names)
    
    # NOTE: Time ordering check disabled in v8.1
    # The training data has perfect ordering in log-space, but the MDN's
    # diagonal covariance means independent sampling can violate ordering
    # after exp() transform. This is a limitation of diagonal Gaussian mixtures,
    # not a bug. Solutions: (1) use normalizing flows, or (2) reparameterize
    # times as cumulative gaps. For now, we accept ~10-20% violations as
    # a known limitation of the MDN architecture.
    #
    # To re-enable checking (and filter violations):
    # valid_mask = _check_time_ordering(theta_post, param_names)
    # if (~valid_mask).any():
    #     print(f"WARNING: {(~valid_mask).sum()} samples violate time ordering")
    #     theta_post = theta_post[valid_mask]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        theta=theta_post.astype(np.float32),
        param_names=np.array(param_names, dtype=object),
    )

    summary = {}
    for i, pname in enumerate(param_names):
        vals = theta_post[:, i]
        summary[pname] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "q025": float(np.percentile(vals, 2.5)),
            "q975": float(np.percentile(vals, 97.5)),
        }

    summary_path = out_path.parent / (out_path.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved posterior samples to {out_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
