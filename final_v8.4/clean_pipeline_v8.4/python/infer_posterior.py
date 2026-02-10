#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch

from npe_demography.config import load_config
from npe_demography.nsf import FlowConfig, NeuralSplineFlow
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


def _extract_prior_bounds(cfg: Dict[str, object], param_names: List[str]) -> Dict[str, Tuple[float, float]]:
    priors = cfg.get("priors")
    if not isinstance(priors, dict):
        raise ValueError("Config missing 'priors' mapping.")

    times_cfg = priors.get("times", {})
    sizes_cfg = priors.get("sizes", {})
    if not isinstance(times_cfg, dict) or not isinstance(sizes_cfg, dict):
        raise ValueError("Config priors.times and priors.sizes must be mappings.")

    bounds: Dict[str, Tuple[float, float]] = {}
    for name in param_names:
        if name in times_cfg:
            t_cfg = times_cfg[name]
            bounds[name] = (float(t_cfg["min"]), float(t_cfg["max"]))
        elif name in sizes_cfg:
            s_cfg = sizes_cfg[name]
            if str(s_cfg.get("dist", "")).lower() == "fixed":
                continue
            if "min" in s_cfg and "max" in s_cfg:
                bounds[name] = (float(s_cfg["min"]), float(s_cfg["max"]))

    extras = priors.get("demography_extras", {})
    if isinstance(extras, dict) and bool(extras.get("enable", False)):
        bn = extras.get("bottleneck", {})
        if isinstance(bn, dict):
            bn_mode = str(bn.get("mode", "shared")).lower()
            time_frac = bn.get("time_fraction", {})
            size_frac = bn.get("size_fraction", {})
            dur_cfg = bn.get("duration_gens", {})
            if bn_mode == "shared":
                if "BN_TIME_FRAC" in param_names:
                    bounds["BN_TIME_FRAC"] = (float(time_frac["min"]), float(time_frac["max"]))
                if "BN_SIZE_FRAC" in param_names:
                    bounds["BN_SIZE_FRAC"] = (float(size_frac["min"]), float(size_frac["max"]))
                if "BN_DUR" in param_names:
                    bounds["BN_DUR"] = (float(dur_cfg["min"]), float(dur_cfg["max"]))
            elif bn_mode == "per_population":
                pops = ["BG01", "SOUTH_LOW", "SOUTH_MID", "EAST", "CENTRAL", "PYRENEES"]
                for pop in pops:
                    time_key = f"BN_TIME_FRAC_{pop}"
                    size_key = f"BN_SIZE_FRAC_{pop}"
                    dur_key = f"BN_DUR_{pop}"
                    if time_key in param_names:
                        bounds[time_key] = (float(time_frac["min"]), float(time_frac["max"]))
                    if size_key in param_names:
                        bounds[size_key] = (float(size_frac["min"]), float(size_frac["max"]))
                    if dur_key in param_names:
                        bounds[dur_key] = (float(dur_cfg["min"]), float(dur_cfg["max"]))

        ex = extras.get("expansion", {})
        if isinstance(ex, dict) and bool(ex.get("enable", True)):
            start_cfg = ex.get("start_fraction", {})
            rate_cfg = ex.get("rate", {})
            if "EXP_START_FRAC" in param_names:
                bounds["EXP_START_FRAC"] = (float(start_cfg["min"]), float(start_cfg["max"]))
            if "EXP_RATE" in param_names:
                bounds["EXP_RATE"] = (float(rate_cfg["min"]), float(rate_cfg["max"]))

        mig = extras.get("migration", {})
        if isinstance(mig, dict) and bool(mig.get("enable", False)):
            m_cfg = mig.get("m", {})
            start_cfg = mig.get("start_fraction", {})
            if "MIG_M" in param_names:
                bounds["MIG_M"] = (float(m_cfg["min"]), float(m_cfg["max"]))
            if "MIG_START_FRAC" in param_names:
                bounds["MIG_START_FRAC"] = (float(start_cfg["min"]), float(start_cfg["max"]))
    return bounds


def _mask_with_bounds(
    theta: np.ndarray,
    param_names: List[str],
    bounds: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    n_samples = theta.shape[0]
    mask = np.ones(n_samples, dtype=bool)
    for i, name in enumerate(param_names):
        if name not in bounds:
            continue
        min_val, max_val = bounds[name]
        mask &= (theta[:, i] >= min_val) & (theta[:, i] <= max_val)
    return mask


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Infer posterior from a trained NSF using observed summaries."
    )
    p.add_argument("--model", default="models/nsf_model.pt", help="Trained model .pt")
    p.add_argument("--obs", default="observed_data/observed_summaries.npz",
                    help="Observed summaries .npz (must contain x or x_obs)")
    p.add_argument(
        "--config",
        required=True,
        help="YAML config file for enforcing prior bounds during inference.",
    )
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


def _load_obs(path: str) -> tuple[np.ndarray, Optional[List[str]], Optional[str]]:
    obs = np.load(path, allow_pickle=True)
    if "x" in obs:
        x_obs = obs["x"].astype(np.float32)
    elif "x_obs" in obs:
        x_obs = obs["x_obs"].astype(np.float32)
    else:
        raise KeyError(f"Observed NPZ must contain 'x' or 'x_obs'. Keys found: {list(obs.keys())}")

    order_key = None
    order = None
    if "pop_order" in obs:
        order_key = "pop_order"
        order = [str(x) for x in obs["pop_order"].tolist()]
    elif "group_order" in obs:
        order_key = "group_order"
        order = [str(x) for x in obs["group_order"].tolist()]

    return x_obs, order, order_key


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

    if ckpt.get("model_type") != "nsf":
        raise ValueError("Checkpoint is not a Neural Spline Flow model.")

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
        theta_dim=theta_dim,
        context_dim=x_dim,
        config=flow_config,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_scaler = ckpt.get("x_scaler")
    theta_scaler = ckpt.get("theta_scaler")
    param_names = _infer_param_names(ckpt)

    ckpt_pop_order = ckpt.get("pop_order")
    if ckpt_pop_order is not None:
        ckpt_pop_order = [str(x) for x in ckpt_pop_order]

    x_obs, obs_order, obs_order_key = _load_obs(args.obs)
    if x_obs.ndim != 1:
        x_obs = x_obs.reshape(-1).astype(np.float32)
    if x_obs.shape[0] != x_dim:
        raise ValueError(f"Observed summary length {x_obs.shape[0]} != model x_dim {x_dim}.")
    if obs_order is not None and ckpt_pop_order is not None:
        if len(obs_order) != len(ckpt_pop_order):
            raise ValueError(
                "Observed population order length does not match model pop_order length. "
                f"Observed ({obs_order_key}) has {len(obs_order)} entries, "
                f"model pop_order has {len(ckpt_pop_order)}."
            )
        if obs_order != ckpt_pop_order:
            print("WARNING: Observed population order labels differ from model pop_order.")
            print(f"  Observed ({obs_order_key}): {obs_order}")
            print(f"  Model pop_order: {ckpt_pop_order}")
            print("  Ensure the ordering matches before trusting posterior results.")

    x_obs_scaled = _apply_scaler(x_obs, x_scaler)
    x_t = torch.from_numpy(x_obs_scaled).unsqueeze(0).to(device)

    n_samples = args.n if args.n is not None else int(ckpt.get("n_posterior_samples", 50000))
    print(f"Sampling {n_samples:,} posterior samples...")

    cfg = load_config(args.config)
    bounds = _extract_prior_bounds(cfg, param_names)
    if not bounds:
        raise ValueError(
            f"No matching prior bounds found in {args.config}; "
            "ensure priors.times and priors.sizes match the trained parameter names."
        )
    print(f"Applying prior bounds from {args.config}")

    accepted: List[np.ndarray] = []
    total_draws = 0
    remaining = n_samples
    max_draws = max(n_samples * 200, 200000)
    while remaining > 0:
        if total_draws >= max_draws:
            raise RuntimeError(
                "Failed to collect enough posterior samples within prior bounds. "
                "Consider increasing n or widening prior ranges."
            )
        batch_size = max(1000, remaining)
        total_draws += batch_size

        # Use vectorised NSF sampling (much faster than the per-sample for-loop)
        with torch.no_grad():
            theta_post_t = model.sample(batch_size, x_t)  # (n, theta_dim)

        theta_post = theta_post_t.cpu().numpy().astype(np.float32)

        # De-standardize from training space
        theta_post = _invert_scaler(theta_post, theta_scaler)

        # Transform from unconstrained to biological space
        theta_post = inverse_transform_theta_vector(theta_post, param_names)

        # Time ordering is enforced by cumulative-gap parameterization in transforms.

        mask = _mask_with_bounds(theta_post, param_names, bounds)
        mask &= _check_time_ordering(theta_post, param_names)
        theta_post = theta_post[mask]

        if theta_post.size == 0:
            continue

        accepted.append(theta_post)
        remaining = n_samples - sum(chunk.shape[0] for chunk in accepted)

    theta_post = np.concatenate(accepted, axis=0)[:n_samples]
    if total_draws > 0:
        acceptance = theta_post.shape[0] / total_draws
        print(f"Accepted {theta_post.shape[0]:,} / {total_draws:,} samples ({acceptance:.2%}).")

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
