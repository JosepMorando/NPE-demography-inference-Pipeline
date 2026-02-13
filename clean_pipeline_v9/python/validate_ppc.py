#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from npe_demography.config import load_config, ensure_dir
from npe_demography.sbi_backend import load_sbi_posterior
from npe_demography.priors import get_fixed_value, is_fixed
# --- CRITICAL FIX: Dynamic GENS extension ---
# Ensure simulation is long enough for the deepest split time in this sample
max_t_sample = max([float(v) for k,v in params.items() if k.startswith('T_')] + [0])
burnin_val = float(cfg['simulation'].get('burnin', 0))
# Use config GENS or (max_t + burnin + buffer), whichever is larger
config_gens = float(cfg['simulation'].get('gens', 0))
needed_gens = max_t_sample + burnin_val + 500
params['GENS'] = max(config_gens, needed_gens)
# ------------------------------------------
from npe_demography.slim import render_slim_script, run_slim
from npe_demography.summaries import compute_summaries_from_trees
from npe_demography.transforms import inverse_transform_theta_vector
from npe_demography.priors import build_size_anchors


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Posterior predictive checks using trained NSF model.")
    p.add_argument("--config", required=True, help="Config YAML")
    p.add_argument("--model", required=True, help="Trained NSF model checkpoint")
    p.add_argument("--obs", required=True, help="Observed summaries (.npz)")
    p.add_argument("--out", default="results/ppc_results.npz", help="Output .npz path")
    p.add_argument("--n-ppc", type=int, default=200, help="Number of posterior predictive sims")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--timeout", type=float, default=300, help="SLiM timeout per sim (seconds)")
    return p


def _apply_scaler(x: np.ndarray, scaler: dict | None) -> np.ndarray:
    if scaler is None:
        return x
    return (x - scaler["mean"]) / scaler["std"]


def _invert_scaler(x: np.ndarray, scaler: dict | None) -> np.ndarray:
    if scaler is None:
        return x
    return x * scaler["std"] + scaler["mean"]


def _load_obs(path: str) -> np.ndarray:
    obs = np.load(path, allow_pickle=True)
    if "x" in obs:
        return obs["x"].astype(np.float32)
    if "x_obs" in obs:
        return obs["x_obs"].astype(np.float32)
    raise KeyError(f"Observed NPZ must contain 'x' or 'x_obs'. Keys found: {list(obs.keys())}")


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    cfg = load_config(args.config)
    pop_order = cfg.get("simulation", {}).get("pop_order", ["p1", "p3", "p4", "p5", "p7", "p8"])
    n_hap = int(cfg["simulation"]["samples_per_group_haploid"])

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if ckpt.get("model_type") != "sbi_nsf":
        raise ValueError("Checkpoint is not an sbi NSF model.")

    model = load_sbi_posterior(ckpt, device)
    x_scaler = ckpt.get("x_scaler")
    theta_scaler = ckpt.get("theta_scaler")
    theta_keys = tuple(str(x) for x in ckpt.get("theta_keys", []))
    size_anchors = build_size_anchors(cfg, theta_keys)
    if not theta_keys:
        raise ValueError("Checkpoint missing theta_keys; cannot run PPC.")

    fixed_params: dict[str, float] = {}
    for name, scfg in cfg["priors"]["sizes"].items():
        if is_fixed(scfg):
            fixed_params[name] = float(get_fixed_value(scfg))

    extras = cfg["priors"].get("demography_extras", {})
    if not bool(extras.get("enable", False)):
        fixed_params.update(
            {
                "BN_TIME_FRAC": 0.0,
                "BN_SIZE_FRAC": 0.0,
                "BN_DUR": 0.0,
                "EXP_START_FRAC": 0.0,
                "EXP_RATE": 0.0,
                "MIG_M": 0.0,
                "MIG_START_FRAC": 0.0,
            }
        )
    else:
        bn = extras.get("bottleneck", {})
        bn_mode = str(bn.get("mode", "shared")).lower()
        if bn_mode == "per_population":
            pops = ["BG01", "SOUTH_LOW", "SOUTH_MID", "EAST", "CENTRAL", "PYRENEES"]
            for pop in pops:
                fixed_params.setdefault(f"BN_TIME_FRAC_{pop}", 0.0)
                fixed_params.setdefault(f"BN_SIZE_FRAC_{pop}", 0.0)
                fixed_params.setdefault(f"BN_DUR_{pop}", 0.0)
        else:
            fixed_params.setdefault("BN_TIME_FRAC", 0.0)
            fixed_params.setdefault("BN_SIZE_FRAC", 0.0)
            fixed_params.setdefault("BN_DUR", 0.0)

        ex = extras.get("expansion", {})
        if not bool(ex.get("enable", True)):
            fixed_params.setdefault("EXP_START_FRAC", 0.0)
            fixed_params.setdefault("EXP_RATE", 0.0)

        mig = extras.get("migration", {})
        if not bool(mig.get("enable", False)):
            fixed_params.setdefault("MIG_M", 0.0)
            fixed_params.setdefault("MIG_START_FRAC", 0.0)

    x_obs = _load_obs(args.obs)
    if x_obs.ndim != 1:
        x_obs = x_obs.reshape(-1).astype(np.float32)
    if x_obs.shape[0] != int(ckpt["x_dim"]):
        raise ValueError(f"Observed summary length {x_obs.shape[0]} != model x_dim {ckpt['x_dim']}.")

    x_obs_scaled = _apply_scaler(x_obs, x_scaler)
    x_t = torch.from_numpy(x_obs_scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        theta_post = model.sample((args.n_ppc,), x=x_t).cpu().numpy().astype(np.float32)

    theta_post = _invert_scaler(theta_post, theta_scaler)
    theta_post = inverse_transform_theta_vector(theta_post, theta_keys, size_anchors=size_anchors)

    tmpdir = str(cfg["simulation"].get("trees_tmpdir", "/dev/shm"))
    ensure_dir(tmpdir)

    rng = np.random.default_rng(123)
    summaries = []
    metas = []

    for i in range(args.n_ppc):
        params = {k: float(theta_post[i, j]) for j, k in enumerate(theta_keys)}
        params.update(fixed_params)
        run_id = f"ppc_{i:05d}"
        trees_path = Path(tmpdir) / f"{run_id}.trees"
        slim_script = Path(tmpdir) / f"{run_id}.slim"

        _, scale_used = render_slim_script(
            cfg["simulation"]["slim_template"],
            slim_script,
            cfg,
            params,
        )

        scale_used = run_slim(
            slim_binary=str(cfg["simulation"]["slim_binary"]),
            slim_script=slim_script,
            tree_out=trees_path,
            params=params,
            cfg=cfg,
            timeout_s=args.timeout,
        )

        x_sim, meta = compute_summaries_from_trees(
            trees_path,
            pop_order=pop_order,
            n_hap_per_pop=n_hap,
            cfg=cfg,
            rng=rng,
            scale_used=scale_used,
        )
        summaries.append(x_sim)
        metas.append(meta)

        trees_path.unlink(missing_ok=True)
        slim_script.unlink(missing_ok=True)

        if (i + 1) % 25 == 0:
            print(f"PPC progress: {i + 1}/{args.n_ppc}")

    x_sim = np.vstack(summaries)
    ppc_means = x_sim.mean(axis=0)
    ppc_pvals = np.mean(x_sim <= x_obs, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x_obs=x_obs,
        x_sim=x_sim,
        ppc_means=ppc_means,
        ppc_pvals=ppc_pvals,
        meta_json=json.dumps(metas),
    )
    print(f"PPC results saved to {out_path}")


if __name__ == "__main__":
    main()
