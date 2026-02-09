#!/usr/bin/env python3
"""
Memory-efficient simulation script with population-genetics scaling.

Scaling approach (from manuscript ABC-SMC pipeline):
  - Ne_slim = Ne / scale_factor
  - T_slim  = T  / scale_factor
  - R_slim  = R  * scale_factor
  - mu_overlay = mu * scale_factor
  Preserves theta = 4*Ne*mu and rho = 4*Ne*r

Results are written to memory-mapped files on disk to minimise RAM usage.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from npe_demography.config import load_config, ensure_dir
from npe_demography.io import read_groups_csv
from npe_demography.priors import sample_from_prior, theta_vector, build_theta_keys
from npe_demography.slim import render_slim_script, run_slim
from npe_demography.summaries import compute_summaries_from_trees
from npe_demography.transforms import transform_theta_vector


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate training simulations for NPE (with scaling).")
    p.add_argument("--config", required=True, help="Path to config YAML")
    p.add_argument("--n", type=int, default=None, help="Number of simulations (overrides config.npe.n_sims)")
    p.add_argument("--out", default="simulations/sim_data.npz", help="Output .npz path")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers (overrides config.npe.num_workers)")
    p.add_argument("--timeout", type=float, default=None, help="Optional SLiM timeout (seconds)")
    p.add_argument("--pilot-retries", type=int, default=None,
                   help="Retries for the pilot simulation if zero SNPs are produced.")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Offset added to simulation indices for seed generation. "
                        "Use when appending to existing sims to avoid seed overlap.")
    p.add_argument("--tmpdir", default=None, help="Override trees_tmpdir from config (for node-local scratch)")
    return p


def _one_sim(storage_idx: int, seed_idx: int, cfg: Dict[str, Any], pop_order: List[str],
             theta_keys: Tuple[str, ...], base_seed: int, tmpdir: str,
             timeout: float | None) -> Tuple[int, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run one simulation: sample params -> scale -> SLiM -> overlay mutations -> summaries."""
    rng = np.random.default_rng(base_seed + seed_idx * 9973)

    params = sample_from_prior(cfg, rng)

    # Unique run IDs
    run_id = uuid.uuid4().hex[:12]
    trees_path = Path(tmpdir) / f"sim_{run_id}.trees"
    slim_script = Path(tmpdir) / f"sim_{run_id}.slim"

    # Render SLiM script (applies scaling internally for bottleneck/expansion/migration blocks)
    _, scale_used = render_slim_script(cfg["simulation"]["slim_template"], slim_script, cfg, params)

    # Run SLiM with scaled parameters
    scale_used = run_slim(
        slim_binary=str(cfg["simulation"]["slim_binary"]),
        slim_script=slim_script,
        tree_out=trees_path,
        params=params,
        cfg=cfg,
        timeout_s=timeout,
    )

    # Compute summaries (overlay mutations with mu * scale_used)
    x, meta = compute_summaries_from_trees(
        trees_path,
        pop_order=pop_order,
        n_hap_per_pop=int(cfg["simulation"]["samples_per_group_haploid"]),
        cfg=cfg,
        rng=rng,
        scale_used=scale_used,
    )

    # theta vector: biological parameters in UNCONSTRAINED space
    # This allows the NSF to learn without constraint violations
    theta_bio = theta_vector(params, theta_keys)
    theta = transform_theta_vector(theta_bio, theta_keys)

    # Cleanup temp files
    try:
        trees_path.unlink(missing_ok=True)
        slim_script.unlink(missing_ok=True)
    except Exception:
        pass

    meta_out = {"run_id": run_id, "sim_idx": storage_idx, "scale_used": scale_used, **meta}
    return storage_idx, x, theta, meta_out


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)

    # Allow tmpdir override for multi-node deployments
    if args.tmpdir is not None:
        cfg["simulation"]["trees_tmpdir"] = args.tmpdir

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing output file to avoid permission errors
    if out_path.exists():
        try:
            out_path.unlink()
            print(f"Removed existing output file: {out_path}")
        except PermissionError:
            print(f"WARNING: Cannot remove existing file {out_path}")
            print(f"         This may cause a permission error during save.")
            print(f"         Try: rm {out_path}")


    group_order, _ = read_groups_csv(cfg["observed"]["groups_csv"])

    pop_order = cfg.get("simulation", {}).get("pop_order", ["p1", "p3", "p4", "p5", "p7", "p8"])
    if len(group_order) != len(pop_order):
        raise ValueError(
            "groups_csv and simulation.pop_order must describe the same number of populations. "
            f"groups_csv has {len(group_order)} groups, pop_order has {len(pop_order)} entries."
        )

    # theta keys: only FREE (non-fixed) parameters — built dynamically from config
    theta_keys_t = build_theta_keys(cfg)

    n_sims = int(args.n if args.n is not None else cfg["npe"]["n_sims"])
    workers = int(args.workers if args.workers is not None else cfg["npe"]["num_workers"])

    tmpdir = str(cfg["simulation"].get("trees_tmpdir", "/dev/shm"))
    ensure_dir(tmpdir)

    base_seed = int(cfg.get("project", {}).get("seed", 42))
    seed_offset = int(args.seed_offset)
    pilot_retries = int(
        args.pilot_retries
        if args.pilot_retries is not None
        else cfg.get("simulation", {}).get("pilot_retries", 3)
    )

    max_scale = int(cfg["simulation"].get("max_scale_factor",
                        cfg["simulation"].get("scale_factor", 200)))
    print(f"Running {n_sims} simulations with {workers} workers")
    print(f"Adaptive scaling: up to {max_scale}x (capped per-sim by min_Ne / safe_min)")
    print(f"Free parameters for NPE: {len(theta_keys_t)} ({', '.join(theta_keys_t)})")
    print("Using memory-mapped arrays to minimise RAM usage")

    # Pilot simulation to determine array shapes
    print("Running pilot simulation to determine array dimensions...")
    pilot_meta = None
    for attempt in range(pilot_retries + 1):
        try:
            _, x_pilot, theta_pilot, pilot_meta = _one_sim(
                0,
                seed_offset + attempt,
                cfg,
                pop_order,
                theta_keys_t,
                base_seed,
                tmpdir,
                args.timeout,
            )
        except Exception as exc:
            if attempt >= pilot_retries:
                raise
            print(f"WARNING: Pilot simulation failed ({exc}). Retrying ({attempt + 1}/{pilot_retries})...")
            continue

        n_variants = int(pilot_meta.get("n_variants", 0)) if pilot_meta else 0
        if n_variants > 0:
            break
        if attempt >= pilot_retries:
            raise RuntimeError(
                "Pilot simulation produced zero SNPs after retries. "
                "Consider increasing mutation rate or adjusting simulation settings."
            )
        print(
            "WARNING: Pilot simulation produced zero SNPs. "
            f"Retrying ({attempt + 1}/{pilot_retries})..."
        )

    x_dim = len(x_pilot)
    theta_dim = len(theta_pilot)
    print(f"Summary dimension: {x_dim}, Parameter dimension: {theta_dim}")

    # Create temporary memory-mapped files
    tmpdir_path = Path(tmpdir)
    X_file = tmpdir_path / f"X_mmap_{uuid.uuid4().hex[:8]}.dat"
    Theta_file = tmpdir_path / f"Theta_mmap_{uuid.uuid4().hex[:8]}.dat"

    X_mmap = np.memmap(X_file, dtype='float32', mode='w+', shape=(n_sims, x_dim))
    Theta_mmap = np.memmap(Theta_file, dtype='float32', mode='w+', shape=(n_sims, theta_dim))

    # Write pilot result
    X_mmap[0] = x_pilot
    Theta_mmap[0] = theta_pilot

    metas = [None] * n_sims
    metas[0] = {"run_id": "pilot", "sim_idx": 0}
    
    # CRITICAL FIX: Track successful simulations to filter out failures
    success_mask = np.zeros(n_sims, dtype=bool)
    success_mask[0] = True  # Pilot succeeded

    completed = 1

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_one_sim, i, seed_offset + i, cfg, pop_order, theta_keys_t, base_seed, tmpdir, args.timeout)
            for i in range(1, n_sims)
        ]

        for fut in as_completed(futs):
            try:
                idx, x, theta, meta = fut.result()
                X_mmap[idx] = x
                Theta_mmap[idx] = theta
                metas[idx] = meta
                success_mask[idx] = True  # Mark as successful
            except Exception as e:
                print(f"WARNING: Simulation failed: {e}")
                continue

            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{n_sims} ({100*completed/n_sims:.1f}%)")

    n_success = success_mask.sum()
    n_failed = n_sims - n_success
    print(f"Progress: {n_success}/{n_sims} succeeded ({100*n_success/n_sims:.1f}%), {n_failed} failed")

    X_mmap.flush()
    Theta_mmap.flush()

    # Convert to regular arrays and save compressed
    # CRITICAL FIX: Only save successful simulations, filter out failed ones
    print(f"Filtering {n_success} successful simulations (discarding {n_failed} failures)...")
    X = np.array(X_mmap[success_mask], dtype=np.float32)
    Theta = np.array(Theta_mmap[success_mask], dtype=np.float32)

    del X_mmap
    del Theta_mmap
    
    # Filter metas to match successful simulations
    metas_success = [m for i, m in enumerate(metas) if success_mask[i]]

    np.savez_compressed(
        out_path,
        X=X,
        Theta=Theta,
        theta_keys=np.array(theta_keys_t, dtype=object),
        pop_order=np.array(pop_order, dtype=object),
        meta_json=json.dumps(metas_success),
        n_requested=n_sims,
        n_success=n_success,
        n_failed=n_failed,
    )

    # Clean up temp files
    try:
        X_file.unlink()
        Theta_file.unlink()
    except Exception:
        pass

    print(f"Saved simulations to {out_path}  (X: {X.shape}, Theta: {Theta.shape})")


if __name__ == "__main__":
    main()
