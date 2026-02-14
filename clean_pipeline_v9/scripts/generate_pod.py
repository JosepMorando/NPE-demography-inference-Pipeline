#!/usr/bin/env python3
"""
Generate Pseudo-Observed Data (POD) with known parameters.

Creates synthetic "observed" data to test if the NPE pipeline can
recover the true parameters. Uses the same scaling as production.

Parameters are drawn randomly from the prior distributions defined in the config.
Fixed parameters are included in the simulation but excluded from the
theta vector that NPE will try to recover.
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, 'python')

from npe_demography.config import load_config
# Added sample_from_prior to imports
from npe_demography.priors import build_theta_keys, is_fixed, sample_from_prior
from npe_demography.slim import render_slim_script, run_slim
from npe_demography.summaries import compute_summaries_from_trees


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-observed data for POD testing")
    parser.add_argument('--config', default='config/config_pod.yaml',
                        help='Configuration file (default: config/config_pod.yaml)')
    parser.add_argument('--out', default='observed_data',
                        help='Output directory (default: observed_data)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for POD generation (overrides config seed)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RNG
    # Use config seed if not provided in args, default to 42
    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(cfg.get("project", {}).get("seed", 42)) + 999  # Offset to differ from training
    
    rng = np.random.default_rng(seed)
    print(f"Using random seed: {seed}")

    # Sample true parameters from prior (respects fixed values & constraints)
    # This replaces the old hardcoded build_pod_true_params function
    print("Sampling POD parameters randomly from priors...")
    theta_true = sample_from_prior(cfg, rng)

    # Determine which params are free (will be inferred by NPE)
    theta_keys = build_theta_keys(cfg)

    # Identify fixed params for reporting
    pri = cfg["priors"]
    fixed_sizes = {name for name, scfg in pri["sizes"].items() if is_fixed(scfg)}

    max_scale = int(cfg["simulation"].get("max_scale_factor",
                        cfg["simulation"].get("scale_factor", 200)))

    print("=" * 70)
    print("GENERATING PSEUDO-OBSERVED DATA (POD)")
    print("=" * 70)
    print(f"\nAdaptive scaling: up to {max_scale}x")
    print(f"Free parameters for NPE: {len(theta_keys)}")
    print(f"Fixed parameters: {sorted(fixed_sizes) if fixed_sizes else 'none'}")
    print("Parameters are in BIOLOGICAL (unscaled) units.\n")

    # Check constraints (sanity check, though sample_from_prior should enforce them)
    print("Checking phylogenetic constraints...")
    ok = True
    if "T_P001" in theta_true:
        # Individual-populations model constraints
        ok &= theta_true['T_P001'] < theta_true['T_BG01'] < theta_true['T_MAJOR_SPLIT']
        ok &= theta_true['T_MAJOR_SPLIT'] < theta_true['T_Sauva'] < theta_true['T_BG07'] < theta_true['T_BG05_BG04']
        ok &= theta_true['T_MAJOR_SPLIT'] < theta_true['T_Montsenymid'] < theta_true['T_PYRENEES']
        ok &= theta_true['T_PYRENEES'] < theta_true['T_Carlac'] < theta_true['T_Conangles_Viros']
        ok &= theta_true['T_PYRENEES'] < theta_true['T_Cimadal_Coscollet']
    else:
        # Grouped-populations model constraints
        ok &= theta_true['T_BG01'] < theta_true['T_CORE']
        ok &= theta_true['T_CORE'] < theta_true['T_SOUTH_LOW'] < theta_true['T_EAST']
        ok &= theta_true['T_CORE'] < theta_true['T_SOUTH_MID']
        ok &= theta_true['T_CORE'] < theta_true['T_INT'] < theta_true['T_CENTRAL']
        ok &= theta_true['T_CORE'] < theta_true['T_INT'] < theta_true['T_PYRENEES']

    if not ok:
        print("ERROR: True parameters violate phylogenetic constraints!")
        sys.exit(1)
    print("Constraints satisfied\n")

    print("True parameters (biological units):")
    print("-" * 70)
    for k in sorted(theta_true.keys()):
        v = theta_true[k]
        status = "  [FIXED]" if k in fixed_sizes else ""
        if k.startswith('T_'):
            print(f"  {k:20s} = {v:>10.0f} gen{status}")
        elif k.startswith('N'):
            print(f"  {k:20s} = {v:>10,d}{status}")
        else:
            print(f"  {k:20s} = {v:>10.6f}")
    print()

    print(f"Parameters NPE will infer ({len(theta_keys)}):")
    print(f"  {', '.join(theta_keys)}")
    print()

    # Generate and run
    print("Step 1: Generating SLiM script (with adaptive scaling)...")
    trees_path = Path('/tmp/pod.trees')
    slim_script = Path('/tmp/pod.slim')

    _, scale_used = render_slim_script(
        cfg['simulation']['slim_template'],
        slim_script, cfg, theta_true
    )
    print(f"  Scale used: {scale_used}x")

    print("\nStep 2: Running SLiM simulation...")
    try:
        scale_used = run_slim(
            slim_binary=cfg['simulation']['slim_binary'],
            slim_script=slim_script,
            tree_out=trees_path,
            params=theta_true,
            cfg=cfg,
            timeout_s=300
        )
        print(f"  SLiM completed (scale={scale_used}x)")
        if not trees_path.exists():
            print(f"ERROR: TreeSeq file not created at {trees_path}")
            sys.exit(1)
        print(f"  TreeSeq file size: {trees_path.stat().st_size:,} bytes")
    except Exception as e:
        print(f"ERROR: SLiM simulation failed: {e}")
        sys.exit(1)

    print("\nStep 3: Computing summary statistics (mu scaled by {0}x)...".format(scale_used))
    pop_order = cfg.get('simulation', {}).get('pop_order', ['p1', 'p3', 'p4', 'p5', 'p7', 'p8'])

    try:
        x_obs, meta = compute_summaries_from_trees(
            trees_path,
            pop_order=pop_order,
            n_hap_per_pop=int(cfg['simulation']['samples_per_group_haploid']),
            cfg=cfg,
            rng=rng,  # Reuse the same RNG state
            scale_used=scale_used,
        )
        print(f"  Summary dimension: {len(x_obs)}")
        print(f"  Sites: {meta['n_sites']:,}, Variants: {meta['n_variants']:,}")
    except Exception as e:
        print(f"ERROR: Failed to compute summaries: {e}")
        sys.exit(1)

    print("\nStep 4: Saving POD data...")

    # Save only FREE parameter values as theta_true for POD recovery check
    theta_true_free = np.array([float(theta_true[k]) for k in theta_keys], dtype=np.float32)

    pod_full = out_dir / 'pod_observed.npz'
    np.savez_compressed(
        pod_full,
        x=x_obs.astype(np.float32),
        theta_true=theta_true_free,
        param_names=np.array(list(theta_keys), dtype=object),
    )
    print(f"  Full POD data: {pod_full}")

    pod_summaries = out_dir / 'pod_summaries.npz'
    np.savez_compressed(pod_summaries, x=x_obs.astype(np.float32))
    print(f"  POD summaries: {pod_summaries}")

    # Cleanup
    try:
        trees_path.unlink(missing_ok=True)
        slim_script.unlink(missing_ok=True)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("POD GENERATION COMPLETE!")
    print(f"  Free params saved: {len(theta_keys)}")
    print(f"  Fixed params used in sim but NOT in theta: {sorted(fixed_sizes)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
