#!/usr/bin/env python3
"""Test script for individual populations configuration."""

import sys
from pathlib import Path
import numpy as np

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from npe_demography.config import load_config
from npe_demography.priors import sample_from_prior, build_theta_keys, theta_vector
from npe_demography.slim import render_slim_script
import tempfile


def test_config_loading():
    """Test loading the individual populations config."""
    print("=" * 60)
    print("TEST 1: Loading configuration")
    print("=" * 60)

    config_path = Path(__file__).parent / "config" / "config_individual_pops.yaml"
    cfg = load_config(str(config_path))

    print(f"✓ Config loaded successfully")
    print(f"  Project name: {cfg['project']['name']}")
    print(f"  Coverage threshold: {cfg['observed']['target_cov']}")
    print(f"  Template: {cfg['simulation']['slim_template']}")
    print(f"  Population order: {cfg['simulation']['pop_order']}")
    print()
    return cfg


def test_prior_sampling(cfg):
    """Test sampling from priors."""
    print("=" * 60)
    print("TEST 2: Prior sampling")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Sample 10 parameter sets
    print("Sampling 10 parameter sets...")
    samples = []
    for i in range(10):
        try:
            params = sample_from_prior(cfg, rng)
            samples.append(params)
            if i == 0:
                print(f"\nFirst sample:")
                # Print times
                time_keys = sorted([k for k in params.keys() if k.startswith("T_")])
                print("  Times:")
                for k in time_keys:
                    print(f"    {k}: {params[k]}")

                # Print sizes
                size_keys = sorted([k for k in params.keys() if k.startswith("N_")])
                print("  Sizes:")
                for k in size_keys:
                    print(f"    {k}: {params[k]}")

                # Print bottleneck params (first few)
                bn_keys = [k for k in params.keys() if k.startswith("BN_")][:6]
                if bn_keys:
                    print("  Bottleneck params (first 6):")
                    for k in bn_keys:
                        print(f"    {k}: {params[k]}")
        except Exception as e:
            print(f"✗ Failed to sample parameter set {i}: {e}")
            return False

    print(f"✓ Successfully sampled {len(samples)} parameter sets")
    print()
    return samples


def test_time_constraints(samples):
    """Test that time constraints are satisfied."""
    print("=" * 60)
    print("TEST 3: Time constraints validation")
    print("=" * 60)

    constraints = [
        ("T_P001", "<", "T_BG01"),
        ("T_BG01", "<", "T_MAJOR_SPLIT"),
        ("T_MAJOR_SPLIT", "<", "T_Sauva"),
        ("T_Sauva", "<", "T_BG07"),
        ("T_BG07", "<", "T_BG05_BG04"),
        ("T_MAJOR_SPLIT", "<", "T_Montsenymid"),
        ("T_Montsenymid", "<", "T_PYRENEES"),
        ("T_PYRENEES", "<", "T_Carlac"),
        ("T_Carlac", "<", "T_Conangles_Viros"),
        ("T_PYRENEES", "<", "T_Cimadal_Coscollet"),
    ]

    all_valid = True
    for i, params in enumerate(samples):
        for left, op, right in constraints:
            if op == "<":
                if not (params[left] < params[right]):
                    print(f"✗ Sample {i}: {left} ({params[left]}) < {right} ({params[right]}) FAILED")
                    all_valid = False

    if all_valid:
        print(f"✓ All {len(samples)} samples satisfy time constraints")
        print("  Constraints checked:")
        for left, op, right in constraints:
            print(f"    {left} {op} {right}")
    print()
    return all_valid


def test_theta_keys(cfg):
    """Test theta key building."""
    print("=" * 60)
    print("TEST 4: Theta keys")
    print("=" * 60)

    theta_keys = build_theta_keys(cfg)
    print(f"✓ Built theta keys: {len(theta_keys)} parameters")
    print("\nTheta keys:")
    for i, key in enumerate(theta_keys):
        if i < 20:  # Print first 20
            print(f"  {i+1}. {key}")
    if len(theta_keys) > 20:
        print(f"  ... and {len(theta_keys) - 20} more")
    print()
    return theta_keys


def test_slim_rendering(cfg, sample):
    """Test SLiM script rendering."""
    print("=" * 60)
    print("TEST 5: SLiM script rendering")
    print("=" * 60)

    template_path = Path(__file__).parent / cfg["simulation"]["slim_template"]
    if not template_path.exists():
        print(f"✗ Template not found: {template_path}")
        return False

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.slim', delete=False) as f:
            out_path = Path(f.name)

        script_path, scale = render_slim_script(
            template_path=template_path,
            out_path=out_path,
            cfg=cfg,
            params=sample
        )

        # Check that script was created and has content
        if script_path.exists() and script_path.stat().st_size > 0:
            print(f"✓ SLiM script rendered successfully")
            print(f"  Output: {script_path}")
            print(f"  Size: {script_path.stat().st_size} bytes")
            print(f"  Scale factor: {scale}")

            # Read and show first few lines
            lines = script_path.read_text().split('\n')
            print(f"\n  First 10 lines of rendered script:")
            for i, line in enumerate(lines[:10], 1):
                print(f"    {i:2d}: {line}")

            # Clean up
            out_path.unlink()
            return True
        else:
            print(f"✗ Script not created or empty")
            return False

    except Exception as e:
        print(f"✗ Failed to render SLiM script: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("INDIVIDUAL POPULATIONS CONFIGURATION TEST")
    print("=" * 60 + "\n")

    # Test 1: Load config
    cfg = test_config_loading()
    if not cfg:
        print("\n✗ Configuration loading failed. Aborting tests.")
        return 1

    # Test 2: Sample from priors
    samples = test_prior_sampling(cfg)
    if not samples:
        print("\n✗ Prior sampling failed. Aborting tests.")
        return 1

    # Test 3: Check time constraints
    if not test_time_constraints(samples):
        print("\n✗ Time constraints validation failed.")
        return 1

    # Test 4: Build theta keys
    theta_keys = test_theta_keys(cfg)
    if not theta_keys:
        print("\n✗ Theta key building failed.")
        return 1

    # Test 5: Render SLiM script
    if not test_slim_rendering(cfg, samples[0]):
        print("\n✗ SLiM script rendering failed.")
        return 1

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
