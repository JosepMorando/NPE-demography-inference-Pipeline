#!/usr/bin/env python3
"""Test parameter transformations to verify correctness."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npe_demography.transforms import (
    transform_to_unconstrained,
    inverse_transform_from_unconstrained,
    transform_theta_vector,
    inverse_transform_theta_vector,
    validate_biological_params
)

def test_single_param_roundtrip():
    """Test that transform → inverse_transform is identity."""
    print("=" * 70)
    print("TEST 1: Single Parameter Roundtrip")
    print("=" * 70)
    
    # Test parameters in biological space
    bio_params = {
        "N_BG01": 50000,
        "N_SOUTH_LOW": 25000,
        "T_CORE": 7000,
        "T_BG01": 1000,
        "BN_TIME_FRAC": 0.4,
        "BN_SIZE_FRAC": 0.15,
        "BN_DUR": 50
    }
    
    theta_keys = tuple(bio_params.keys())
    
    # Forward transform
    unconstrained = transform_to_unconstrained(bio_params, theta_keys)
    
    # Inverse transform
    recovered = inverse_transform_from_unconstrained(unconstrained, theta_keys)
    
    # Check roundtrip
    print("\nOriginal → Transform → Inverse:")
    all_close = True
    for key in theta_keys:
        orig = bio_params[key]
        rec = recovered[key]
        close = np.isclose(orig, rec, rtol=1e-6)
        all_close = all_close and close
        status = "✓" if close else "✗"
        print(f"  {status} {key:20s}: {orig:12.4f} → {rec:12.4f}  (diff: {abs(orig-rec):.2e})")
    
    if all_close:
        print("\n✅ PASSED: Roundtrip preserves all parameters")
    else:
        print("\n❌ FAILED: Some parameters not preserved")
        return False
    
    return True


def test_batch_transform():
    """Test batch transformation of theta vectors."""
    print("\n" + "=" * 70)
    print("TEST 2: Batch Theta Vector Transform")
    print("=" * 70)
    
    theta_keys = ("N_BG01", "T_CORE", "BN_TIME_FRAC", "BN_DUR")
    
    # Create batch of biological parameters
    n_samples = 100
    theta_bio = np.array([
        [50000 + 1000*i, 7000 + 100*i, 0.3 + 0.001*i, 50 + i]
        for i in range(n_samples)
    ], dtype=np.float32)
    
    print(f"\nBatch size: {n_samples}")
    print(f"Parameters: {theta_keys}")
    
    # Transform
    theta_unconstrained = transform_theta_vector(theta_bio, theta_keys)
    
    # Inverse
    theta_recovered = inverse_transform_theta_vector(theta_unconstrained, theta_keys)
    
    # Check
    max_diff = np.max(np.abs(theta_bio - theta_recovered))
    all_close = np.allclose(theta_bio, theta_recovered, rtol=1e-6)
    
    print(f"\nMax difference after roundtrip: {max_diff:.2e}")
    
    if all_close:
        print("✅ PASSED: Batch transform preserves all samples")
    else:
        print(f"❌ FAILED: Some samples not preserved (max diff: {max_diff})")
        return False
    
    return True


def test_constraint_enforcement():
    """Test that inverse transform enforces constraints."""
    print("\n" + "=" * 70)
    print("TEST 3: Constraint Enforcement")
    print("=" * 70)
    
    theta_keys = ("N_BG01", "T_CORE", "BN_TIME_FRAC", "BN_DUR")
    
    # Sample from wide range in unconstrained space (including extreme values)
    n_samples = 1000
    np.random.seed(42)
    
    # Unconstrained samples (normal distribution centered at 0)
    theta_unconstrained = np.random.randn(n_samples, len(theta_keys)).astype(np.float32)
    theta_unconstrained *= 5.0  # Scale to get wide range
    
    print(f"\nGenerated {n_samples} samples in unconstrained space")
    print(f"Range: [{theta_unconstrained.min():.2f}, {theta_unconstrained.max():.2f}]")
    
    # Transform to biological space
    theta_bio = inverse_transform_theta_vector(theta_unconstrained, theta_keys)
    
    # Check constraints
    print("\nChecking constraints:")
    
    # Population sizes must be positive
    n_bg01 = theta_bio[:, 0]
    print(f"  N_BG01: min={n_bg01.min():.2f}, max={n_bg01.max():.2f}")
    if np.all(n_bg01 > 0):
        print("    ✓ All positive")
    else:
        print(f"    ✗ Found {np.sum(n_bg01 <= 0)} non-positive values")
        return False
    
    # Times must be positive
    t_core = theta_bio[:, 1]
    print(f"  T_CORE: min={t_core.min():.2f}, max={t_core.max():.2f}")
    if np.all(t_core > 0):
        print("    ✓ All positive")
    else:
        print(f"    ✗ Found {np.sum(t_core <= 0)} non-positive values")
        return False
    
    # Fractions must be in [0, 1]
    bn_frac = theta_bio[:, 2]
    print(f"  BN_TIME_FRAC: min={bn_frac.min():.4f}, max={bn_frac.max():.4f}")
    if np.all((bn_frac >= 0) & (bn_frac <= 1)):
        print("    ✓ All in [0, 1]")
    else:
        print(f"    ✗ Found {np.sum((bn_frac < 0) | (bn_frac > 1))} out-of-bounds values")
        return False
    
    # Durations must be positive
    bn_dur = theta_bio[:, 3]
    print(f"  BN_DUR: min={bn_dur.min():.2f}, max={bn_dur.max():.2f}")
    if np.all(bn_dur > 0):
        print("    ✓ All positive")
    else:
        print(f"    ✗ Found {np.sum(bn_dur <= 0)} non-positive values")
        return False
    
    print("\n✅ PASSED: All constraints enforced")
    return True


def test_validation():
    """Test parameter validation function."""
    print("\n" + "=" * 70)
    print("TEST 4: Parameter Validation")
    print("=" * 70)
    
    theta_keys = ("N_BG01", "T_CORE", "BN_TIME_FRAC")
    
    # Valid parameters
    valid_params = {
        "N_BG01": 50000,
        "T_CORE": 7000,
        "BN_TIME_FRAC": 0.4
    }
    
    try:
        validate_biological_params(valid_params, theta_keys)
        print("✓ Valid parameters accepted")
    except ValueError as e:
        print(f"✗ Valid parameters rejected: {e}")
        return False
    
    # Invalid: negative Ne
    invalid_params_1 = {**valid_params, "N_BG01": -5000}
    try:
        validate_biological_params(invalid_params_1, theta_keys)
        print("✗ Negative Ne not caught")
        return False
    except ValueError:
        print("✓ Negative Ne rejected")
    
    # Invalid: negative time
    invalid_params_2 = {**valid_params, "T_CORE": -1000}
    try:
        validate_biological_params(invalid_params_2, theta_keys)
        print("✗ Negative time not caught")
        return False
    except ValueError:
        print("✓ Negative time rejected")
    
    # Invalid: fraction > 1
    invalid_params_3 = {**valid_params, "BN_TIME_FRAC": 1.5}
    try:
        validate_biological_params(invalid_params_3, theta_keys)
        print("✗ Fraction > 1 not caught")
        return False
    except ValueError:
        print("✓ Fraction > 1 rejected")
    
    # Invalid: fraction < 0
    invalid_params_4 = {**valid_params, "BN_TIME_FRAC": -0.1}
    try:
        validate_biological_params(invalid_params_4, theta_keys)
        print("✗ Fraction < 0 not caught")
        return False
    except ValueError:
        print("✓ Fraction < 0 rejected")
    
    print("\n✅ PASSED: Validation correctly accepts/rejects parameters")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PARAMETER TRANSFORMATION TESTS - Version 7.0")
    print("=" * 70)
    
    tests = [
        ("Single Parameter Roundtrip", test_single_param_roundtrip),
        ("Batch Vector Transform", test_batch_transform),
        ("Constraint Enforcement", test_constraint_enforcement),
        ("Parameter Validation", test_validation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST CRASHED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        return 0
    else:
        n_failed = sum(1 for _, passed in results if not passed)
        print(f"{n_failed} TEST(S) FAILED ❌")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
