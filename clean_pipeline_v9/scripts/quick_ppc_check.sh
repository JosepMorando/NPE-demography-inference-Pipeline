#!/bin/bash
#
# Quick PPC check - terminal-based summary
#

set -euo pipefail

PPC_FILE="${1:-real_data_results/ppc_results.npz}"

if [[ ! -f "$PPC_FILE" ]]; then
  echo "ERROR: PPC file not found: $PPC_FILE"
  echo "Usage: $0 [ppc_results.npz]"
  exit 1
fi

python3 << EOF
import numpy as np

print("="*70)
print("QUICK PPC CHECK")
print("="*70)
print(f"\nFile: $PPC_FILE\n")

# Load data
ppc = np.load("$PPC_FILE")
pvals = ppc['ppc_pvals']
x_obs = ppc['x_obs']
x_sim = ppc['x_sim']
means = ppc['ppc_means']

n_ppc, n_stats = x_sim.shape

# Summary
print(f"Posterior predictive sims: {n_ppc}")
print(f"Summary statistics:        {n_stats}")

# P-value distribution
pval_mean = np.mean(pvals)
pval_median = np.median(pvals)

# Extreme counts
n_low = np.sum(pvals < 0.05)
n_high = np.sum(pvals > 0.95)
n_extreme = n_low + n_high
pct_extreme = 100 * n_extreme / n_stats

print("\n" + "-"*70)
print("P-VALUE DISTRIBUTION")
print("-"*70)
print(f"Mean:   {pval_mean:.3f}  {'✓ Good' if 0.4 < pval_mean < 0.6 else '⚠ Check'}")
print(f"Median: {pval_median:.3f}  {'✓ Good' if 0.4 < pval_median < 0.6 else '⚠ Check'}")

print("\n" + "-"*70)
print("EXTREME P-VALUES")
print("-"*70)
print(f"p < 0.05:  {n_low:4d}  ({100*n_low/n_stats:5.1f}%)")
print(f"p > 0.95:  {n_high:4d}  ({100*n_high/n_stats:5.1f}%)")
print(f"Total:     {n_extreme:4d}  ({pct_extreme:5.1f}%)")
print(f"\nExpected if model fits well: ~10% extreme")

# Overall assessment
print("\n" + "="*70)
print("ASSESSMENT")
print("="*70)

if pct_extreme < 15 and 0.4 < pval_mean < 0.6:
    print("✓✓ EXCELLENT - Model fits data very well!")
    print("   Proceed with confidence.")
elif pct_extreme < 25 and 0.3 < pval_mean < 0.7:
    print("✓  GOOD - Acceptable model fit")
    print("   Minor discrepancies, but generally reliable.")
elif pct_extreme < 35:
    print("⚠  MARGINAL - Some statistics poorly fit")
    print("   Consider: more simulations, different priors, or model refinement.")
else:
    print("✗  POOR - Model does not fit data well")
    print("   Action needed: check priors, increase training data, or revise model.")

print("\nFor detailed analysis with plots, run:")
print("  python3 scripts/analyze_ppc_results.py --ppc $PPC_FILE")
print("="*70)
EOF
