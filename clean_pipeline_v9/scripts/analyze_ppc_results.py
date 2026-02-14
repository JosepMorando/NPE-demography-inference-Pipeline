#!/usr/bin/env python3
"""
Analyze Posterior Predictive Check (PPC) Results

PPC validates model fit by comparing observed data to simulations
from the posterior distribution. Good fit: p-values should be
roughly uniform [0,1] without extreme clustering near 0 or 1.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def build_argparser():
    p = argparse.ArgumentParser(description="Analyze PPC results")
    p.add_argument("--ppc", default="real_data_results/ppc_results.npz",
                   help="Path to PPC results NPZ file")
    p.add_argument("--out", default="real_data_results/ppc_analysis",
                   help="Output directory for plots and reports")
    return p


def categorize_statistics(n_stats, n_pops=11):
    """
    Categorize summary statistics based on typical pipeline structure.
    Adjust based on your actual summary statistic order.
    """
    categories = {}
    idx = 0

    # 1D SFS (bins per population)
    bins_per_pop = 10  # Typical default
    n_sfs = n_pops * bins_per_pop
    categories['SFS'] = list(range(idx, idx + n_sfs))
    idx += n_sfs

    # Heterozygosity (one per population)
    categories['Heterozygosity'] = list(range(idx, idx + n_pops))
    idx += n_pops

    # Pairwise Fst (n_pops choose 2)
    n_fst = (n_pops * (n_pops - 1)) // 2
    categories['Fst'] = list(range(idx, idx + n_fst))
    idx += n_fst

    # Pairwise Dxy
    categories['Dxy'] = list(range(idx, idx + n_fst))
    idx += n_fst

    # Any remaining statistics
    if idx < n_stats:
        categories['Other'] = list(range(idx, n_stats))

    return categories


def analyze_ppc(ppc_path: str, out_dir: str):
    """Comprehensive PPC analysis with diagnostics and visualizations."""

    # Load results
    print("="*70)
    print("POSTERIOR PREDICTIVE CHECK (PPC) ANALYSIS")
    print("="*70)
    print(f"\nLoading: {ppc_path}")

    ppc = np.load(ppc_path)
    x_obs = ppc['x_obs']
    x_sim = ppc['x_sim']
    ppc_pvals = ppc['ppc_pvals']
    ppc_means = ppc['ppc_means']

    n_ppc, n_stats = x_sim.shape
    print(f"  - Posterior predictive simulations: {n_ppc}")
    print(f"  - Summary statistics: {n_stats}")

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. P-value Distribution Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("1. P-VALUE DISTRIBUTION")
    print("="*70)

    # Overall p-value diagnostics
    pval_median = np.median(ppc_pvals)
    pval_mean = np.mean(ppc_pvals)
    pval_std = np.std(ppc_pvals)

    # Count extreme p-values
    n_extreme_low = np.sum(ppc_pvals < 0.05)
    n_extreme_high = np.sum(ppc_pvals > 0.95)
    n_extreme_total = n_extreme_low + n_extreme_high
    pct_extreme = 100 * n_extreme_total / n_stats

    print(f"\nP-value summary:")
    print(f"  Mean:   {pval_mean:.3f}  (ideal: ~0.5)")
    print(f"  Median: {pval_median:.3f}  (ideal: ~0.5)")
    print(f"  Std:    {pval_std:.3f}")
    print(f"\nExtreme p-values:")
    print(f"  p < 0.05:  {n_extreme_low:4d} ({100*n_extreme_low/n_stats:.1f}%)")
    print(f"  p > 0.95:  {n_extreme_high:4d} ({100*n_extreme_high/n_stats:.1f}%)")
    print(f"  Total:     {n_extreme_total:4d} ({pct_extreme:.1f}%)")
    print(f"\n  Expected under good fit: ~10% extreme")

    # Interpretation
    print(f"\n{'Interpretation:':<20}", end='')
    if pct_extreme < 15 and 0.4 < pval_mean < 0.6:
        print("✓ EXCELLENT - Model fits data well")
    elif pct_extreme < 25 and 0.3 < pval_mean < 0.7:
        print("✓ GOOD - Acceptable model fit")
    elif pct_extreme < 35:
        print("⚠ MARGINAL - Some statistics poorly fit")
    else:
        print("✗ POOR - Model does not fit data well")

    # ========================================================================
    # 2. Identify Problematic Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("2. PROBLEMATIC STATISTICS (p < 0.05 or p > 0.95)")
    print("="*70)

    extreme_idx = np.where((ppc_pvals < 0.05) | (ppc_pvals > 0.95))[0]

    if len(extreme_idx) > 0:
        print(f"\nFound {len(extreme_idx)} statistics with extreme p-values:\n")
        print(f"{'Index':<8} {'P-value':<10} {'Observed':<12} {'Pred Mean':<12} {'Status'}")
        print("-"*70)

        for idx in extreme_idx[:50]:  # Show first 50
            status = "Too low" if ppc_pvals[idx] < 0.05 else "Too high"
            print(f"{idx:<8} {ppc_pvals[idx]:<10.4f} {x_obs[idx]:<12.4f} "
                  f"{ppc_means[idx]:<12.4f} {status}")

        if len(extreme_idx) > 50:
            print(f"\n... and {len(extreme_idx) - 50} more")
    else:
        print("\n✓ No statistics with extreme p-values!")

    # ========================================================================
    # 3. Category-wise Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("3. CATEGORY-WISE ANALYSIS")
    print("="*70)

    categories = categorize_statistics(n_stats)

    print(f"\n{'Category':<20} {'N Stats':<10} {'Mean p-val':<12} "
          f"{'% Extreme':<12} {'Status'}")
    print("-"*70)

    for cat_name, indices in categories.items():
        if not indices:
            continue
        cat_pvals = ppc_pvals[indices]
        cat_mean = np.mean(cat_pvals)
        cat_extreme = 100 * np.sum((cat_pvals < 0.05) | (cat_pvals > 0.95)) / len(indices)

        status = "✓" if cat_extreme < 20 and 0.3 < cat_mean < 0.7 else "⚠"
        print(f"{cat_name:<20} {len(indices):<10} {cat_mean:<12.3f} "
              f"{cat_extreme:<12.1f} {status}")

    # ========================================================================
    # 4. Visualizations
    # ========================================================================
    print("\n" + "="*70)
    print("4. GENERATING VISUALIZATIONS")
    print("="*70)

    # Figure 1: P-value histogram
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.hist(ppc_pvals, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axhline(n_stats/20, color='red', linestyle='--',
                label='Expected (uniform)')
    ax1.set_xlabel('P-value', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'PPC P-value Distribution\n'
                  f'Mean={pval_mean:.3f}, {pct_extreme:.1f}% extreme',
                  fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    fig1_path = out_path / "ppc_pvalue_histogram.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig1_path}")

    # Figure 2: Observed vs Predicted scatter
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))

    # Color points by p-value extremeness
    colors = np.where((ppc_pvals < 0.05) | (ppc_pvals > 0.95), 'red', 'steelblue')

    ax2.scatter(ppc_means, x_obs, c=colors, alpha=0.5, s=20)

    # 1:1 line
    min_val = min(ppc_means.min(), x_obs.min())
    max_val = max(ppc_means.max(), x_obs.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--',
             label='Perfect fit (1:1 line)')

    ax2.set_xlabel('Predicted (Posterior Mean)', fontsize=12)
    ax2.set_ylabel('Observed', fontsize=12)
    ax2.set_title('Observed vs Posterior Predictive Mean\n'
                  f'(Red = extreme p-values)', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig2_path = out_path / "ppc_obs_vs_pred.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig2_path}")

    # Figure 3: P-value vs statistic index
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 6))

    # Plot all p-values
    ax3.scatter(range(n_stats), ppc_pvals, c=colors, alpha=0.6, s=10)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(0.05, color='red', linestyle='--', alpha=0.3, label='p=0.05/0.95')
    ax3.axhline(0.95, color='red', linestyle='--', alpha=0.3)

    # Mark category boundaries
    for cat_name, indices in categories.items():
        if indices:
            ax3.axvline(min(indices), color='black', linestyle=':', alpha=0.3)
            mid = (min(indices) + max(indices)) / 2
            ax3.text(mid, 1.02, cat_name, ha='center', fontsize=9, rotation=0)

    ax3.set_xlabel('Statistic Index', fontsize=12)
    ax3.set_ylabel('P-value', fontsize=12)
    ax3.set_title('PPC P-values by Statistic Type', fontsize=14)
    ax3.set_ylim(-0.05, 1.1)
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig3_path = out_path / "ppc_pvalues_by_index.png"
    fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig3_path}")

    # Figure 4: Posterior predictive intervals
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (cat_name, indices) in enumerate(list(categories.items())[:4]):
        ax = axes[i]
        if not indices:
            continue

        # Get subset of statistics
        subset = indices[:min(50, len(indices))]  # Max 50 stats per plot

        # Calculate percentiles
        for j, stat_idx in enumerate(subset):
            sim_vals = x_sim[:, stat_idx]
            percentiles = np.percentile(sim_vals, [2.5, 25, 50, 75, 97.5])

            # Plot credible interval
            ax.plot([j, j], [percentiles[0], percentiles[4]],
                   'lightblue', linewidth=2, alpha=0.5)
            ax.plot([j, j], [percentiles[1], percentiles[3]],
                   'steelblue', linewidth=4, alpha=0.7)

            # Plot observed value
            color = 'red' if (ppc_pvals[stat_idx] < 0.05 or
                            ppc_pvals[stat_idx] > 0.95) else 'green'
            ax.scatter(j, x_obs[stat_idx], c=color, s=30, zorder=10,
                      marker='o', edgecolors='black', linewidths=0.5)

        ax.set_xlabel(f'{cat_name} Statistic Index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{cat_name} (showing {len(subset)} stats)', fontsize=11)
        ax.grid(alpha=0.3)

    plt.suptitle('Posterior Predictive Credible Intervals\n'
                 'Dark blue=50% CI, Light blue=95% CI, Points=Observed (green=good, red=poor)',
                 fontsize=12)
    plt.tight_layout()

    fig4_path = out_path / "ppc_credible_intervals.png"
    fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig4_path}")

    plt.close('all')

    # ========================================================================
    # 5. Save Summary Report
    # ========================================================================
    report_path = out_path / "ppc_summary_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("POSTERIOR PREDICTIVE CHECK (PPC) SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Input file: {ppc_path}\n")
        f.write(f"Number of posterior predictive simulations: {n_ppc}\n")
        f.write(f"Number of summary statistics: {n_stats}\n\n")

        f.write("P-VALUE DISTRIBUTION:\n")
        f.write(f"  Mean:   {pval_mean:.4f}\n")
        f.write(f"  Median: {pval_median:.4f}\n")
        f.write(f"  Std:    {pval_std:.4f}\n\n")

        f.write("EXTREME P-VALUES:\n")
        f.write(f"  p < 0.05:  {n_extreme_low:4d} ({100*n_extreme_low/n_stats:.1f}%)\n")
        f.write(f"  p > 0.95:  {n_extreme_high:4d} ({100*n_extreme_high/n_stats:.1f}%)\n")
        f.write(f"  Total:     {n_extreme_total:4d} ({pct_extreme:.1f}%)\n\n")

        if pct_extreme < 15 and 0.4 < pval_mean < 0.6:
            status = "EXCELLENT"
        elif pct_extreme < 25 and 0.3 < pval_mean < 0.7:
            status = "GOOD"
        elif pct_extreme < 35:
            status = "MARGINAL"
        else:
            status = "POOR"

        f.write(f"OVERALL ASSESSMENT: {status}\n\n")

        f.write("CATEGORY BREAKDOWN:\n")
        for cat_name, indices in categories.items():
            if not indices:
                continue
            cat_pvals = ppc_pvals[indices]
            cat_mean = np.mean(cat_pvals)
            cat_extreme = 100 * np.sum((cat_pvals < 0.05) |
                                      (cat_pvals > 0.95)) / len(indices)
            f.write(f"  {cat_name:20s}: mean p={cat_mean:.3f}, "
                   f"{cat_extreme:.1f}% extreme\n")

    print(f"  ✓ Saved: {report_path}")

    print("\n" + "="*70)
    print("PPC ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {out_dir}/")
    print("\nGenerated files:")
    print("  - ppc_pvalue_histogram.png")
    print("  - ppc_obs_vs_pred.png")
    print("  - ppc_pvalues_by_index.png")
    print("  - ppc_credible_intervals.png")
    print("  - ppc_summary_report.txt")


if __name__ == "__main__":
    args = build_argparser().parse_args()
    analyze_ppc(args.ppc, args.out)
