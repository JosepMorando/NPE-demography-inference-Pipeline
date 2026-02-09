#!/usr/bin/env python3
"""
Check if NPE successfully recovered the true POD parameters.

This compares the true parameters used to generate POD with the
posterior distribution inferred by NPE.
"""
import numpy as np
import json
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check POD parameter recovery")
    parser.add_argument('--true', default='pod_test/pod_observed.npz',
                       help='True parameters file')
    parser.add_argument('--posterior', default='pod_test/results/pod_posterior.npz',
                       help='Posterior samples file')
    parser.add_argument('--out', default='pod_test/results',
                       help='Output directory for plots')
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    
    print("=" * 80)
    print("POD PARAMETER RECOVERY CHECK")
    print("=" * 80)
    print()
    
    # Load POD true parameters
    pod_file = Path(args.true)
    if not pod_file.exists():
        print(f"✗ ERROR: POD file not found: {pod_file}")
        print("  Run generate_pod.py first!")
        sys.exit(1)
    
    pod_data = np.load(pod_file, allow_pickle=True)
    theta_true = pod_data['theta_true']
    param_names_true = list(pod_data['param_names'])
    
    print(f"Loaded true parameters from: {pod_file}")
    print(f"Parameters: {len(param_names_true)}")
    print()
    
    # Load posterior
    post_file = Path(args.posterior)
    if not post_file.exists():
        print(f"✗ ERROR: Posterior file not found: {post_file}")
        print("  Run inference first:")
        print("    python3 python/infer_posterior.py --model models/mdn_model.pt \\")
        print("      --obs observed_data/pod_summaries.npz --out results/pod_posterior.npz")
        sys.exit(1)
    
    post_data = np.load(post_file, allow_pickle=True)
    theta_post = post_data['theta']
    param_names_post = list(post_data['param_names'])
    
    print(f"Loaded posterior from: {post_file}")
    print(f"Posterior samples: {theta_post.shape[0]:,}")
    print()
    
    # Load summary statistics
    summary_file = Path(args.out) / 'pod_posterior_summary.json'
    if not summary_file.exists():
        print(f"✗ ERROR: Summary file not found: {summary_file}")
        print("  This should be created by infer_posterior.py")
        sys.exit(1)
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    # Check parameter recovery
    print("=" * 80)
    print("PARAMETER RECOVERY RESULTS")
    print("=" * 80)
    print()
    print(f"{'Parameter':<20} {'True Value':<12} {'Posterior':<30} {'Status'}")
    print(f"{'':20} {'':12} {'Median [95% CI]':<30}")
    print("-" * 80)
    
    coverage_count = 0
    bias_list = []
    rel_error_list = []
    
    for i, param in enumerate(param_names_true):
        true_val = theta_true[i]
        s = summary[param]
        
        median = s['median']
        ci_low = s['q025']
        ci_high = s['q975']
        
        # Check if true value is in 95% CI
        covered = ci_low <= true_val <= ci_high
        coverage_count += covered
        
        # Compute bias and relative error
        bias = median - true_val
        rel_error = abs(bias / true_val) if true_val != 0 else 0
        bias_list.append(bias)
        rel_error_list.append(rel_error)
        
        # Format output
        if param.startswith('N'):
            true_str = f"{true_val:>11,.0f}"
            med_str = f"{median:>8,.0f}"
            ci_str = f"[{ci_low:>8,.0f}, {ci_high:>8,.0f}]"
        else:
            true_str = f"{true_val:>11.3f}"
            med_str = f"{median:>8.3f}"
            ci_str = f"[{ci_low:>8.3f}, {ci_high:>8.3f}]"
        
        coverage_str = "✓ GOOD" if covered else "✗ MISS"
        
        # Add warning if bias is large
        if rel_error > 0.2:  # >20% relative error
            coverage_str += " (LARGE BIAS)"
        
        post_str = f"{med_str} {ci_str}"
        print(f"{param:<20} {true_str:<12} {post_str:<30} {coverage_str}")
    
    print("-" * 80)
    print()
    
    # Summary statistics
    total_params = len(param_names_true)
    coverage_pct = 100 * coverage_count / total_params
    mean_rel_error = np.mean(rel_error_list) * 100
    
    print("SUMMARY STATISTICS:")
    print(f"  Coverage: {coverage_count}/{total_params} ({coverage_pct:.1f}%)")
    print(f"  Target: ~95% for well-calibrated inference")
    print(f"  Mean relative error: {mean_rel_error:.1f}%")
    print()
    
    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if coverage_pct >= 90 and coverage_pct <= 100:
        print("✓ EXCELLENT: Coverage is near 95%, inference is well-calibrated!")
    elif coverage_pct >= 80:
        print("⚠ ACCEPTABLE: Coverage is reasonable but could be better.")
        print("  Consider generating more training simulations.")
    elif coverage_pct >= 70:
        print("⚠ WARNING: Coverage is low. Model may be overconfident.")
        print("  Recommendations:")
        print("    - Generate more training simulations (50k+)")
        print("    - Increase MDN complexity (more hidden units/components)")
        print("    - Check for bugs in summary statistic computation")
    else:
        print("✗ POOR: Coverage is very low. Model is not working properly!")
        print("  This suggests:")
        print("    - Model not learning from data")
        print("    - Summary statistics not informative")
        print("    - Bugs in pipeline")
        print("  Do NOT use this model for real data analysis!")
    
    print()
    
    if mean_rel_error < 10:
        print("✓ Relative errors are small, estimates are accurate.")
    elif mean_rel_error < 20:
        print("⚠ Moderate relative errors. Estimates have some bias.")
    else:
        print("✗ Large relative errors. Systematic bias detected!")
    
    print()
    print("=" * 80)
    
    # Create a simple visualization if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        print("\nGenerating visualization...")
        
        fig, axes = plt.subplots(4, 6, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(param_names_true):
            if i >= len(axes):
                break
            
            idx = param_names_post.index(param)
            post_vals = theta_post[:, idx]
            true_val = theta_true[i]
            
            # Plot posterior distribution
            axes[i].hist(post_vals, bins=50, density=True, alpha=0.7, 
                        color='skyblue', edgecolor='black', linewidth=0.5)
            
            # Add true value line
            axes[i].axvline(true_val, color='red', linewidth=2, 
                           label='True value', linestyle='-')
            
            # Add posterior median
            axes[i].axvline(summary[param]['median'], color='blue', 
                           linewidth=1.5, linestyle='--', label='Posterior median')
            
            # Add 95% CI
            axes[i].axvline(summary[param]['q025'], color='gray', 
                           linewidth=1, linestyle=':', alpha=0.7)
            axes[i].axvline(summary[param]['q975'], color='gray', 
                           linewidth=1, linestyle=':', alpha=0.7, label='95% CI')
            
            # Check if covered
            covered = summary[param]['q025'] <= true_val <= summary[param]['q975']
            title_color = 'green' if covered else 'red'
            
            axes[i].set_xlabel('Value', fontsize=8)
            axes[i].set_ylabel('Density', fontsize=8)
            axes[i].set_title(param, fontsize=10, color=title_color, fontweight='bold')
            axes[i].legend(fontsize=6, loc='upper right')
            axes[i].tick_params(labelsize=7)
        
        # Hide unused subplots
        for i in range(len(param_names_true), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'POD Parameter Recovery - Coverage: {coverage_pct:.1f}%', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out_file = Path(args.out) / 'pod_recovery.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to: {out_file}")
        
        # Also save a simple coverage plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        covered = []
        for i, param in enumerate(param_names_true):
            true_val = theta_true[i]
            s = summary[param]
            covered.append(s['q025'] <= true_val <= s['q975'])
        
        colors = ['green' if c else 'red' for c in covered]
        ax.bar(range(len(param_names_true)), [1]*len(param_names_true), color=colors, alpha=0.6)
        ax.set_xticks(range(len(param_names_true)))
        ax.set_xticklabels(param_names_true, rotation=45, ha='right')
        ax.set_ylabel('Covered by 95% CI')
        ax.set_ylim([0, 1.2])
        ax.axhline(0.95, color='blue', linestyle='--', label='Target (95%)')
        ax.set_title(f'Parameter Coverage: {coverage_count}/{total_params} ({coverage_pct:.1f}%)')
        ax.legend()
        
        plt.tight_layout()
        out_file2 = Path(args.out) / 'pod_coverage.png'
        plt.savefig(out_file2, dpi=300, bbox_inches='tight')
        print(f"✓ Saved coverage plot to: {out_file2}")
        
    except ImportError:
        print("\n⚠ matplotlib not available, skipping visualization")
    
    print()
    print("=" * 80)
    print("CHECK COMPLETE!")
    print("=" * 80)
    
    # Return exit code based on coverage
    if coverage_pct < 70:
        print("\n✗ FAILED: Poor coverage, do not use for real data!")
        sys.exit(1)
    elif coverage_pct < 85:
        print("\n⚠ WARNING: Acceptable but not ideal coverage")
        sys.exit(0)
    else:
        print("\n✓ SUCCESS: Good coverage, pipeline validated!")
        sys.exit(0)


if __name__ == "__main__":
    main()
