#!/bin/bash
#
# Production Analysis Workflow
# Full demographic inference pipeline for real Pool-seq data
#

set -euo pipefail

echo "======================================================================"
echo "PRODUCTION ANALYSIS WORKFLOW"
echo "======================================================================"
echo ""
echo "This runs the full demographic inference pipeline on your real data."
echo ""
echo "IMPORTANT: Have you validated the pipeline with POD testing?"
echo "If not, run: bash scripts/run_pod_test.sh first!"
echo ""
read -p "Continue with production analysis? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Configuration
CONFIG="config/config_production.yaml"

# Create output directories
mkdir -p observed_data simulations models results

echo ""
echo "Step 1/4: Computing observed summary statistics..."
echo "----------------------------------------------------------------------"
echo "Extracting statistics from Pool-seq data..."
Rscript r/compute_observed_summaries.R \
  --pooldata observed_data/Pooldata_demography.RData \
  --object filt.pooldata \
  --groups config/groups_6clusters.csv \
  --target_cov 20 \
  --out observed_data
echo ""

echo "Step 2/4: Generating training simulations (50,000 sims)..."
echo "----------------------------------------------------------------------"
echo "This will take 4-8 hours with 20 workers..."
echo "Started at: $(date)"
python3 python/simulate.py \
  --config "$CONFIG" \
  --out simulations/sim_data.npz \
  --workers 20
echo "Finished at: $(date)"
echo ""

echo "Step 3/4: Training neural spline flow..."
echo "----------------------------------------------------------------------"
echo "This will take 30-90 minutes..."
echo "Started at: $(date)"
python3 python/train_npe.py \
  --config "$CONFIG" \
  --simulations simulations/sim_data.npz \
  --out models/nsf_model.pt
echo "Finished at: $(date)"
echo ""

echo "Step 4/4: Inferring posterior distribution..."
echo "----------------------------------------------------------------------"
echo "Sampling 100,000 parameter sets from posterior..."
python3 python/infer_posterior.py \
  --model models/nsf_model.pt \
  --obs observed_data/observed_summaries.npz \
  --out results/posterior_samples.npz
echo ""

echo "======================================================================"
echo "ANALYSIS COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - results/posterior_samples.npz (raw posterior samples)"
echo "  - results/posterior_summary.json (parameter estimates)"
echo ""
echo "View parameter estimates:"
echo "  cat results/posterior_summary.json | jq"
echo ""
echo "Next steps:"
echo "  1. Load results in Python/R for visualization"
echo "  2. Create posterior plots"
echo "  3. Compare to prior expectations"
echo "  4. Validate with posterior predictive checks"
echo ""
echo "======================================================================"
