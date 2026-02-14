#!/bin/bash
#
# Production Analysis Workflow for Individual Populations Model (Single-node)
# Full demographic inference pipeline for real Pool-seq data
#
# Runs the 12-population individual model with 59 free parameters on real data.
#

set -euo pipefail

echo "======================================================================"
echo "PRODUCTION ANALYSIS WORKFLOW - INDIVIDUAL POPULATIONS MODEL"
echo "======================================================================"
echo ""
echo "This runs the full demographic inference pipeline on your real data."
echo "Model: 12 individual populations, 59 free parameters"
echo ""
echo "IMPORTANT: Have you validated the pipeline with POD testing?"
echo "If not, run: bash scripts/run_pod_test_individual.sh first!"
echo ""
read -p "Continue with production analysis? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Configuration
CONFIG="config/config_individual_pops.yaml"

# Create output directories
mkdir -p observed_data simulations models results

# Validate required files
echo ""
echo "Validating configuration files..."
for file in "$CONFIG" "config/groups_12individuals.csv" "templates/model_individual_pops.slim.tpl"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: Required file not found: $file"
    exit 1
  fi
done
echo "✓ All required files found"
echo ""

echo "Step 1/4: Computing observed summary statistics..."
echo "----------------------------------------------------------------------"
echo "Extracting statistics from Pool-seq data..."
Rscript r/compute_observed_summaries.R \
  --pooldata observed_data/Pooldata_demography.RData \
  --object filt.pooldata \
  --groups config/groups_12individuals.csv \
  --target_cov 10 \
  --out observed_data
echo ""

echo "Step 2/4: Generating training simulations (100,000 sims)..."
echo "----------------------------------------------------------------------"
echo "Note: 59-parameter model requires large simulation budget"
echo "This will take 6-12 hours with 30 workers on single node..."
echo "For faster execution, use: bash scripts/run_production_multinode_individual.sh"
echo "Started at: $(date)"
python3 python/simulate.py \
  --config "$CONFIG" \
  --out simulations/sim_data.npz \
  --workers 30
echo "Finished at: $(date)"
echo ""

echo "Step 3/4: Training neural spline flow..."
echo "----------------------------------------------------------------------"
echo "Note: Training 59-parameter model may take 1-3 hours (or 4-8 hours on CPU)"
echo "Started at: $(date)"
python3 python/train_npe.py \
  --config "$CONFIG" \
  --simulations simulations/sim_data.npz \
  --out models/nsf_model.pt
echo "Finished at: $(date)"
echo ""

echo "Step 4/4: Inferring posterior distribution..."
echo "----------------------------------------------------------------------"
echo "Sampling 20,000 parameter sets from posterior..."
python3 python/infer_posterior.py \
  --model models/nsf_model.pt \
  --obs observed_data/observed_summaries.npz \
  --out results/posterior_samples.npz \
  --config "$CONFIG"
echo ""

echo "======================================================================"
echo "ANALYSIS COMPLETE - INDIVIDUAL POPULATIONS MODEL!"
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
echo "  2. Create posterior plots for all 12 populations"
echo "  3. Compare to prior expectations"
echo "  4. Validate with posterior predictive checks:"
echo "     python3 python/validate_ppc.py --config $CONFIG \\"
echo "       --model models/nsf_model.pt \\"
echo "       --obs observed_data/observed_summaries.npz \\"
echo "       --out results/ppc_results.npz"
echo ""
echo "======================================================================"
