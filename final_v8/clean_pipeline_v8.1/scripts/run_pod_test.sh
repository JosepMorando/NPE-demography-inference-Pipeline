#!/bin/bash
#
# POD Testing Workflow
# Validates the NPE pipeline using pseudo-observed data with known parameters
#

set -euo pipefail

echo "======================================================================"
echo "POD TESTING WORKFLOW"
echo "======================================================================"
echo ""
echo "This validates the pipeline by testing if it can recover known"
echo "parameters from synthetic data."
echo ""

# Configuration
CONFIG="config/config_pod.yaml"
POD_DIR="pod_test"

mkdir -p "$POD_DIR"/{simulations,models,results}

echo "Step 1/5: Generating pseudo-observed data..."
echo "----------------------------------------------------------------------"
python3 scripts/generate_pod.py --config "$CONFIG" --out "$POD_DIR"
echo ""

echo "Step 2/5: Generating training simulations (5,000 sims)..."
echo "----------------------------------------------------------------------"
echo "This will take ~30-60 minutes..."
python3 python/simulate.py \
  --config "$CONFIG" \
  --out "$POD_DIR/simulations/sim_data.npz" \
  --workers 70
echo ""

echo "Step 3/5: Training neural network..."
echo "----------------------------------------------------------------------"
echo "This will take ~10-20 minutes..."
python3 python/train_npe.py \
  --config "$CONFIG" \
  --simulations "$POD_DIR/simulations/sim_data.npz" \
  --out "$POD_DIR/models/mdn_model.pt"
echo ""

echo "Step 4/5: Inferring posterior for POD..."
echo "----------------------------------------------------------------------"
python3 python/infer_posterior.py \
  --model "$POD_DIR/models/mdn_model.pt" \
  --obs "$POD_DIR/pod_summaries.npz" \
  --out "$POD_DIR/results/pod_posterior.npz"
echo ""

echo "Step 5/5: Checking parameter recovery..."
echo "----------------------------------------------------------------------"
python3 scripts/check_pod_recovery.py \
  --true "$POD_DIR/pod_observed.npz" \
  --posterior "$POD_DIR/results/pod_posterior.npz" \
  --out "$POD_DIR/results"
echo ""

echo "======================================================================"
echo "POD TESTING COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to: $POD_DIR/results/"
echo ""
echo "Check parameter recovery:"
echo "  - Coverage should be ~90-95%"
echo "  - See plots in $POD_DIR/results/"
echo ""
echo "If coverage is good (>85%), proceed with production analysis."
echo "If coverage is poor (<80%), increase n_sims or check pipeline."
echo ""
echo "======================================================================"
