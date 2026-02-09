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

echo "Step 1/7: Generating pseudo-observed data..."
echo "----------------------------------------------------------------------"
python3 scripts/generate_pod.py --config "$CONFIG" --out "$POD_DIR"
echo ""

echo "Step 2/7: Generating training simulations (5,000 sims)..."
echo "----------------------------------------------------------------------"
echo "This will take ~30-60 minutes..."
python3 python/simulate.py \
  --config "$CONFIG" \
  --out "$POD_DIR/simulations/sim_data.npz" \
  --workers 70
echo ""

echo "Step 3/7: Training neural spline flow..."
echo "----------------------------------------------------------------------"
echo "This will take ~10-20 minutes..."
python3 python/train_npe.py \
  --config "$CONFIG" \
  --simulations "$POD_DIR/simulations/sim_data.npz" \
  --out "$POD_DIR/models/nsf_model.pt"
echo ""

echo "Step 4/7: Inferring posterior for POD..."
echo "----------------------------------------------------------------------"
python3 python/infer_posterior.py \
  --model "$POD_DIR/models/nsf_model.pt" \
  --obs "$POD_DIR/pod_summaries.npz" \
  --out "$POD_DIR/results/pod_posterior.npz"
echo ""

echo "Step 5/7: Checking parameter recovery..."
echo "----------------------------------------------------------------------"
python3 scripts/check_pod_recovery.py \
  --true "$POD_DIR/pod_observed.npz" \
  --posterior "$POD_DIR/results/pod_posterior.npz" \
  --out "$POD_DIR/results"
echo ""

echo "Step 6/7: Running SBC validation..."
echo "----------------------------------------------------------------------"
python3 python/validate_sbc.py \
  --model "$POD_DIR/models/nsf_model.pt" \
  --simulations "$POD_DIR/simulations/sim_data.npz" \
  --out "$POD_DIR/results/sbc_results.npz"
echo ""

echo "Step 7/7: Running PPC validation..."
echo "----------------------------------------------------------------------"
python3 python/validate_ppc.py \
  --config "$CONFIG" \
  --model "$POD_DIR/models/nsf_model.pt" \
  --obs "$POD_DIR/pod_summaries.npz" \
  --out "$POD_DIR/results/ppc_results.npz"
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
