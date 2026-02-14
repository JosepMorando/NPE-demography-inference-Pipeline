#!/bin/bash
#
# POD Testing Workflow for Individual Populations Model (Single-node)
# Validates the NPE pipeline using pseudo-observed data with known parameters
#
# Tests the 12-population individual model with 59 free parameters.
#

set -euo pipefail

echo "======================================================================"
echo "POD TESTING WORKFLOW - INDIVIDUAL POPULATIONS MODEL"
echo "======================================================================"
echo ""
echo "This validates the individual populations pipeline (12 pops, 59 params)"
echo "by testing if it can recover known parameters from synthetic data."
echo ""

# Configuration
CONFIG="config/config_pod_individual.yaml"
POD_DIR="pod_individual"

mkdir -p "$POD_DIR"/{simulations,models,results}

# Validate required files
echo "Validating configuration files..."
for file in "$CONFIG" "config/groups_12individuals.csv" "templates/model_individual_pops.slim.tpl"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: Required file not found: $file"
    exit 1
  fi
done
echo "✓ All required files found"
echo ""

echo "Step 1/7: Generating pseudo-observed data..."
echo "----------------------------------------------------------------------"
python3 scripts/generate_pod.py --config "$CONFIG" --out "$POD_DIR"
echo ""

echo "Step 2/7: Generating training simulations (50,000 sims)..."
echo "----------------------------------------------------------------------"
echo "Note: 59-parameter model requires more simulations"
echo "This will take ~2-4 hours on single node..."
echo "Started at: $(date)"
python3 python/simulate.py \
  --config "$CONFIG" \
  --out "$POD_DIR/simulations/sim_data.npz" \
  --workers 30
echo "Finished at: $(date)"
echo ""

echo "Step 3/7: Training neural spline flow..."
echo "----------------------------------------------------------------------"
echo "Note: Training 59-parameter model may take 30-60 minutes (or 2-4 hours on CPU)"
echo "Started at: $(date)"
python3 python/train_npe.py \
  --config "$CONFIG" \
  --simulations "$POD_DIR/simulations/sim_data.npz" \
  --out "$POD_DIR/models/nsf_model.pt"
echo "Finished at: $(date)"
echo ""

echo "Step 4/7: Inferring posterior for POD..."
echo "----------------------------------------------------------------------"
python3 python/infer_posterior.py \
  --model "$POD_DIR/models/nsf_model.pt" \
  --obs "$POD_DIR/pod_summaries.npz" \
  --out "$POD_DIR/results/pod_posterior.npz" \
  --config "$CONFIG"
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
  --config "$CONFIG" \
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
echo "POD TESTING COMPLETE - INDIVIDUAL POPULATIONS MODEL!"
echo "======================================================================"
echo ""
echo "Model tested: 12 populations, 59 free parameters"
echo "Results saved to: $POD_DIR/results/"
echo ""
echo "Check parameter recovery:"
echo "  - Coverage should be ~90-95% (acceptable: >85%)"
echo "  - See plots and detailed report in $POD_DIR/results/"
echo ""
echo "Interpretation for 59-parameter model:"
echo "  - Excellent: Coverage 90-100%"
echo "  - Good:      Coverage 85-90%"
echo "  - Warning:   Coverage 75-85% (consider more sims)"
echo "  - Poor:      Coverage <75% (increase n_sims or check pipeline)"
echo ""
echo "If coverage is good, proceed with production analysis."
echo "If coverage is suboptimal, consider:"
echo "  - Increasing n_sims to 75k-100k"
echo "  - Running hierarchical test (without bottlenecks first)"
echo "  - Checking summary statistics quality"
echo ""
echo "For faster testing on multiple nodes, use:"
echo "  bash scripts/run_pod_test_multinode_individual.sh"
echo ""
echo "======================================================================"
