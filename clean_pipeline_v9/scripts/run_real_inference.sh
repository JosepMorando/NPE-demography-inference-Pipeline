#!/bin/bash
#
# Run NPE inference on real observed data
# Uses trained model from POD test to infer demographic parameters
#

set -euo pipefail

echo "======================================================================"
echo "NPE INFERENCE ON REAL DATA - INDIVIDUAL POPULATIONS"
echo "======================================================================"
echo ""

# Configuration
CONFIG="config/config_individual_pops.yaml"
MODEL="pod_individual/models/nsf_model.pt"
OBS_DATA="observed_data/observed_summaries.npz"
OUTPUT_DIR="real_data_results"
POSTERIOR_OUT="$OUTPUT_DIR/posterior.npz"

# Check if model exists, otherwise suggest POD test
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: Trained model not found at: $MODEL"
  echo ""
  echo "You need to run the POD test first to train the model:"
  echo "  bash scripts/run_pod_test_individual.sh"
  echo ""
  echo "Or specify a different model path."
  exit 1
fi

# Check if observed data exists
if [[ ! -f "$OBS_DATA" ]]; then
  echo "ERROR: Observed data not found at: $OBS_DATA"
  echo ""
  echo "You need to compute summary statistics from your VCF/RData first."
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Config:        $CONFIG"
echo "  Model:         $MODEL"
echo "  Observed data: $OBS_DATA"
echo "  Output:        $POSTERIOR_OUT"
echo ""

echo "Step 1/3: Checking observed vs simulated data ranges..."
echo "----------------------------------------------------------------------"
# Optional: Check if observed data is within simulation range
if [[ -f "scripts/check_obs_vs_sim.py" ]]; then
  python3 scripts/check_obs_vs_sim.py || echo "Warning: Some observed stats may be outside simulation range"
fi
echo ""

echo "Step 2/3: Running NPE inference on real data..."
echo "----------------------------------------------------------------------"
echo "This will draw 20,000 posterior samples from the trained model"
echo "Started at: $(date)"
python3 python/infer_posterior.py \
  --model "$MODEL" \
  --obs "$OBS_DATA" \
  --out "$POSTERIOR_OUT" \
  --config "$CONFIG"
echo "Finished at: $(date)"
echo ""

echo "Step 3/3: Running posterior predictive checks..."
echo "----------------------------------------------------------------------"
echo "Validating inference quality with PPC"
python3 python/validate_ppc.py \
  --config "$CONFIG" \
  --model "$MODEL" \
  --obs "$OBS_DATA" \
  --out "$OUTPUT_DIR/ppc_results.npz"
echo ""

echo "======================================================================"
echo "INFERENCE COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Files created:"
echo "  - $POSTERIOR_OUT       : Posterior samples (20,000 draws)"
echo "  - $OUTPUT_DIR/ppc_results.npz : Posterior predictive check results"
echo ""
echo "Next steps:"
echo "  1. Visualize posterior distributions"
echo "  2. Check PPC results for model fit"
echo "  3. Extract point estimates (median, MAP) and credible intervals"
echo "  4. Compare with prior expectations"
echo ""
echo "To examine posterior:"
echo "  python3 -c 'import numpy as np; d=np.load(\"$POSTERIOR_OUT\"); print(d.files); print(d[\"theta\"].shape)'"
echo ""
echo "======================================================================"
