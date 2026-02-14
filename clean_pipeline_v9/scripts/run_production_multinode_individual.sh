#!/bin/bash
#
# Production Analysis Workflow for Individual Populations Model (Multi-node)
# Full demographic inference pipeline for real Pool-seq data across multiple nodes.
#
# Runs the 12-population individual model with 59 free parameters on real data.
#
# Usage:
#   bash scripts/run_production_multinode_individual.sh
#   bash scripts/run_production_multinode_individual.sh --reuse /path/to/existing/sim_data.npz
#
# Environment variables:
#   NODES="geu-master geu-worker1 geu-worker2"   (space-separated node names)
#   WORKERS_PER_NODE=70                           (parallel workers per node)
#   ENABLE_SIM_COMPRESSION=0|1                    (default 0 for speed)
#

set -euo pipefail

# ---- Parse arguments ----
REUSE_NPZ=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reuse)
      REUSE_NPZ="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--reuse /path/to/existing/sim_data.npz]"
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo "PRODUCTION ANALYSIS WORKFLOW - INDIVIDUAL POPULATIONS (MULTI-NODE)"
echo "======================================================================"
echo ""
echo "This runs the full demographic inference pipeline on your real data,"
echo "splitting training simulations across multiple nodes and merging."
echo ""
echo "Model: 12 individual populations, 59 free parameters"
echo ""
echo "IMPORTANT: Have you validated the pipeline with POD testing?"
echo "If not, run: bash scripts/run_pod_test_multinode_individual.sh first!"
echo ""

read -p "Continue with production analysis? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Configuration
CONFIG="config/config_individual_pops.yaml"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-70}"
NODES_STR="${NODES:-geu-master geu-worker1 geu-worker2}"
ENABLE_SIM_COMPRESSION="${ENABLE_SIM_COMPRESSION:-0}"
read -r -a NODES <<< "$NODES_STR"

# Threading safety: avoid nested BLAS/OpenMP threading inside each worker process.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Create output directories
mkdir -p observed_data simulations models results logs

# Resolve absolute project path (must be accessible from all nodes via shared FS)
PROJECT_ROOT="$(pwd)"

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

echo "Config: $CONFIG"
echo "Nodes: ${NODES[*]}"
echo "Workers per node: $WORKERS_PER_NODE"
echo "Simulation compression: $ENABLE_SIM_COMPRESSION (0 is fastest)"
if [[ -n "$REUSE_NPZ" ]]; then
  echo "Reusing existing simulations: $REUSE_NPZ"
fi
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

echo "Step 2/4: Generating training simulations (multi-node)..."
echo "----------------------------------------------------------------------"

# Determine total number of simulations from config
TOTAL_SIMS="$(PYTHONPATH="${PROJECT_ROOT}/python:${PYTHONPATH:-}" python3 - <<PY
from npe_demography.config import load_config
cfg = load_config("${CONFIG}")
print(int(cfg.get("npe", {}).get("n_sims", 100000)))
PY
)"

# If reusing, subtract existing sims from total
NEW_SIMS="$TOTAL_SIMS"
if [[ -n "$REUSE_NPZ" ]]; then
  if [[ ! -f "$REUSE_NPZ" ]]; then
    echo "ERROR: Reuse file not found: $REUSE_NPZ"
    exit 1
  fi
  EXISTING_SIMS="$(PYTHONPATH="${PROJECT_ROOT}/python:${PYTHONPATH:-}" python3 - <<PY
import numpy as np
z = np.load("${REUSE_NPZ}", allow_pickle=True)
print(z["X"].shape[0])
PY
)"
  echo "Existing simulations: $EXISTING_SIMS"
  echo "Target total: $TOTAL_SIMS"
  NEW_SIMS=$(( TOTAL_SIMS - EXISTING_SIMS ))
  if [[ "$NEW_SIMS" -le 0 ]]; then
    echo "Already have >= $TOTAL_SIMS sims. Copying existing file."
    cp "$REUSE_NPZ" "simulations/sim_data.npz"
    NEW_SIMS=0
  else
    echo "Need $NEW_SIMS new simulations"
  fi
fi

if [[ "$NEW_SIMS" -gt 0 ]]; then
  NUM_NODES="${#NODES[@]}"
  BASE=$(( NEW_SIMS / NUM_NODES ))
  REM=$(( NEW_SIMS % NUM_NODES ))

  # When reusing existing sims, offset seeds to avoid duplicating the same draws
  SEED_OFFSET=0
  if [[ -n "$REUSE_NPZ" ]]; then
    SEED_OFFSET="$EXISTING_SIMS"
  fi

  echo "Total simulations: $TOTAL_SIMS"
  echo "Distributing $NEW_SIMS new sims across $NUM_NODES nodes (seed_offset=$SEED_OFFSET)"
  echo "Chunking: base=$BASE remainder=$REM"
  echo "Started at: $(date)"
  echo ""

  PARTS=()
  PIDS=()
  LOGS=()
  TAIL_PIDS=()
  SEED_CURSOR="$SEED_OFFSET"

  cleanup_tails() {
    for tpid in "${TAIL_PIDS[@]:-}"; do
      kill "$tpid" 2>/dev/null || true
    done
  }
  trap cleanup_tails EXIT

  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"

    n_chunk="$BASE"
    if [[ "$i" -lt "$REM" ]]; then
      n_chunk=$(( BASE + 1 ))
    fi

    part_out="simulations/sim_part$((i+1)).npz"
    PARTS+=("$part_out")

    # Create unique temp directory for this node
    NODE_TMPDIR="/tmp/sim_scratch_${node}_$$"
    NODE_LOCAL_OUT="${NODE_TMPDIR}/sim_part$((i+1)).npz"

    NODE_SEED_OFFSET="$SEED_CURSOR"
    SEED_CURSOR=$(( SEED_CURSOR + n_chunk ))

    log_file="logs/sim_${node}.log"
    LOGS+=("$log_file")
    : > "$log_file"

    echo "[${node}] launching: n=${n_chunk}, workers=${WORKERS_PER_NODE}, seed_offset=${NODE_SEED_OFFSET}, tmpdir=${NODE_TMPDIR}"

    tail -n 0 -F "$log_file" 2>/dev/null | sed -u "s/^/[${node}] /" &
    TAIL_PIDS+=("$!")

    COMPRESS_FLAG=""
    if [[ "$ENABLE_SIM_COMPRESSION" == "1" ]]; then
      COMPRESS_FLAG="--compress-output"
    fi

    # Launch simulation chunk on each node. Assumes shared filesystem for PROJECT_ROOT.
    ssh "$node" "cd '$PROJECT_ROOT' && \
      mkdir -p '${NODE_TMPDIR}' && \
      export PYTHONPATH='$PROJECT_ROOT/python:\${PYTHONPATH:-}' && \
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && \
      python3 -u python/simulate.py \
        --config '$CONFIG' \
        --n '$n_chunk' \
        --out '${NODE_LOCAL_OUT}' \
        --seed-offset '$NODE_SEED_OFFSET' \
        --workers '$WORKERS_PER_NODE' \
        --tmpdir '${NODE_TMPDIR}' \
        ${COMPRESS_FLAG} && \
      cp '${NODE_LOCAL_OUT}' '$part_out' && \
      rm -rf '${NODE_TMPDIR}'" > "$log_file" 2>&1 &

    PIDS+=("$!")
  done

  # Wait for all nodes to finish
  FAIL=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      FAIL=1
    fi
  done

  cleanup_tails
  trap - EXIT

  if [[ "$FAIL" -ne 0 ]]; then
    echo ""
    echo "ERROR: One or more simulation chunks failed."
    echo "Check per-node output above and logs in logs/."
    exit 1
  fi

  echo ""
  echo "All chunks finished at: $(date)"

  # Build merge list: include reuse file first (if any), then new parts
  MERGE_PARTS=()
  if [[ -n "$REUSE_NPZ" ]]; then
    MERGE_PARTS+=("$REUSE_NPZ")
  fi
  MERGE_PARTS+=("${PARTS[@]}")

  echo "Merging ${#MERGE_PARTS[@]} parts -> simulations/sim_data.npz"
  python3 python/merge_sim_npz.py --out simulations/sim_data.npz "${MERGE_PARTS[@]}"
  echo "Finished merge at: $(date)"
  echo ""
fi

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
  --config "$CONFIG" \
  --model models/nsf_model.pt \
  --obs observed_data/observed_summaries.npz \
  --out results/posterior_samples.npz
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
