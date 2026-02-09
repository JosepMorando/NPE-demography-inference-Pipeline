#!/bin/bash
#
# Production Analysis Workflow with Reuse (Multi-node)
# Full demographic inference pipeline for real Pool-seq data with ability to reuse existing simulations.
#
# Usage:
#   bash scripts/run_production_multinode_reuse.sh
#   bash scripts/run_production_multinode_reuse.sh --reuse /path/to/existing/sims.npz
#
# Environment variables:
#   NODES="geu-master geu-worker1 geu-worker2"
#   WORKERS_PER_NODE=70
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
echo "PRODUCTION ANALYSIS WORKFLOW (MULTI-NODE WITH REUSE)"
echo "======================================================================"
echo ""
echo "This runs the full demographic inference pipeline on your real data."
echo ""

if [[ -n "$REUSE_NPZ" ]]; then
  echo "REUSE MODE: Will add new simulations to existing file"
  echo "Existing file: $REUSE_NPZ"
  echo ""
fi

echo "IMPORTANT: Have you validated the pipeline with POD testing?"
echo "If not, run: bash scripts/run_pod_test_multinode.sh first!"
echo ""

read -p "Continue with production analysis? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Configuration
CONFIG="config/config_production.yaml"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-40}"
NODES_STR="${NODES:-geu-master geu-worker1 geu-worker2}"
read -r -a NODES <<< "$NODES_STR"

# Threading safety
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Create output directories
mkdir -p observed_data simulations models results

PROJECT_ROOT="$(pwd)"

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

echo "Step 2/4: Generating training simulations (multi-node)..."
echo "----------------------------------------------------------------------"

# Read total sims from config
TOTAL_SIMS="$(PYTHONPATH="${PROJECT_ROOT}/python:${PYTHONPATH:-}" python3 - <<PY
from npe_demography.config import load_config
cfg = load_config("${CONFIG}")
print(int(cfg.get("npe", {}).get("n_sims", 50000)))
PY
)"

# If reusing, subtract existing sims from total
NEW_SIMS="$TOTAL_SIMS"
SEED_OFFSET=0
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
  SEED_OFFSET="$EXISTING_SIMS"
  
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

  echo ""
  echo "Config: $CONFIG"
  echo "Nodes: ${NODES[*]}"
  echo "Workers per node: $WORKERS_PER_NODE"
  echo "Distributing $NEW_SIMS new sims across $NUM_NODES nodes (seed_offset=$SEED_OFFSET)"
  echo "Started at: $(date)"
  echo ""

  PARTS=()
  PIDS=()

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

    echo "[${node}] launching: n=${n_chunk}, workers=${WORKERS_PER_NODE}, out=${part_out}, tmpdir=${NODE_TMPDIR}"

    ssh "$node" "cd '$PROJECT_ROOT' && \
      mkdir -p '${NODE_TMPDIR}' && \
      export PYTHONPATH='$PROJECT_ROOT/python:\${PYTHONPATH:-}' && \
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && \
      python3 python/simulate.py \
        --config '$CONFIG' \
        --n '$n_chunk' \
        --out '$part_out' \
        --seed-offset '$SEED_OFFSET' \
        --workers '$WORKERS_PER_NODE' \
        --tmpdir '${NODE_TMPDIR}' && \
      rm -rf '${NODE_TMPDIR}'" &

    PIDS+=("$!")
  done

  # Wait for all nodes
  FAIL=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      FAIL=1
    fi
  done

  if [[ "$FAIL" -ne 0 ]]; then
    echo ""
    echo "ERROR: One or more simulation chunks failed."
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
  echo ""
fi

echo ""
echo "Step 3/4: Training neural spline flow..."
echo "----------------------------------------------------------------------"
echo "Started at: $(date)"
python3 python/train_npe.py \
  --config "$CONFIG" \
  --simulations simulations/sim_data.npz \
  --out models/nsf_model.pt
echo "Finished at: $(date)"
echo ""

echo "Step 4/4: Inferring posterior distribution..."
echo "----------------------------------------------------------------------"
echo "Sampling posterior..."
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
echo "======================================================================"
