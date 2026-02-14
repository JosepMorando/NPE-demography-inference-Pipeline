#!/bin/bash
#
# POD Testing Workflow (Multi-node)
# Validates the NPE pipeline using pseudo-observed data with known parameters.
# Splits simulation work across multiple nodes and merges results.
#
# Usage:
#   bash scripts/run_pod_test_multinode.sh
#   bash scripts/run_pod_test_multinode.sh --reuse /path/to/existing/sim_data.npz
#
# Environment variables:
#   NODES="geu-master geu-worker1 geu-worker2"   (space-separated node names)
#   WORKERS_PER_NODE=70                           (parallel workers per node)
#   ENABLE_SIM_COMPRESSION=0|1                    (default 0 for speed)
#

set -euo pipefail
CONFIG="config/config_pod.yaml"

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
echo "POD TESTING WORKFLOW (MULTI-NODE)"
echo "======================================================================"
echo ""
echo "This validates the pipeline by testing if it can recover known"
echo "parameters from synthetic data."
echo ""

# Configuration
CONFIG="config/config_pod.yaml"
POD_DIR="pod_test"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-70}"
NODES_STR="${NODES:-geu-master geu-worker1 geu-worker2}"
ENABLE_SIM_COMPRESSION="${ENABLE_SIM_COMPRESSION:-0}"
read -r -a NODES <<< "$NODES_STR"

# Threading safety
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$POD_DIR"/{simulations,models,results,logs}

PROJECT_ROOT="$(pwd)"

echo "Config: $CONFIG"
echo "Nodes: ${NODES[*]}"
echo "Workers per node: $WORKERS_PER_NODE"
echo "Simulation compression: $ENABLE_SIM_COMPRESSION (0 is fastest)"
if [[ -n "$REUSE_NPZ" ]]; then
  echo "Reusing existing simulations: $REUSE_NPZ"
fi
echo ""

# ---- Step 1: Generate POD ----
echo "Step 1/7: Generating pseudo-observed data..."
echo "----------------------------------------------------------------------"
python3 scripts/generate_pod.py --config "$CONFIG" --out "$POD_DIR"
echo ""

# ---- Step 2: Simulations (multi-node) ----
echo "Step 2/7: Generating training simulations (multi-node)..."
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
    cp "$REUSE_NPZ" "$POD_DIR/simulations/sim_data.npz"
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

  echo "Distributing $NEW_SIMS new sims across $NUM_NODES nodes (seed_offset=$SEED_OFFSET)"
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

    part_out="$POD_DIR/simulations/sim_part$((i+1)).npz"
    PARTS+=("$part_out")

    # Create unique temp directory for this node
    NODE_TMPDIR="/tmp/sim_scratch_${node}_$$"
    NODE_LOCAL_OUT="${NODE_TMPDIR}/sim_part$((i+1)).npz"

    NODE_SEED_OFFSET="$SEED_CURSOR"
    SEED_CURSOR=$(( SEED_CURSOR + n_chunk ))

    log_file="$POD_DIR/logs/sim_${node}.log"
    LOGS+=("$log_file")
    : > "$log_file"

    echo "[${node}] launching: n=${n_chunk}, workers=${WORKERS_PER_NODE}, seed_offset=${NODE_SEED_OFFSET}, tmpdir=${NODE_TMPDIR}"

    tail -n 0 -F "$log_file" 2>/dev/null | sed -u "s/^/[${node}] /" &
    TAIL_PIDS+=("$!")

    COMPRESS_FLAG=""
    if [[ "$ENABLE_SIM_COMPRESSION" == "1" ]]; then
      COMPRESS_FLAG="--compress-output"
    fi

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

  # Wait for all nodes
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
    echo "Check per-node output above and logs in $POD_DIR/logs/."
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

  echo "Merging ${#MERGE_PARTS[@]} parts -> $POD_DIR/simulations/sim_data.npz"
  python3 python/merge_sim_npz.py --out "$POD_DIR/simulations/sim_data.npz" "${MERGE_PARTS[@]}"
  echo ""
fi

echo ""

# ---- Step 3: Train ----
echo "Step 3/7: Training neural spline flow..."
echo "----------------------------------------------------------------------"
echo "Started at: $(date)"
python3 python/train_npe.py \
  --config "$CONFIG" \
  --simulations "$POD_DIR/simulations/sim_data.npz" \
  --out "$POD_DIR/models/nsf_model.pt"
echo "Finished at: $(date)"
echo ""

# ---- Step 4: Infer posterior ----
echo "Step 4/7: Inferring posterior for POD..."
echo "----------------------------------------------------------------------"
python3 python/infer_posterior.py \
  --model "$POD_DIR/models/nsf_model.pt" \
  --obs "$POD_DIR/pod_summaries.npz" \
  --out "$POD_DIR/results/pod_posterior.npz" \
  --config "$CONFIG"
echo ""

# ---- Step 5: Check recovery ----
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
