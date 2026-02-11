#!/bin/bash
#
# Production Analysis Workflow (Multi-node)
# Full demographic inference pipeline for real Pool-seq data across multiple nodes.
#

set -euo pipefail

echo "======================================================================"
echo "PRODUCTION ANALYSIS WORKFLOW (MULTI-NODE)"
echo "======================================================================"
echo ""
echo "This runs the full demographic inference pipeline on your real data,"
echo "splitting training simulations across multiple nodes and merging."
echo ""
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

# Determine total number of simulations from config (same as single-node workflow).
TOTAL_SIMS="$(PYTHONPATH="${PROJECT_ROOT}/python:${PYTHONPATH:-}" python3 - <<PY
from npe_demography.config import load_config
cfg = load_config("${CONFIG}")
print(int(cfg.get("npe", {}).get("n_sims", 50000)))
PY
)"
NUM_NODES="${#NODES[@]}"

echo "Config: $CONFIG"
echo "Nodes: ${NODES[*]}"
echo "Workers per node: $WORKERS_PER_NODE"
echo "Simulation compression: $ENABLE_SIM_COMPRESSION (0 is fastest)"
echo "Total simulations: $TOTAL_SIMS"
echo ""

# Split simulations as evenly as possible across nodes
BASE=$(( TOTAL_SIMS / NUM_NODES ))
REM=$(( TOTAL_SIMS % NUM_NODES ))

echo "Chunking: base=$BASE remainder=$REM"
echo "Started at: $(date)"
echo ""

PARTS=()
PIDS=()
TAIL_PIDS=()
SEED_CURSOR=0

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

  NODE_SEED_OFFSET="$SEED_CURSOR"
  SEED_CURSOR=$(( SEED_CURSOR + n_chunk ))

  NODE_TMPDIR="/tmp/sim_scratch_${node}_$$"
  NODE_LOCAL_OUT="${NODE_TMPDIR}/sim_part$((i+1)).npz"
  log_file="logs/sim_${node}.log"
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
echo "All chunks finished. Merging parts -> simulations/sim_data.npz"
python3 python/merge_sim_npz.py --out simulations/sim_data.npz "${PARTS[@]}"
echo "Finished merge at: $(date)"
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
  --config "$CONFIG" \
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
