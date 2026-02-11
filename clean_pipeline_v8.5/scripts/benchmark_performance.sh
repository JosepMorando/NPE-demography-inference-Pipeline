#!/bin/bash
#
# Performance Benchmark Script for Multi-Node Simulations
#
# Tests different configurations to identify bottlenecks and optimal settings.
#
# Usage:
#   bash scripts/benchmark_performance.sh
#

set -euo pipefail

CONFIG="config/config_pod.yaml"
PROJECT_ROOT="$(pwd)"
BENCHMARK_DIR="benchmark_results"
mkdir -p "$BENCHMARK_DIR"

# Threading safety
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "======================================================================"
echo "PERFORMANCE BENCHMARK SUITE"
echo "======================================================================"
echo ""
echo "This script tests different configurations to identify optimal settings"
echo "for multi-node simulation performance."
echo ""

# Get baseline CPU info
echo "System Information:"
echo "----------------------------------------------------------------------"
NODE_NAME="$(hostname)"
echo "Node: $NODE_NAME"
if command -v nproc &> /dev/null; then
    N_CORES=$(nproc)
    echo "Total cores: $N_CORES"
fi
if command -v free &> /dev/null; then
    free -h | grep "Mem:"
fi
echo ""

# ============================================================================
# Test 1: Single-node baseline with varying worker counts
# ============================================================================
echo "Test 1: Worker Count Optimization (Single Node)"
echo "======================================================================"
echo ""

N_SIMS_TEST=50  # Small batch for quick tests

for WORKERS in 10 20 40 70; do
    echo "Testing with $WORKERS workers..."
    echo "----------------------------------------------------------------------"
    
    OUT_FILE="$BENCHMARK_DIR/test1_workers_${WORKERS}.npz"
    LOG_FILE="$BENCHMARK_DIR/test1_workers_${WORKERS}.log"
    
    START_TIME=$(date +%s)
    
    python3 python/simulate.py \
        --config "$CONFIG" \
        --n "$N_SIMS_TEST" \
        --workers "$WORKERS" \
        --out "$OUT_FILE" \
        2>&1 | tee "$LOG_FILE"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo "Completed in ${ELAPSED}s"
    echo "Throughput: $(python3 -c "print(f'{$N_SIMS_TEST / $ELAPSED:.2f}')") sims/sec"
    echo ""
done

# ============================================================================
# Test 2: tmpdir location comparison
# ============================================================================
echo ""
echo "Test 2: Storage Location Performance"
echo "======================================================================"
echo ""

WORKERS_OPTIMAL=40  # Use result from Test 1 or default

for TMPDIR_LOC in "/dev/shm" "/tmp"; do
    echo "Testing with tmpdir=$TMPDIR_LOC..."
    echo "----------------------------------------------------------------------"
    
    OUT_FILE="$BENCHMARK_DIR/test2_tmpdir_$(basename $TMPDIR_LOC).npz"
    LOG_FILE="$BENCHMARK_DIR/test2_tmpdir_$(basename $TMPDIR_LOC).log"
    
    # Create tmpdir
    TMPDIR_TEST="${TMPDIR_LOC}/benchmark_test_$$"
    mkdir -p "$TMPDIR_TEST"
    
    START_TIME=$(date +%s)
    
    python3 python/simulate.py \
        --config "$CONFIG" \
        --n "$N_SIMS_TEST" \
        --workers "$WORKERS_OPTIMAL" \
        --tmpdir "$TMPDIR_TEST" \
        --out "$OUT_FILE" \
        2>&1 | tee "$LOG_FILE"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    # Cleanup
    rm -rf "$TMPDIR_TEST"
    
    echo "Completed in ${ELAPSED}s"
    echo "Throughput: $(python3 -c "print(f'{$N_SIMS_TEST / $ELAPSED:.2f}')") sims/sec"
    echo ""
done

# ============================================================================
# Test 3: Multi-node scaling
# ============================================================================
echo ""
echo "Test 3: Multi-Node Scaling Test"
echo "======================================================================"
echo ""

NODES_STR="${NODES:-geu-master geu-worker1 geu-worker2}"
read -r -a NODES_ARRAY <<< "$NODES_STR"

if [ "${#NODES_ARRAY[@]}" -gt 1 ]; then
    echo "Testing single node vs multi-node with 100 simulations..."
    echo ""
    
    # Single node reference
    echo "Single Node (${NODES_ARRAY[0]}):"
    echo "----------------------------------------------------------------------"
    
    OUT_FILE="$BENCHMARK_DIR/test3_single_node.npz"
    LOG_FILE="$BENCHMARK_DIR/test3_single_node.log"
    
    START_TIME=$(date +%s)
    
    ssh "${NODES_ARRAY[0]}" "cd '$PROJECT_ROOT' && \
        export PYTHONPATH='$PROJECT_ROOT/python:\${PYTHONPATH:-}' && \
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && \
        python3 python/simulate.py \
            --config '$CONFIG' \
            --n 100 \
            --workers '$WORKERS_OPTIMAL' \
            --tmpdir '/tmp/benchmark_single_$$' \
            --out '$OUT_FILE' && \
        rm -rf '/tmp/benchmark_single_$$'" 2>&1 | tee "$LOG_FILE"
    
    END_TIME=$(date +%s)
    ELAPSED_SINGLE=$((END_TIME - START_TIME))
    
    echo "Single node: ${ELAPSED_SINGLE}s"
    echo ""
    
    # Multi-node test
    echo "Multi-Node (${#NODES_ARRAY[@]} nodes):"
    echo "----------------------------------------------------------------------"
    
    N_TOTAL=100
    N_NODES="${#NODES_ARRAY[@]}"
    BASE=$((N_TOTAL / N_NODES))
    REM=$((N_TOTAL % N_NODES))
    
    START_TIME=$(date +%s)
    PIDS=()
    PARTS=()
    
    for i in "${!NODES_ARRAY[@]}"; do
        node="${NODES_ARRAY[$i]}"
        
        n_chunk="$BASE"
        if [[ "$i" -lt "$REM" ]]; then
            n_chunk=$((BASE + 1))
        fi
        
        part_out="$BENCHMARK_DIR/test3_multi_part$((i+1)).npz"
        PARTS+=("$part_out")
        
        NODE_TMPDIR="/tmp/benchmark_multi_${node}_$$"
        
        echo "[${node}] Launching: n=${n_chunk}, tmpdir=${NODE_TMPDIR}"
        
        ssh "$node" "cd '$PROJECT_ROOT' && \
            mkdir -p '${NODE_TMPDIR}' && \
            export PYTHONPATH='$PROJECT_ROOT/python:\${PYTHONPATH:-}' && \
            export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && \
            python3 python/simulate.py \
                --config '$CONFIG' \
                --n '$n_chunk' \
                --workers '$WORKERS_OPTIMAL' \
                --tmpdir '${NODE_TMPDIR}' \
                --out '$part_out' && \
            rm -rf '${NODE_TMPDIR}'" &
        
        PIDS+=("$!")
    done
    
    # Wait for all
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
    
    # Merge
    python3 python/merge_sim_npz.py \
        --out "$BENCHMARK_DIR/test3_multi_merged.npz" \
        "${PARTS[@]}"
    
    END_TIME=$(date +%s)
    ELAPSED_MULTI=$((END_TIME - START_TIME))
    
    echo ""
    echo "Multi-node: ${ELAPSED_MULTI}s"
    echo "Speedup: $(python3 -c "print(f'{$ELAPSED_SINGLE / $ELAPSED_MULTI:.2f}x')")"
    echo ""
else
    echo "Skipping multi-node test (only one node configured)"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "BENCHMARK SUMMARY"
echo "======================================================================"
echo ""
echo "Results saved to: $BENCHMARK_DIR/"
echo ""
echo "Recommendations:"
echo "----------------------------------------------------------------------"

# Analyze Test 1 results
if [ -f "$BENCHMARK_DIR/test1_workers_10.log" ]; then
    echo "Worker count analysis:"
    for WORKERS in 10 20 40 70; do
        LOG_FILE="$BENCHMARK_DIR/test1_workers_${WORKERS}.log"
        if [ -f "$LOG_FILE" ]; then
            # Extract timing from log
            if grep -q "Saved simulations" "$LOG_FILE"; then
                echo "  - $WORKERS workers: $(grep -A 5 "Progress: $N_SIMS_TEST" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+(?=%)')"
            fi
        fi
    done
    echo ""
fi

echo "Review detailed logs in $BENCHMARK_DIR/ for further analysis."
echo ""
echo "======================================================================"
