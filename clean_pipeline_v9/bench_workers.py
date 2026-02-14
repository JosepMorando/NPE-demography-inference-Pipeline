import time
import subprocess
import shutil
from pathlib import Path

# Config
CONFIG = "config/config_pod.yaml"
N_SIMS = 200  # Enough to saturate the cores
WORKER_COUNTS = [60, 40]

def run_benchmark(workers):
    print(f"\n[BENCHMARK] Testing with {workers} workers...")
    
    # Clean up previous run
    output_file = f"bench_{workers}.npz"
    if Path(output_file).exists():
        Path(output_file).unlink()
        
    cmd = [
        "python3", "python/simulate.py",
        "--config", CONFIG,
        "--n", str(N_SIMS),
        "--workers", str(workers),
        "--out", output_file
    ]
    
    start_time = time.time()
    try:
        # Run and hide the noisy output
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"❌ Run failed with {workers} workers!")
        return 0
        
    duration = time.time() - start_time
    throughput = N_SIMS / duration
    print(f"✅ Finished in {duration:.2f}s")
    print(f"🚀 Throughput: {throughput:.2f} sims/sec")
    return throughput

print("========================================")
print("      Core Scaling Benchmark")
print("========================================")

results = {}
for w in WORKER_COUNTS:
    results[w] = run_benchmark(w)

print("\n================ RESULT ================")
w60 = results[60]
w40 = results[40]

print(f"60 Workers: {w60:.2f} sims/sec")
print(f"40 Workers: {w40:.2f} sims/sec")

if w40 > w60:
    diff = (w40 - w60) / w60 * 100
    print(f"\n🏆 WINNER: 40 Workers is {diff:.1f}% FASTER!")
    print("Recommendation: Use --workers 40")
else:
    diff = (w60 - w40) / w40 * 100
    print(f"\n🏆 WINNER: 60 Workers is {diff:.1f}% FASTER!")
    print("Recommendation: Keep --workers 60")
