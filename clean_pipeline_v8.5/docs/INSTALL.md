# Installation Guide

## System Requirements

- Linux or macOS
- 16 GB RAM minimum (32 GB recommended)
- 50 GB free disk space
- 20+ CPU cores recommended

## Step 1: Install SLiM 5

### From source (recommended)
```bash
wget https://github.com/MesserLab/SLiM/releases/download/v4.2.2/SLiM.zip
unzip SLiM.zip
cd SLiM
mkdir build && cd build
cmake ..
make
sudo make install
```

### Verify installation
```bash
slim --version  # Should show SLiM version 4.x or 5.x
```

## Step 2: Install R and Packages

```bash
# Install R (version 4.0+)
sudo apt-get install r-base r-base-dev  # Ubuntu/Debian
# or
brew install r  # macOS

# Install R packages
R -e 'install.packages(c("poolfstat", "dplyr", "reticulate"))'
```

## Step 3: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv npe_env
source npe_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Verify Installation

```bash
# Test SLiM
slim --version

# Test R packages
Rscript -e "library(poolfstat); library(dplyr)"

# Test Python packages
python3 -c "import numpy, torch, tskit, msprime; print('All packages OK')"
```

## Step 5: Place Your Data

Copy your Pool-seq RData file:
```bash
cp /path/to/your/Pooldata_demography.RData observed_data/
```

## Quick Test

Run a quick simulation test:
```bash
python3 python/simulate.py \
  --config config/config_pod.yaml \
  --out test_sim.npz \
  --n 5 \
  --workers 2
```

If successful, you'll see:
```
Saved simulations to test_sim.npz  (X: (5, 102), Theta: (5, 24))
```

## Troubleshooting

### SLiM not found
```bash
# Add to PATH
export PATH=$PATH:/usr/local/bin
# or specify full path in config
slim_binary: "/full/path/to/slim"
```

### R package install fails
```bash
# Install system dependencies
sudo apt-get install libssl-dev libcurl4-openssl-dev
```

### PyTorch install fails
```bash
# Install CPU-only version (smaller)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Out of memory
- Reduce `num_workers` in config
- Use smaller `genome_length`
- Close other applications

## Next Steps

1. Run POD test: `bash scripts/run_pod_test.sh`
2. If successful, run production: `bash scripts/run_production.sh`

## Multi-node execution (no scheduler)

If you have multiple SSH-accessible nodes (e.g. `geu-master`, `geu-worker1`, `geu-worker2`) and a shared filesystem, you can split the simulation step across nodes.

### One-time SSH setup
On the node where you launch the workflow:

```bash
ssh-keygen -t rsa -b 4096
ssh-copy-id geu-worker1
ssh-copy-id geu-worker2
```

(Optional but recommended) ensure SSH uses your key for all nodes:

```bash
cat > ~/.ssh/config <<'EOF'
Host geu-master geu-worker1 geu-worker2
  User $USER
  IdentityFile ~/.ssh/id_rsa
  IdentitiesOnly yes
  ControlMaster auto
  ControlPersist 10m
  ControlPath ~/.ssh/cm-%h
EOF
chmod 600 ~/.ssh/config
```

### Run production workflow across nodes
Use the multi-node script:

```bash
bash scripts/run_production_multinode.sh
```

By default it uses:
- Nodes: `geu-master geu-worker1 geu-worker2` (override with `NODES="..."`)
- Workers per node: `70` (override with `WORKERS_PER_NODE=...`)

Example:

```bash
NODES="geu-master geu-worker1 geu-worker2" WORKERS_PER_NODE=70 bash scripts/run_production_multinode.sh
```
