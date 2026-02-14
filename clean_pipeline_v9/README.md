# NPE Demography Inference Pipeline — v9

This is the active v9 pipeline for demographic inference in *Fagus sylvatica* using the **Individual Populations Model**.

## Model Overview

**12 individual populations with full phylogenetic structure** (59 free parameters)

### Populations:
- **P001** - Outgroup/ancestor (not in observed data, used for tree root)
- **11 observed populations**: BG01, BG05, BG04, BG07, Sauva, Montsenymid, Carlac, Conangles, Viros, Cimadal, Coscollet

### Phylogeny:
```
(P001,(BG01,((((BG05,BG04),BG07),Sauva),(Montsenymid,((Carlac,(Conangles,Viros)),(Cimadal,Coscollet)))))))
```

### Parameters:
- **11 split times**: T_P001, T_BG01, T_MAJOR_SPLIT, T_Sauva, T_BG07, T_BG05_BG04, T_Montsenymid, T_PYRENEES, T_Carlac, T_Conangles_Viros, T_Cimadal_Coscollet
- **12 population sizes**: N_P001, N_BG01, N_BG05, N_BG04, N_BG07, N_Sauva, N_Montsenymid, N_Carlac, N_Conangles, N_Viros, N_Cimadal, N_Coscollet
- **Bottleneck parameters**: Per-population mode (time_fraction, size_fraction, duration_gens for each of 12 populations)

### Ghost/Node Populations:
The SLiM model includes 11 ghost populations (p0, p2, p4-p7, p12, p14-p15, p17, p20) representing internal phylogenetic nodes. These are **necessary** for modeling the tree structure, set to N=1 after splitting for memory efficiency, and not sampled in the output.

## Quick Start

### Recommended: POD Testing First

Validate the pipeline with pseudo-observed data before running on real data:

```bash
# Single-node (2-4 hours)
bash scripts/run_pod_test_individual.sh

# Multi-node (faster, ~30-60 minutes)
bash scripts/run_pod_test_multinode_individual.sh
```

### Production Analysis

Run the full pipeline on real Pool-seq data:

```bash
# Single-node (6-12 hours for 100k simulations)
bash scripts/run_production_individual.sh

# Multi-node (recommended, ~2-4 hours with 3 nodes)
bash scripts/run_production_multinode_individual.sh
```

## End-to-End Workflow (Manual Steps)

### 1) Compute observed summaries (R)

```bash
Rscript r/compute_observed_summaries.R \
  --pooldata observed_data/Pooldata_demography.RData \
  --object filt.pooldata \
  --groups config/groups_12individuals.csv \
  --target_cov 10 \
  --out observed_data
```

Output: `observed_data/observed_summaries.npz`

### 2) Simulate training data (Python + SLiM)

```bash
# Single-node
python3 python/simulate.py \
  --config config/config_individual_pops.yaml \
  --out simulations/sim_data.npz \
  --workers 30

# Multi-node: Use run_production_multinode_individual.sh for automatic distribution
```

Output: `simulations/sim_data.npz` (100,000 simulations for production)

### 3) Train posterior estimator (Neural Spline Flow)

```bash
python3 python/train_npe.py \
  --config config/config_individual_pops.yaml \
  --simulations simulations/sim_data.npz \
  --out models/nsf_model.pt
```

Output: `models/nsf_model.pt`

### 4) Infer posterior on observed data

```bash
python3 python/infer_posterior.py \
  --model models/nsf_model.pt \
  --obs observed_data/observed_summaries.npz \
  --out results/posterior_samples.npz \
  --config config/config_individual_pops.yaml
```

Outputs:
- `results/posterior_samples.npz` (20,000 posterior samples)
- `results/posterior_summary.json` (parameter estimates with credible intervals)

### 5) Validate quality

```bash
# Posterior predictive checks
python3 python/validate_ppc.py \
  --config config/config_individual_pops.yaml \
  --model models/nsf_model.pt \
  --obs observed_data/observed_summaries.npz \
  --out results/ppc_results.npz

# Simulation-based calibration
python3 python/validate_sbc.py \
  --config config/config_individual_pops.yaml \
  --model models/nsf_model.pt \
  --simulations simulations/sim_data.npz \
  --out results/sbc_results.npz
```

## Multi-Node Execution

For faster execution across multiple nodes via SSH:

### Environment Variables:
```bash
export NODES="geu-master geu-worker1 geu-worker2"  # Space-separated node names
export WORKERS_PER_NODE=70                          # Parallel workers per node
export ENABLE_SIM_COMPRESSION=0                     # 0=faster, 1=smaller files
```

### Scripts:
- `scripts/run_production_multinode_individual.sh` - Production pipeline (multi-node)
- `scripts/run_pod_test_multinode_individual.sh` - POD testing (multi-node)

Both support `--reuse /path/to/existing/sim_data.npz` to resume from partial simulations.

### Example:
```bash
# Run with 3 nodes, reusing existing 50k simulations
NODES="geu-master geu-worker1 geu-worker2" WORKERS_PER_NODE=70 \
  bash scripts/run_production_multinode_individual.sh --reuse simulations/sim_data_50k.npz
```

## Configuration Files

### Production (`config/config_individual_pops.yaml`):
- **n_sims**: 100,000 simulations
- **target_cov**: 10x coverage threshold
- **batch_size**: 1024
- **workers**: 60
- **hidden_sizes**: [256, 256]
- **n_posterior_samples**: 20,000

### POD Testing (`config/config_pod_individual.yaml`):
- **n_sims**: 50,000 simulations (lighter for validation)
- **target_cov**: 10x coverage threshold
- **batch_size**: 512
- **workers**: 30
- **hidden_sizes**: [128, 128] (lighter architecture)
- **n_posterior_samples**: 10,000

## Validation

Run configuration validation:
```bash
bash validate_config.sh
```

Checks:
- ✓ Configuration files exist
- ✓ Groups CSV exists (13 lines: header + 12 populations)
- ✓ SLiM template exists
- ✓ YAML syntax is valid
- ✓ Coverage threshold is 10
- ✓ Population order is defined
- ✓ Template path references individual populations model

## Documentation

- **Cleanup summary**: `CLEANUP_SUMMARY.md` - Recent pipeline cleanup and changes
- **Individual populations guide**: `INDIVIDUAL_POPULATIONS_README.md` - Detailed model documentation
- **Install guide**: `docs/INSTALL.md` - Installation instructions
- **Roadmap**: `docs/ROADMAP.md` - Development roadmap

## Expected Runtime

### Single-node (30 workers):
- **POD testing** (50k sims): ~2-4 hours total
  - Simulations: 1-2 hours
  - Training: 30-60 minutes
  - Inference + validation: 30 minutes
- **Production** (100k sims): ~6-12 hours total
  - Simulations: 4-8 hours
  - Training: 1-3 hours
  - Inference: 30 minutes

### Multi-node (3 nodes, 70 workers each):
- **POD testing**: ~30-60 minutes total
- **Production**: ~2-4 hours total

## Interpreting Results

### POD Testing:
- **Excellent**: Coverage 90-100%
- **Good**: Coverage 85-90%
- **Warning**: Coverage 75-85% (consider more sims)
- **Poor**: Coverage <75% (increase n_sims or check pipeline)

### Production:
- Review `results/posterior_summary.json` for parameter estimates
- Check credible intervals and posterior distributions
- Validate with PPC to ensure model captures observed data patterns

## Troubleshooting

### High-dimensional inference (59 parameters):
- Ensure adequate simulation budget (100k recommended)
- Monitor training loss convergence
- Check SBC results for unbiased inference
- Use PPC to validate summary statistics quality

### Multi-node issues:
- Verify shared filesystem access across all nodes
- Check SSH connectivity without passwords
- Review logs in `logs/sim_*.log` for node-specific errors
- Ensure consistent Python/R environments across nodes
