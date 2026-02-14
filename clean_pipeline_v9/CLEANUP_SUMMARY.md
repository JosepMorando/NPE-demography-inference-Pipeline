# Pipeline Cleanup Summary

## Overview

This document summarizes the cleanup performed on the NPE demography inference pipeline to focus exclusively on the full population set (individual populations model) for both production and POD testing.

## Changes Made

### 1. Configuration Files Fixed

**Updated to use correct population groups:**
- `config/config_individual_pops.yaml` - Changed `groups_csv` from `groups_11pops.csv` to `groups_12individuals.csv`
- `config/config_pod_individual.yaml` - Changed `groups_csv` from `groups_11pops.csv` to `groups_12individuals.csv`

### 2. Scripts Updated

**POD test scripts fixed:**
- `scripts/run_pod_test_individual.sh` - Updated file validation to check for `groups_12individuals.csv`
- `scripts/run_pod_test_multinode_individual.sh` - Updated file validation to check for `groups_12individuals.csv`

**New production scripts created:**
- `scripts/run_production_individual.sh` - Single-node production pipeline for individual populations
- `scripts/run_production_multinode_individual.sh` - Multi-node production pipeline for individual populations

### 3. Files Deleted

**Grouped model files (6-cluster model) removed:**
- `config/config_production.yaml` - 6-cluster production config
- `config/config_pod.yaml` - 6-cluster POD config
- `config/groups_6clusters.csv` - 6-cluster population mapping
- `config/groups_11pops.csv` - 11-population mapping (incorrect for 12-pop model)
- `templates/model.slim.tpl` - Grouped model SLiM template
- `scripts/run_production.sh` - Grouped model production script
- `scripts/run_production_multinode.sh` - Grouped model multi-node production script
- `scripts/run_production_multinode_reuse.sh` - Grouped model multi-node with reuse
- `scripts/run_pod_test.sh` - Grouped model POD test script
- `scripts/run_pod_test_multinode.sh` - Grouped model multi-node POD test

**Unnecessary utility files removed:**
- `bench_workers.py` - Worker benchmarking utility
- `bench_40.npz` - Benchmark results (40 workers)
- `bench_60.npz` - Benchmark results (60 workers)
- `inspect_nans.py` - Debug script for NaN detection
- `patch_ppc.py` - One-time patching script
- `debug.npz` - Debug data file
- `CHANGELOG_v8.4.md` - Old changelog
- `test_individual_pops.py` - Test script
- `clean_pipeline_v9/master_kill.txt` - Duplicate kill script

**Root-level cleanup:**
- `master_kill.txt` - Emergency kill script
- `v8_bug_report.md` - Bug report for old version

### 4. Remaining Pipeline Structure

**Configuration files:**
- `config/config_individual_pops.yaml` - Production config (100k sims, 12 populations, 59 parameters)
- `config/config_pod_individual.yaml` - POD testing config (50k sims)
- `config/groups_12individuals.csv` - 12-population mapping (including P001 outgroup)

**Templates:**
- `templates/model_individual_pops.slim.tpl` - Individual populations SLiM model

**Pipeline scripts:**
- `scripts/run_production_individual.sh` - Production single-node
- `scripts/run_production_multinode_individual.sh` - Production multi-node (with --reuse support)
- `scripts/run_pod_test_individual.sh` - POD testing single-node
- `scripts/run_pod_test_multinode_individual.sh` - POD testing multi-node (with --reuse support)

**Utility scripts (kept):**
- `scripts/generate_pod.py` - POD generation
- `scripts/check_pod_recovery.py` - Parameter recovery analysis
- `scripts/check_obs_vs_sim.py` - Observed vs simulated comparison
- `scripts/test_transforms.py` - Data transformation tests
- `scripts/benchmark_performance.sh` - Performance benchmarking
- `validate_config.sh` - Configuration validation

## Population Model

The pipeline now exclusively uses the **Individual Populations Model** with:

### Populations Modeled:
1. **P001** (outgroup/ancestor) - Not in observed data but necessary for tree root
2. **Observed populations** (11 populations):
   - BG01, BG05, BG04, BG07, Sauva, Montsenymid, Carlac, Conangles, Viros, Cimadal, Coscollet

### Phylogeny:
```
(P001,(BG01,((((BG05,BG04),BG07),Sauva),(Montsenymid,((Carlac,(Conangles,Viros)),(Cimadal,Coscollet)))))))
```

### Ghost/Node Populations:
The SLiM model includes 11 ghost/node populations (p0, p2, p4-p7, p12, p14-p15, p17, p20) that are **necessary** for modeling the phylogenetic tree structure. These populations:
- Represent internal nodes of the phylogenetic tree
- Are set to N=1 after splitting to minimize memory usage
- Are NOT sampled in the output
- Cannot be removed without breaking the tree topology

### Parameters:
- **59 free parameters total:**
  - 11 split times (T_P001, T_BG01, T_MAJOR_SPLIT, etc.)
  - 12 population sizes (N_P001, N_BG01, N_BG05, etc.)
  - 11 fixed ghost population sizes (N0, N_Node1-10, all fixed at 10,000)
  - Optional: bottleneck parameters (per-population mode)

## Validation

All configurations validated successfully:
```bash
cd clean_pipeline_v9
bash validate_config.sh
```

Results:
- ✓ Configuration file exists
- ✓ Groups CSV exists (13 lines: header + 12 populations)
- ✓ SLiM template exists (194 lines)
- ✓ YAML syntax is valid
- ✓ Coverage threshold is 10
- ✓ Population order is defined (11 observed populations)
- ✓ Template path references individual populations model

## Usage

### POD Testing (Recommended First):
```bash
# Single-node
bash scripts/run_pod_test_individual.sh

# Multi-node
bash scripts/run_pod_test_multinode_individual.sh

# Multi-node with reuse
bash scripts/run_pod_test_multinode_individual.sh --reuse /path/to/existing/sim_data.npz
```

### Production Analysis:
```bash
# Single-node
bash scripts/run_production_individual.sh

# Multi-node (faster)
bash scripts/run_production_multinode_individual.sh

# Multi-node with reuse
bash scripts/run_production_multinode_individual.sh --reuse /path/to/existing/sim_data.npz
```

## Notes

- **Coverage threshold**: Set to 10x (configurable in config files)
- **Population order**: Excludes P001 outgroup from summary statistics (only used for tree root)
- **Simulation budget**: 100k for production, 50k for POD testing
- **Model architecture**: NSF (Neural Spline Flow) with hidden_sizes [256, 256] for production
- **Bottlenecks**: Enabled with per-population mode (each of 12 populations can have individual bottlenecks)

## Git Branch

All changes committed to branch: `claude/cleanup-prod-pod-pipeline-Jv8hv`
