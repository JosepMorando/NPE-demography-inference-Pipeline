# Individual Populations Model - Documentation

## Overview

This document describes the individual populations model for the NPE demography inference pipeline. This model infers demographic parameters (split times, effective population sizes, and bottlenecks) for each individual population rather than grouped populations.

## Phylogenetic Structure

The individual populations model follows this phylogeny:

```
((P001,(BG01,((((BG05,BG04),BG07),Sauva),(Montsenymid,((Carlac,(Conangles,Viros)),(Cimadal,Coscollet)))))))
```

### Tree Structure

```
p0 (Ancestral)
├─ p1 (P001)
└─ p2 (Node1, ghost)
   ├─ p3 (BG01)
   └─ p4 (Node2, ghost)
      ├─ p5 (Node3, ghost - Southern clade)
      │  ├─ p11 (Sauva)
      │  └─ p6 (Node4, ghost - Eastern group)
      │     ├─ p10 (BG07)
      │     └─ p7 (Node5, ghost)
      │        ├─ p8 (BG05)
      │        └─ p9 (BG04)
      └─ p12 (Node6, ghost - Northern clade)
         ├─ p13 (Montsenymid)
         └─ p14 (Node7, ghost - Pyrenees)
            ├─ p15 (Node8, ghost - Western Pyrenees)
            │  ├─ p16 (Carlac)
            │  └─ p17 (Node9, ghost)
            │     ├─ p18 (Conangles)
            │     └─ p19 (Viros)
            └─ p20 (Node10, ghost - Eastern Pyrenees)
               ├─ p21 (Cimadal)
               └─ p22 (Coscollet)
```

### Sampled Populations (12)

1. **P001** (p1) - Basal lineage
2. **BG01** (p3) - Early diverging lineage
3. **BG05** (p8) - Eastern group
4. **BG04** (p9) - Eastern group
5. **BG07** (p10) - Eastern group
6. **Sauva** (p11) - Southern lineage
7. **Montsenymid** (p13) - Mid-elevation southern
8. **Carlac** (p16) - Western Pyrenees
9. **Conangles** (p18) - Western Pyrenees
10. **Viros** (p19) - Western Pyrenees
11. **Cimadal** (p21) - Eastern Pyrenees
12. **Coscollet** (p22) - Eastern Pyrenees

### Ghost Populations (10)

These intermediate populations are required for the phylogenetic structure but are not sampled. Their effective population sizes are fixed to N=10,000 to reduce model complexity.

- **Node1** (p2) - Ancestor of BG01 and rest
- **Node2** (p4) - Ancestor of southern and northern clades
- **Node3** (p5) - Southern clade ancestor
- **Node4** (p6) - Ancestor of BG05/BG04/BG07
- **Node5** (p7) - Ancestor of BG05 and BG04
- **Node6** (p12) - Northern clade ancestor
- **Node7** (p14) - Pyrenees ancestor
- **Node8** (p15) - Western Pyrenees ancestor
- **Node9** (p17) - Ancestor of Conangles/Viros
- **Node10** (p20) - Eastern Pyrenees ancestor

## Configuration Files

### 1. Main Configuration: `config/config_individual_pops.yaml`

Key features:
- **Coverage threshold**: 10 (easily configurable via `observed.target_cov`)
- **Template**: `templates/model_individual_pops.slim.tpl`
- **Population order**: All 12 sampled populations
- **Bottleneck inference**: Enabled with per-population mode

### 2. Population Mapping: `config/groups_12individuals.csv`

Maps each population to itself (no grouping):

```csv
Pop,Group
P001,P001
BG01,BG01
BG05,BG05
BG04,BG04
BG07,BG07
Sauva,Sauva
Montsenymid,Montsenymid
Carlac,Carlac
Conangles,Conangles
Viros,Viros
Cimadal,Cimadal
Coscollet,Coscollet
```

### 3. SLiM Template: `templates/model_individual_pops.slim.tpl`

Implements the full phylogenetic structure with:
- 23 total subpopulations (12 sampled + 10 ghost + 1 ancestral)
- Hierarchical splits following the phylogeny
- Support for bottleneck, expansion, and migration injection

## Parameters Inferred

### Split Times (11 parameters)

All times are in biological generations (scaled internally for SLiM):

1. **T_P001** - Root split (P001 vs rest)
2. **T_BG01** - BG01 divergence
3. **T_MAJOR_SPLIT** - Southern vs northern clades
4. **T_Sauva** - Sauva divergence
5. **T_BG07** - BG07 divergence
6. **T_BG05_BG04** - BG05/BG04 split
7. **T_Montsenymid** - Montsenymid divergence
8. **T_PYRENEES** - Pyrenean split
9. **T_Carlac** - Carlac divergence
10. **T_Conangles_Viros** - Conangles/Viros split
11. **T_Cimadal_Coscollet** - Cimadal/Coscollet split

### Time Constraints

The following phylogenetic constraints are enforced during prior sampling:

```
T_P001 < T_BG01 < T_MAJOR_SPLIT
T_MAJOR_SPLIT < T_Sauva < T_BG07 < T_BG05_BG04  (southern branch)
T_MAJOR_SPLIT < T_Montsenymid < T_PYRENEES  (northern branch)
T_PYRENEES < T_Carlac < T_Conangles_Viros  (western Pyrenees)
T_PYRENEES < T_Cimadal_Coscollet  (eastern Pyrenees)
```

### Effective Population Sizes (12 + 11 parameters)

**Sampled populations (12 free parameters):**
- N_P001, N_BG01, N_BG05, N_BG04, N_BG07, N_Sauva
- N_Montsenymid, N_Carlac, N_Conangles, N_Viros, N_Cimadal, N_Coscollet
- Prior: loguniform(2,500, 150,000)

**Ghost populations (11 fixed parameters):**
- N0, N_Node1, N_Node2, N_Node3, N_Node4, N_Node5
- N_Node6, N_Node7, N_Node8, N_Node9, N_Node10
- Fixed value: 10,000

### Bottleneck Parameters (per-population mode)

When `demography_extras.enable: true` and `bottleneck.mode: per_population`:

For each of the 12 sampled populations:
- **BN_TIME_FRAC_{POP}**: When bottleneck starts (fraction of branch length)
  - Prior: uniform(0.2, 0.6)
- **BN_SIZE_FRAC_{POP}**: Bottleneck intensity (fraction of base size)
  - Prior: loguniform(0.05, 0.3)
- **BN_DUR_{POP}**: Bottleneck duration in generations
  - Prior: discrete_uniform(10, 100)

Total bottleneck parameters: 12 × 3 = 36 parameters

## Total Parameter Count

- Split times: 11
- Population sizes (free): 12
- Population sizes (fixed): 11
- Bottleneck parameters (optional): 36

**Total free parameters (with bottlenecks enabled):** 11 + 12 + 36 = **59 parameters**

## Changing the Coverage Threshold

The coverage threshold can be easily changed by modifying the configuration file:

```yaml
observed:
  target_cov: 10  # Change this value
```

Common values:
- **10**: Lower coverage, more data retention, less strict
- **20**: Production setting (original model)
- **30**: High coverage, stricter filtering

## Running the Pipeline

### 1. Compute Observed Summary Statistics

```bash
Rscript r/compute_observed_summaries.R \
  --pooldata observed_data/Pooldata_demography.RData \
  --object filt.pooldata \
  --groups config/groups_12individuals.csv \
  --target_cov 10 \
  --out observed_data/observed_summaries_individual.npz
```

### 2. Generate Training Simulations

```bash
python python/simulate.py \
  --config config/config_individual_pops.yaml \
  --n 100000 \
  --out simulations/sim_data_individual.npz \
  --workers 60
```

### 3. Train NSF Model

```bash
python python/train_npe.py \
  --config config/config_individual_pops.yaml \
  --simulations simulations/sim_data_individual.npz \
  --out models/nsf_model_individual.pt
```

### 4. Infer Posterior

```bash
python python/infer_posterior.py \
  --model models/nsf_model_individual.pt \
  --obs observed_data/observed_summaries_individual.npz \
  --out results/posterior_individual.npz
```

## Validation

A validation script is provided to check the configuration:

```bash
bash validate_config.sh
```

This checks:
1. Configuration file exists and is valid YAML
2. Groups CSV exists and has correct format
3. SLiM template exists
4. Coverage threshold is set correctly
5. Population order is defined
6. Template path references the individual populations model

## Model Detection

The code automatically detects which model (grouped vs. individual) is being used based on the template filename:
- If `slim_template` contains "individual" → Individual populations model
- Otherwise → Grouped populations model

This allows both models to coexist and be used with the same codebase.

## Improvements Made

1. **Configurable coverage threshold**: Easy to change via config file
2. **Per-population inference**: Each population modeled independently
3. **Bottleneck inference**: Individual bottleneck parameters for each population
4. **Automatic model detection**: Code adapts based on configuration
5. **Backward compatibility**: Original grouped model still works
6. **Comprehensive validation**: Test scripts to verify configuration

## Technical Notes

### Adaptive Scaling

The pipeline uses adaptive scaling to make large effective population sizes computationally tractable:
- Minimum scaled population: 50 diploids
- Maximum scale factor: 100×
- Recombination rate scaled up to compensate

### Summary Statistics

The same summary statistics are computed as in the grouped model:
- 1D site frequency spectrum per population
- FST and nucleotide diversity (π)
- Folded 2D-SFS between population pairs
- Higher-order moments (H3)

### Neural Architecture

- Model: Neural Spline Flow (NSF) with masked autoregressive flows
- Hidden layers: [256, 256]
- Flow layers: 5
- Spline bins: 8
- Training: 100k simulations, 200 epochs maximum

## References

See main pipeline documentation for general NPE demography inference methodology.

---

**Last updated:** 2024
**Configuration version:** Individual populations v1
