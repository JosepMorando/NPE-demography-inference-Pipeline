# NPE Demography Inference Pipeline (v9)

This repository tracks the **current v9 pipeline** for demographic inference
in *Fagus sylvatica* using:

- SLiM forward simulations
- R-based Pool-seq summary statistics
- Neural posterior estimation (Neural Spline Flow)
- POD/SBC/PPC validation

The active pipeline is located at:

- `clean_pipeline_v9/`

## Model

**Individual Populations Model** (12 populations, 59 free parameters)

- **P001** (outgroup/ancestor)
- **11 observed populations**: BG01, BG05, BG04, BG07, Sauva, Montsenymid, Carlac, Conangles, Viros, Cimadal, Coscollet

**Phylogeny:**
```
(P001,(BG01,((((BG05,BG04),BG07),Sauva),(Montsenymid,((Carlac,(Conangles,Viros)),(Cimadal,Coscollet)))))))
```

**Parameters:**
- 11 split times (T_P001, T_BG01, T_MAJOR_SPLIT, etc.)
- 12 population sizes (N_P001, N_BG01, N_BG05, etc.)
- Per-population bottleneck parameters (optional)

## Quick start

```bash
cd clean_pipeline_v9

# 1) Validate setup and recovery first (POD testing)
bash scripts/run_pod_test_individual.sh

# 2) Run production workflow
bash scripts/run_production_individual.sh

# For multi-node (faster):
bash scripts/run_pod_test_multinode_individual.sh
bash scripts/run_production_multinode_individual.sh
```

## Documentation

- Pipeline usage: `clean_pipeline_v9/README.md`
- Cleanup summary: `clean_pipeline_v9/CLEANUP_SUMMARY.md`
- Installation: `clean_pipeline_v9/docs/INSTALL.md`
- Development roadmap: `clean_pipeline_v9/docs/ROADMAP.md`

## Repository layout

```text
clean_pipeline_v9/
  config/                          # Configuration files
    config_individual_pops.yaml    # Production config (100k sims)
    config_pod_individual.yaml     # POD testing config (50k sims)
    groups_12individuals.csv       # Population mapping
  python/                          # Core pipeline modules
    npe_demography/                # Package modules
    simulate.py                    # Simulation orchestration
    train_npe.py                   # NSF training
    infer_posterior.py             # Posterior inference
    validate_sbc.py                # SBC validation
    validate_ppc.py                # PPC validation
    merge_sim_npz.py               # Multi-node merge utility
  r/                               # R scripts
    compute_observed_summaries.R   # Pool-seq statistics
  scripts/                         # Pipeline orchestration
    run_production_individual.sh            # Production (single-node)
    run_production_multinode_individual.sh  # Production (multi-node)
    run_pod_test_individual.sh              # POD testing (single-node)
    run_pod_test_multinode_individual.sh    # POD testing (multi-node)
    generate_pod.py                         # POD generation
    check_pod_recovery.py                   # Recovery analysis
  templates/                       # SLiM templates
    model_individual_pops.slim.tpl # Individual populations model
  observed_data/                   # Observed Pool-seq data
  docs/                            # Documentation
```
