# NPE Demography Inference Pipeline (v9)

This repository tracks the **current v9 pipeline** for demographic inference
in *Fagus* using:

- SLiM simulations
- R-based Pool-seq summary statistics
- Neural posterior estimation (NSF)
- POD/SBC/PPC validation

The active pipeline is located at:

- `final_v8.4/clean_pipeline_v8.4`

## Quick start

```bash
cd final_v8.4/clean_pipeline_v8.4

# 1) Validate setup and recovery first
bash scripts/run_pod_test.sh

# 2) Run production workflow
bash scripts/run_production.sh
```

## Documentation

- Pipeline usage: `final_v8.4/clean_pipeline_v8.4/README.md`
- Installation: `final_v8.4/clean_pipeline_v8.4/docs/INSTALL.md`
- Version notes: `final_v8.4/clean_pipeline_v8.4/CHANGELOG_v8.4.md`
- Development roadmap: `final_v8.4/clean_pipeline_v8.4/docs/ROADMAP.md`

## Repository layout

```text
final_v8.4/
  clean_pipeline_v8.4/
    config/      # production + POD configs
    python/      # simulation, training, inference, validation
    r/           # observed summary statistic generation
    scripts/     # orchestration scripts
    docs/        # install + roadmap docs
```
