# NPE Demography Inference Pipeline (v8.1)

This repository contains the **current** (v8.1) NPE pipeline for demographic
inference in *Fagus* using SLiM simulations, summary statistics, and a neural
spline flow (NSF) posterior estimator. The full, runnable pipeline lives under
`final_v8/clean_pipeline_v8.1`.

## Pipeline at a Glance

1. **Observed summaries**: compute Pool-seq summary statistics in R.
2. **Simulations**: run SLiM simulations and compute matching summaries.
3. **Train NPE**: fit a neural spline flow to infer `p(θ | x)`.
4. **Infer posterior**: sample posterior parameters (time ordering enforced via
   cumulative-gap parameterization).
4. **Infer posterior**: sample posterior parameters for observed data.
5. **Validate**: POD, SBC, and PPC checks.

## Where to Start

- **Pipeline overview & usage:** `final_v8/clean_pipeline_v8.1/README_v8.1.md`
- **Installation:** `final_v8/clean_pipeline_v8.1/docs/INSTALL.md`
- **Version history:** `final_v8/clean_pipeline_v8.1/CHANGELOG_v8.1.md`

## Quick Commands

```bash
# POD validation (recommended first)
cd final_v8/clean_pipeline_v8.1
bash scripts/run_pod_test.sh

# Production run (after POD passes)
bash scripts/run_production.sh
```

## Repository Layout

```
final_v8/clean_pipeline_v8.1/
  config/        # configs for production + POD
  python/        # simulation, training, inference, validation
  r/             # observed summary stats (Pool-seq)
  scripts/       # orchestration scripts
  docs/          # install + pipeline docs
```
