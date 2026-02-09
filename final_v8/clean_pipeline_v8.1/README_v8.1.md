# NPE Demography Inference Pipeline — v8.1

This document describes the **current** pipeline (v8.1) shipped in
`final_v8/clean_pipeline_v8.1`. It combines SLiM simulations, Pool-seq summary
statistics, and neural posterior estimation using a **neural spline flow (NSF)**.

> **Version note:** v8.1 disables a time-ordering check in post-processing that
> produced false positives when using diagonal covariance models; see
> `CHANGELOG_v8.1.md` for full context.

---

## Pipeline Overview

### 1) Compute observed summary statistics (R)
Observed Pool-seq data are summarized in R so they match the simulated
statistics.

```bash
Rscript r/compute_observed_summaries.R \
  --pooldata observed_data/Pooldata_demography.RData \
  --object filt.pooldata \
  --groups config/groups_6clusters.csv \
  --target_cov 20 \
  --out observed_data
```

**Output:** `observed_data/observed_summaries.npz`

### 2) Run SLiM simulations (Python)
Simulate datasets under the demographic model defined in the config.

```bash
python3 python/simulate.py \
  --config config/config_production.yaml \
  --out simulations/sim_data.npz \
  --workers 70
```

**Output:** `simulations/sim_data.npz` containing:
- `X` (summary statistics)
- `Theta` (parameters)
- `theta_keys`, `pop_order`, `meta_json`

### 3) Train the NSF posterior estimator
Fit a neural spline flow to learn \( p(\theta | x) \).

```bash
python3 python/train_npe.py \
  --config config/config_production.yaml \
  --simulations simulations/sim_data.npz \
  --out models/nsf_model.pt
```

**Output:** `models/nsf_model.pt` (includes scalers and model metadata).

### 4) Infer posterior for observed data
Sample posterior parameters for the observed summary statistics.

```bash
python3 python/infer_posterior.py \
  --model models/nsf_model.pt \
  --obs observed_data/observed_summaries.npz \
  --out results/posterior_samples.npz
```

**Outputs:**
- `results/posterior_samples.npz` (posterior samples)
- `results/posterior_summary.json` (summary statistics and intervals)

### 5) Validate the pipeline
Use POD, simulation-based calibration (SBC), and posterior predictive checks (PPC):

```bash
bash scripts/run_pod_test.sh
```

This script runs:
- `scripts/generate_pod.py` (pseudo-observed data)
- `python/simulate.py` (training sims)
- `python/train_npe.py` (NSF training)
- `python/infer_posterior.py` (POD posterior)
- `scripts/check_pod_recovery.py` (coverage check)
- `python/validate_sbc.py` and `python/validate_ppc.py`

---

## Configuration Highlights

### `config/config_production.yaml`
- `simulation`: SLiM runtime settings, mutation overlay, scaling, and sample sizes.
- `priors`: time, size, and demographic extras (bottlenecks, expansion, migration).
- `npe`: number of sims, NSF architecture, and training hyperparameters.

### Bottlenecks (per-population)
The production config uses **per-population** bottleneck parameters so each
population can have its own timing/size/duration. See:
`priors.demography_extras.bottleneck` in the config.

---

## Outputs & Artifacts

| Stage | Output |
| --- | --- |
| Observed summaries | `observed_data/observed_summaries.npz` |
| Simulations | `simulations/sim_data.npz` |
| Trained model | `models/nsf_model.pt` |
| Posterior samples | `results/posterior_samples.npz` |
| Posterior summary | `results/posterior_summary.json` |

---

## Recommended Workflow

1. **Install dependencies** via `docs/INSTALL.md`.
2. **Run POD validation**: `bash scripts/run_pod_test.sh`.
3. **Run production**: `bash scripts/run_production.sh`.
4. **Inspect results** in `results/`.

---

## Additional Notes

- The NSF model includes standardized summary statistics and parameters; the
  scalers are stored inside the model checkpoint.
- For large runs, consider the multi-node scripts:
  `scripts/run_production_multinode.sh` or
  `scripts/run_production_multinode_reuse.sh`.
