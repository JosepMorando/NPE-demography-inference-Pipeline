# Publishable Results Roadmap (v8.1)

This roadmap outlines the **recommended end-to-end workflow** for producing
publishable demographic inference results using the v8.1 pipeline in
`final_v8/clean_pipeline_v8.1`.

---

## Phase 0 — Prerequisites & Environment

1. **Install dependencies** following `docs/INSTALL.md`.
2. **Verify SLiM** (v5.x compatible) and R packages.
3. **Confirm data layout**:
   - `observed_data/Pooldata_demography.RData`
   - `config/groups_6clusters.csv`
   - config files in `config/`

---

## Phase 1 — Sanity & POD Validation (Required)

**Goal:** Confirm the pipeline can recover known parameters before using real data.

1. **Run POD test (single node)**:
   ```bash
   bash scripts/run_pod_test.sh
   ```
   This generates pseudo-observed data, trains the NSF model, and checks recovery.

2. **Interpret POD outputs**:
   - Check coverage from `pod_test/results/` (ideally ≥ 85–90%).
   - Inspect recovery plots if generated.

3. **If coverage is poor**:
   - Increase `npe.n_sims` in `config/config_pod.yaml`.
   - Rerun POD until recovery stabilizes.

**Publishable threshold:** Only proceed if POD recovery is acceptable and stable.

---

## Phase 2 — Production Simulations (Real Data)

**Goal:** Generate a large, clean simulation bank for inference.

1. **Review `config/config_production.yaml`**:
   - Prior ranges and population sizes
   - Bottleneck mode and parameters
   - Simulation scaling and genome length

2. **Run production pipeline**:
   ```bash
   bash scripts/run_production.sh
   ```

3. **Optional multi-node run**:
   ```bash
   bash scripts/run_production_multinode.sh
   ```
   The multi-node scripts now assign **unique seed offsets** per node to avoid
   duplicated simulations.

**Publishable threshold:** Use ≥150k simulations for strong posterior quality,
and consider 500k for high confidence if compute allows.

---

## Phase 3 — Posterior Diagnostics (Required)

**Goal:** Ensure the posterior is calibrated and interpretable.

1. **Inspect posterior samples**:
   - `results/posterior_samples.npz`
   - `results/posterior_summary.json`

2. **Run posterior predictive checks (PPC)**:
   ```bash
   python3 python/validate_ppc.py \
     --config config/config_production.yaml \
     --model models/nsf_model.pt \
     --obs observed_data/observed_summaries.npz \
     --out results/ppc_results.npz
   ```

3. **Run SBC (Simulation-Based Calibration)**:
   ```bash
   python3 python/validate_sbc.py \
     --model models/nsf_model.pt \
     --simulations simulations/sim_data.npz \
     --out results/sbc_results.npz
   ```

**Publishable threshold:** PPC should show reasonable fit to observed summaries,
and SBC should not show strong calibration failures.

---

## Phase 4 — Sensitivity & Robustness (Recommended)

**Goal:** Demonstrate robustness to modeling choices.

1. **Priors sensitivity**:
   - Widen/narrow key priors and check posterior stability.

2. **Bottleneck mode comparison**:
   - Compare `shared` vs `per_population` (if biologically plausible).

3. **Replicates**:
   - Run the full pipeline with different random seeds.

**Publishable threshold:** Results should be qualitatively consistent across
reasonable prior choices and seeds.

---

## Phase 5 — Figures & Reporting

**Goal:** Generate final figures and results for manuscript.

1. **Posterior visualizations**:
   - Marginal densities for times and Ne
   - Pairwise correlations and joint plots
   - Highest posterior density intervals

2. **Model fit**:
   - PPC plots of observed vs posterior predictive summaries
   - SBC rank histograms

3. **Narrative**:
   - Link demographic events to historical/biogeographic context
   - Justify priors and summarize uncertainty

---

## Checklist for “Publishable-Ready”

- [ ] POD recovery passes (≥85–90% coverage).
- [ ] Simulation bank is large enough (≥150k, ideally 500k).
- [ ] PPC shows acceptable fit to observed summaries.
- [ ] SBC indicates adequate calibration.
- [ ] Sensitivity analyses performed.
- [ ] All results reproducible with documented configs and seeds.

---

## Reproducibility Notes

- Record the exact `config/*.yaml` files used.
- Archive `models/nsf_model.pt`, `simulations/sim_data.npz`, and `results/`.
- Note the SLiM and package versions.

---

## Suggested Next Steps

If any checklist item fails, iterate on:
- Increasing `n_sims`
- Adjusting priors or model complexity
- Re-running POD validation

This roadmap is designed to support **defensible, publishable** results with a
clear validation trail.
