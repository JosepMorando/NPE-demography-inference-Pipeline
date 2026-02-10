# Publishable Results Roadmap (v8.4)

This roadmap describes the path from validated pipeline runs to publishable
results using:

- `final_v8.4/clean_pipeline_v8.4`

## Phase 1 — Reproducible baseline

1. Run `bash scripts/run_pod_test.sh` and archive logs.
2. Confirm POD recovery with `scripts/check_pod_recovery.py` outputs.
3. Run SBC/PPC checks and store metrics in `results/`.

## Phase 2 — Production inference

1. Generate final observed summaries from curated Pool-seq inputs.
2. Run production simulations and train final NSF model.
3. Infer posterior and export posterior summaries.

## Phase 3 — Reporting

1. Consolidate parameter tables and uncertainty intervals.
2. Add reproducibility appendix (config hashes, command logs).
3. Draft figures and biological interpretation.

## Definition of done

- POD/SBC/PPC checks are archived and interpretable.
- Production run is reproducible from committed configs/scripts.
- Manuscript-ready tables and figures are generated from pipeline artifacts.
