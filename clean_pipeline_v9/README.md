# NPE Demography Inference Pipeline — v9

This is the active v9 pipeline.

## End-to-end workflow

### 1) Compute observed summaries (R)

```bash
Rscript r/compute_observed_summaries.R \
  --pooldata observed_data/Pooldata_demography.RData \
  --object filt.pooldata \
  --groups config/groups_6clusters.csv \
  --target_cov 20 \
  --out observed_data
```

Output: `observed_data/observed_summaries.npz`

### 2) Simulate training data (Python + SLiM)

```bash
python3 python/simulate.py \
  --config config/config_production.yaml \
  --out simulations/sim_data.npz \
  --workers 70
```

Output: `simulations/sim_data.npz`

### 3) Train posterior estimator (sbi NSF)

```bash
python3 python/train_npe.py \
  --config config/config_production.yaml \
  --simulations simulations/sim_data.npz \
  --out models/nsf_model.pt
```

Output: `models/nsf_model.pt`

### 4) Infer posterior on observed data

```bash
python3 python/infer_posterior.py \
  --model models/nsf_model.pt \
  --obs observed_data/observed_summaries.npz \
  --out results/posterior_samples.npz
```

Outputs:
- `results/posterior_samples.npz`
- `results/posterior_summary.json`

### 5) Validate quality

```bash
bash scripts/run_pod_test.sh
python3 python/validate_sbc.py --config config/config_pod.yaml --model models/nsf_model.pt
python3 python/validate_ppc.py --config config/config_pod.yaml --model models/nsf_model.pt
```

## Multi-node options

For SSH-accessible multi-node execution:

- `scripts/run_production_multinode.sh`
- `scripts/run_production_multinode_reuse.sh`
- `scripts/run_pod_test_multinode.sh`

## See also

- Install guide: `docs/INSTALL.md`
- Roadmap: `docs/ROADMAP.md`
- v8.4 changelog: `CHANGELOG_v8.4.md`
