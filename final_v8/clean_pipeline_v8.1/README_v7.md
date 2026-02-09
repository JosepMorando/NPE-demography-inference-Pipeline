# NPE Demographic Inference Pipeline - Version 7.0

Quick reference for v7 changes and usage.

## What's New?

1. **✅ Fixed constraint violations** - No more negative Ne/T or invalid fractions
2. **🆕 Per-population bottlenecks** - Independent bottleneck parameters per population

## Quick Start

```bash
# Production run with per-population bottlenecks
python python/simulate.py --config config/config_production.yaml
python python/train_npe.py --config config/config_production.yaml
python python/infer_posterior.py

# Validation
bash scripts/run_pod_test.sh
```

## Key Changes from v6

- **Transformations:** Log/logit transforms applied automatically (guarantees valid parameters)
- **Bottlenecks:** Choose `mode: shared` (v6 style) or `mode: per_population` (new)
- **Must retrain:** v6 models incompatible due to transform changes

See `CHANGELOG_v7.md` for complete details.
