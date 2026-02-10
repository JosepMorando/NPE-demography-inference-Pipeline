# Changelog - Clean Pipeline

## v6 — Multi-node Simulation Runner (No scheduler)

### Added
- `scripts/run_production_multinode.sh`: production workflow that splits the simulation step across multiple SSH-accessible nodes and merges results.
- `python/merge_sim_npz.py`: utility to merge simulation parts into a single `simulations/sim_data.npz` (concatenates `X` and `Theta`, preserves `theta_keys`, `pop_order`, and `meta_json`).

### Changed
- **Nothing else.** Training, inference, priors, summaries, scaling, and configs are unchanged. Only the orchestration of Step 2 (simulation generation) was adapted for multi-node execution.

---
## v5 — Model Simplification & Adaptive Scaling

### Model Simplification (24 → 18 free parameters)

Based on POD recovery analysis showing several parameters were unidentifiable:

**Fixed parameters (removed from NPE inference):**
- `N0` (ghost ancestor): 72% bias, fixed at 50,000
- `N_CORE` (unsampled node): 34% bias, fixed at 20,000

**Removed processes:**
- `EXP_RATE` + `EXP_START_FRAC`: expansion rate was completely unidentifiable
  (posterior = prior, -104% relative bias)
- `MIG_M` + `MIG_START_FRAC`: migration rate unidentifiable (248% CV, CI spanning zero)

**Kept N_INT** as free despite being an unsampled node — it showed remarkable
recovery (0.2% bias, 5.5% CV), suggesting the data genuinely constrains it.

**Result: 18 free parameters** = 8 split times + 7 pop sizes + 3 bottleneck params

### Adaptive Scaling

Replaced fixed `scale_factor: 50` with adaptive per-simulation scaling:
- Config key: `max_scale_factor` (default 200)
- Each simulation uses: `scale = min(max_scale_factor, min_Ne / safe_min_diploids)`
- When all populations are large, scale can reach 200x → faster simulations
- When a small population is drawn, scale automatically reduces to keep it safe
- Backward compatible: old `scale_factor` key is treated as `max_scale_factor`

### Config Infrastructure

- `priors.sizes` supports `{dist: fixed, value: X}` — passed to SLiM, excluded from NPE
- `demography_extras.expansion.enable` / `migration.enable` for independent control
- `build_theta_keys(cfg)` dynamically determines free parameters from config

---

## What Was Changed (v4)

### Cleaned Up
- ✅ Removed unnecessary duplicate scripts (`simulate.py`, `simulate_efficient.py` → kept memory-efficient `simulate.py`)
- ✅ Removed debug scripts (`debug_slim_simple.py`, `test_fixed_constraints.py`)
- ✅ Consolidated configs into two clear versions: production and POD
- ✅ Organized all utility scripts into `scripts/` directory
- ✅ Created proper documentation structure in `docs/`

### Updated Parameters

#### Production Config (`config_production.yaml`)
**Realistic beech parameters:**
- `burnin`: 500 → **10,000** (adequate burn-in)
- `gens`: 2,000 → **30,000** (realistic timescale)
- `genome_length`: 50kb → **100 Mb** (chromosome-scale)
- `target_snps`: 5,394 → **5,000** (typical pool-seq)
- `samples_per_group_haploid`: 20 → **40** (better coverage)
- `N0`: 2,000-4,000 → **50,000-500,000** (realistic beech Ne)
- `N_derived`: 200-2,000 → **5,000-100,000** (derived populations)
- `T_max`: 1,900 → **10,000** (deeper time)
- `n_sims`: 1,000 → **50,000** (better inference)
- `hidden_sizes`: [256,256,256] → **[512,512,512]** (more capacity)
- `n_components`: 8 → **16** (multimodal posteriors)
- `n_posterior_samples`: 50,000 → **100,000** (precise CIs)

#### POD Config (`config_pod.yaml`)
**Fast testing parameters:**
- `burnin`: 500 (kept)
- `gens`: 2,000 (kept)
- `genome_length`: **10 Mb** (10× faster)
- `target_snps`: **1,000** (faster computation)
- `N_range`: **1,000-50,000** (simpler)
- `T_range`: **100-1,500** (narrower)
- `n_sims`: **5,000** (quick test)
- `hidden_sizes`: **[256,256]** (smaller network)
- `n_posterior_samples`: **10,000** (quick check)

### New Features
- ✅ Separate production and POD workflows with shell scripts
- ✅ Comprehensive README with quick start guide
- ✅ Installation documentation
- ✅ Proper .gitignore for version control
- ✅ Requirements.txt for Python dependencies
- ✅ Clear directory structure

### Fixed Issues
- ✅ PyTorch 2.6 `weights_only` compatibility in `infer_posterior.py`
- ✅ Phylogenetic constraints properly enforced in `priors.py`
- ✅ Expansion rate overflow protection
- ✅ Memory efficiency with memory-mapped simulation

## Migration from Old Pipeline

If you have existing results:

```bash
# Copy your data
cp old_pipeline/observed_data/Pooldata_demography.RData clean_pipeline/observed_data/

# Copy existing results (if compatible)
cp -r old_pipeline/simulations clean_pipeline/  # if using same params
cp -r old_pipeline/models clean_pipeline/       # if trained with compatible config
```

## Speed Improvements

- **POD test**: 2-3 hours (was 6+ hours with old settings)
- **Production**: 6-12 hours for 50k sims (configurable)
- **Memory**: ~2 GB max (was >250 GB before memory-mapping fix)

## Parameter Rationale

### Why these beech values?

**Generation time (50 years):**
- Long-lived tree species
- Age at first reproduction: 30-40 years
- Maximum age: 300+ years

**Effective population size (5k-500k):**
- Large geographic range
- High density in suitable habitat
- Moderate to high gene flow

**Timescale (500-10k generations = 25ka-500ka):**
- Covers Last Glacial Maximum (~20ka)
- Multiple glacial cycles
- Post-glacial expansion

**Mutation rate (7.77 × 10⁻⁹):**
- Typical plant nuclear genome rate
- Calibrated from related species

## Backward Compatibility

The clean pipeline is **compatible** with:
- ✅ Existing Pool-seq RData files
- ✅ SLiM 5.x
- ✅ Python 3.8+
- ✅ R 4.0+

**Not compatible** with:
- ❌ Old config files (need to migrate to new format)
- ❌ SLiM 3.x (use SLiM 4+ or 5)
- ❌ Models trained with very different parameter ranges
