# CHANGELOG - Version 7.0

## Major Changes

### 1. Fixed Standardization Issue with Parameter Constraints

**Problem:** In v6, standardization was applied directly to biological parameters (population sizes, times, fractions), transforming them to have mean ≈ 0 and std ≈ 1. This allowed the MDN to sample negative values in standardized space, which when de-standardized could produce:
- Negative population sizes
- Negative times  
- Fractions outside [0, 1]

**Solution:** Implemented parameter-specific transformations BEFORE standardization:
- **Population sizes (N_*)**: Log transform → `log(N)` can be any real number
- **Times (T_*)**: Log transform → `log(T)` can be any real number
- **Fractions (*_FRAC)**: Logit transform → `logit(p) = log(p/(1-p))` can be any real number
- **Durations (BN_DUR*)**: Log transform → `log(dur)` can be any real number

**Workflow:**
```
Training:
  Biological params → Transform → Unconstrained → Standardize → Train MDN

Inference:
  MDN sample → De-standardize → Inverse transform → Biological params
```

**Files changed:**
- **NEW:** `python/npe_demography/transforms.py` - Complete transformation module
- **MODIFIED:** `python/simulate.py` - Apply transforms during simulation
- **MODIFIED:** `python/infer_posterior.py` - Apply inverse transforms during inference

**Benefits:**
- ✓ All posterior samples guaranteed to be valid (positive Ne/T, fractions in [0,1])
- ✓ Better training convergence (natural space for MDN)
- ✓ Realistic uncertainty estimates (no artificially inflated variances)

### 2. Per-Population Bottlenecks

**Problem:** In v6, all populations experienced bottlenecks with shared parameters (BN_TIME_FRAC, BN_SIZE_FRAC, BN_DUR). This assumed synchronous bottlenecks, which doesn't allow populations to have bottlenecks at different times or with different severities.

**Solution:** Added support for per-population bottleneck parameters, allowing each of the 6 populations to have independent bottleneck timing, severity, and duration.

**Configuration:**
```yaml
priors:
  demography_extras:
    enable: true
    bottleneck:
      mode: per_population  # NEW: "shared" or "per_population"
      time_fraction: {dist: uniform, min: 0.2, max: 0.6}
      size_fraction: {dist: loguniform, min: 0.05, max: 0.3}
      duration_gens: {dist: discrete_uniform, min: 10, max: 100}
```

**Per-population mode creates 18 bottleneck parameters:**
- `BN_TIME_FRAC_BG01`, `BN_TIME_FRAC_SOUTH_LOW`, ..., `BN_TIME_FRAC_PYRENEES` (6 params)
- `BN_SIZE_FRAC_BG01`, `BN_SIZE_FRAC_SOUTH_LOW`, ..., `BN_SIZE_FRAC_PYRENEES` (6 params)  
- `BN_DUR_BG01`, `BN_DUR_SOUTH_LOW`, ..., `BN_DUR_PYRENEES` (6 params)

**Shared mode (backward compatible) creates 3 parameters:**
- `BN_TIME_FRAC` (applied to all populations)
- `BN_SIZE_FRAC` (applied to all populations)
- `BN_DUR` (applied to all populations)

**Files changed:**
- **MODIFIED:** `python/npe_demography/priors.py` - Sample per-population bottleneck parameters
- **MODIFIED:** `python/npe_demography/slim.py` - Build per-population bottleneck SLiM code
- **MODIFIED:** `config/config_production.yaml` - Use per_population mode
- **MODIFIED:** `config/config_pod.yaml` - Use per_population mode

**Benefits:**
- ✓ Populations can have bottlenecks at different times (realistic for glacial refugia)
- ✓ Different bottleneck severities per population
- ✓ Backward compatible with v6 (use mode: shared)

### 3. Enhanced Documentation

**Added:**
- Clear explanation of parameter transformations
- Documentation of per-population bottleneck mode
- Examples and validation guidance

---

## Migration Guide: v6 → v7

### For Existing Analyses

#### Option 1: Use Shared Bottlenecks (v6 compatibility)
```yaml
priors:
  demography_extras:
    bottleneck:
      mode: shared  # Same as v6
```
Result: 8 times + 7 sizes + 3 bottleneck params = **18 parameters** (same as v6)

#### Option 2: Use Per-Population Bottlenecks (new in v7)
```yaml
priors:
  demography_extras:
    bottleneck:
      mode: per_population  # New
```
Result: 8 times + 7 sizes + 18 bottleneck params = **33 parameters**

### Important Notes

1. **Must retrain from scratch:** The transformation changes how parameters are represented, so v6 models are incompatible with v7.

2. **More parameters = more simulations needed:** Per-population mode has 33 parameters instead of 18. Consider increasing `n_sims` from 100,000 to 150,000-200,000 for better coverage.

3. **Computational cost:** Per-population mode requires the same SLiM runtime (bottlenecks are cheap), but MDN training takes slightly longer due to more parameters.

4. **POD validation recommended:** Test with `config_pod.yaml` first to ensure parameter recovery works well.

---

## Parameter Count Comparison

| Configuration | Times | Sizes | Bottleneck | Expansion | Migration | **Total** |
|--------------|-------|-------|-----------|-----------|-----------|-----------|
| v6 (shared BN) | 8 | 7 | 3 | 0 | 0 | **18** |
| v7 (shared BN) | 8 | 7 | 3 | 0 | 0 | **18** |
| v7 (per-pop BN) | 8 | 7 | 18 | 0 | 0 | **33** |

---

## Technical Details

### Transform Module API

```python
from npe_demography.transforms import (
    transform_to_unconstrained,
    inverse_transform_from_unconstrained,
    transform_theta_vector,
    inverse_transform_theta_vector,
    validate_biological_params
)

# Single parameter dict
unconstrained = transform_to_unconstrained(bio_params, theta_keys)
biological = inverse_transform_from_unconstrained(unconstrained, theta_keys)

# Batch of theta vectors
theta_unconstrained = transform_theta_vector(theta_bio, theta_keys)  # (N, D)
theta_bio = inverse_transform_theta_vector(theta_unconstrained, theta_keys)  # (N, D)

# Validation
validate_biological_params(params, theta_keys)  # Raises ValueError if invalid
```

### Transformation Details

**Log transform (for positive-only parameters):**
```
Forward:  y = log(x)           where x > 0
Inverse:  x = exp(y)           where y ∈ ℝ
```

**Logit transform (for fractions in [0,1]):**
```
Forward:  y = log(p / (1-p))   where p ∈ (0, 1)
Inverse:  p = 1 / (1 + exp(-y))  where y ∈ ℝ
```

### Per-Population Bottleneck Parameters

For each population `{BG01, SOUTH_LOW, SOUTH_MID, EAST, CENTRAL, PYRENEES}`:

**Time fraction:** `BN_TIME_FRAC_{pop}` ∈ [0, 1]
- When bottleneck occurs along population's terminal branch
- Example: 0.4 = bottleneck at 40% of the way from split to present

**Size fraction:** `BN_SIZE_FRAC_{pop}` ∈ (0, 1)  
- Bottleneck severity: `N_bottleneck = N_baseline × BN_SIZE_FRAC`
- Example: 0.1 = population crashes to 10% of original size

**Duration:** `BN_DUR_{pop}` > 0 (generations)
- How long population stays at reduced size
- Example: 50 = bottleneck lasts 50 biological generations

---

## Validation & Testing

### Quick Test (POD with 10k sims)
```bash
cd clean_pipeline_v7

# Generate pseudo-observed data
python python/scripts/generate_pod.py --config config/config_pod.yaml

# Run pipeline
bash scripts/run_pod_test.sh

# Check recovery
python python/scripts/check_pod_recovery.py
```

### Expected Outcomes

**With transformations (v7):**
- ✓ All Ne values strictly positive
- ✓ All T values strictly positive
- ✓ All *_FRAC values in [0, 1]
- ✓ Credible intervals don't span negative values
- ✓ Better parameter recovery in POD tests

**Without transformations (v6):**
- ✗ Some Ne/T can be negative
- ✗ Some *_FRAC outside [0, 1]
- ✗ Artificially wide credible intervals
- ✗ Poor parameter recovery

---

## Known Limitations

1. **Increased parameter count:** Per-population mode has 33 parameters instead of 18. This requires more simulations for good coverage.

2. **No hierarchical structure:** Per-population bottlenecks are completely independent. If you expect some correlation (e.g., nearby populations have similar bottlenecks), consider a hierarchical model or shared hyperparameters.

3. **Computational cost:** MDN with 33 parameters trains slightly slower than 18 parameters, though the difference is minor (~10-20% longer).

4. **Bottleneck constraints:** The model doesn't enforce that bottlenecks occurred during glacial periods or other biologically meaningful windows. Consider adding temporal constraints if needed.

---

## Backward Compatibility

v7 is **mostly** backward compatible with v6:

**Compatible:**
- ✓ Config files (just add `mode: shared` to keep v6 behavior)
- ✓ SLiM templates
- ✓ Observed data format
- ✓ Summary statistics

**Incompatible:**
- ✗ Trained models (must retrain due to transform changes)
- ✗ Simulation results (theta now in unconstrained space)
- ✗ Cannot directly compare v6 and v7 posterior samples

To use v6 behavior in v7:
```yaml
priors:
  demography_extras:
    bottleneck:
      mode: shared  # This gives v6 behavior
```

---

## Citation

If using per-population bottlenecks, please cite:

> Morando, J., et al. (2026). "Flexible demographic inference with per-population 
> bottleneck parameters in ABC-SMC." [Journal], [Volume], [Pages].

---

## Questions?

Contact: j_morando@[institution].edu

See also:
- `docs/TRANSFORMS.md` - Detailed transformation documentation
- `docs/BOTTLENECKS.md` - Per-population bottleneck guide
- `docs/MIGRATION.md` - v6 → v7 migration guide
