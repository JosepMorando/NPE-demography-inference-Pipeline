from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np


def _loguniform(rng: np.random.Generator, a: float, b: float) -> float:
    la, lb = math.log(a), math.log(b)
    return float(math.exp(rng.uniform(la, lb)))


def _discrete_uniform(rng: np.random.Generator, a: int, b: int) -> int:
    return int(rng.integers(a, b + 1))


def _continuous_uniform(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.uniform(a, b))


def sample_times_with_constraints(times_cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, int]:
    """Sample split times with hard phylogeny constraints.

    Phylogeny structure from the SLiM template:

    p0 (ancestor)
    ├─ p1 (BG01) at T_BG01
       └─ p2 (CORE) at T_CORE
          ├─ p3 (SOUTH_LOW) at T_SOUTH_LOW
          │  └─ p5 (EAST) at T_EAST
          ├─ p4 (SOUTH_MID) at T_SOUTH_MID
          └─ p6 (INT) at T_INT
             ├─ p7 (CENTRAL) at T_CENTRAL
             └─ p8 (PYRENEES) at T_PYRENEES

    Required constraints (forward-time: parent splits BEFORE children):
    - T_BG01 < T_CORE
    - T_CORE < T_SOUTH_LOW < T_EAST
    - T_CORE < T_SOUTH_MID
    - T_CORE < T_INT < T_CENTRAL
    - T_CORE < T_INT < T_PYRENEES
    """
    keys = list(times_cfg.keys())
    for k in keys:
        if "min" not in times_cfg[k] or "max" not in times_cfg[k]:
            raise ValueError(f"times.{k} must define min and max")

    for attempt in range(50000):
        t = {k: int(rng.integers(int(times_cfg[k]["min"]), int(times_cfg[k]["max"]) + 1)) for k in keys}

        # Check all phylogenetic constraints
        if not (t["T_BG01"] < t["T_CORE"]):
            continue
        if not (t["T_CORE"] < t["T_SOUTH_LOW"] < t["T_EAST"]):
            continue
        if not (t["T_CORE"] < t["T_SOUTH_MID"]):
            continue
        if not (t["T_CORE"] < t["T_INT"] < t["T_CENTRAL"]):
            continue
        if not (t["T_CORE"] < t["T_INT"] < t["T_PYRENEES"]):
            continue

        return t

    raise RuntimeError("Failed to sample times satisfying constraints after 50,000 attempts. Widen prior bounds.")


def sample_times_individual_pops(times_cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, int]:
    """Sample split times for individual populations model.

    Phylogeny: ((P001,(BG01,((((BG05,BG04),BG07),Sauva),(Montsenymid,((Carlac,(Conangles,Viros)),(Cimadal,Coscollet)))))))

    Structure:
    p0 (ancestor)
    ├─ p1 (P001) at T_P001
    └─ p2 (Node1) at T_P001
       ├─ p3 (BG01) at T_BG01
       └─ p4 (Node2) at T_BG01
          ├─ p5 (Node3, southern) at T_MAJOR_SPLIT
          │  ├─ p11 (Sauva) at T_Sauva
          │  └─ p6 (Node4) at T_Sauva
          │     ├─ p10 (BG07) at T_BG07
          │     └─ p7 (Node5) at T_BG07
          │        ├─ p8 (BG05) at T_BG05_BG04
          │        └─ p9 (BG04) at T_BG05_BG04
          └─ p12 (Node6, northern) at T_MAJOR_SPLIT
             ├─ p13 (Montsenymid) at T_Montsenymid
             └─ p14 (Node7, Pyrenees) at T_Montsenymid
                ├─ p15 (Node8, west) at T_PYRENEES
                │  ├─ p16 (Carlac) at T_Carlac
                │  └─ p17 (Node9) at T_Carlac
                │     ├─ p18 (Conangles) at T_Conangles_Viros
                │     └─ p19 (Viros) at T_Conangles_Viros
                └─ p20 (Node10, east) at T_PYRENEES
                   ├─ p21 (Cimadal) at T_Cimadal_Coscollet
                   └─ p22 (Coscollet) at T_Cimadal_Coscollet

    Required constraints (forward-time: parent splits BEFORE children):
    - T_P001 < T_BG01
    - T_BG01 < T_MAJOR_SPLIT
    - T_MAJOR_SPLIT < T_Sauva < T_BG07 < T_BG05_BG04 (southern branch)
    - T_MAJOR_SPLIT < T_Montsenymid < T_PYRENEES (northern branch)
    - T_PYRENEES < T_Carlac < T_Conangles_Viros (western Pyrenees)
    - T_PYRENEES < T_Cimadal_Coscollet (eastern Pyrenees)
    """
    keys = list(times_cfg.keys())
    for k in keys:
        if "min" not in times_cfg[k] or "max" not in times_cfg[k]:
            raise ValueError(f"times.{k} must define min and max")

    def _bounds(name: str) -> Tuple[int, int]:
        cfg = times_cfg[name]
        return int(cfg["min"]), int(cfg["max"])

    def _pick(name: str, lo: int, hi: int) -> int:
        mn, mx = _bounds(name)
        a = max(mn, lo)
        b = min(mx, hi)
        if a > b:
            raise RuntimeError(
                f"Infeasible bounds for {name}: prior [{mn}, {mx}] and constraint-adjusted [{a}, {b}] do not overlap."
            )
        return int(rng.integers(a, b + 1))

    # Direct constrained sampling (no rejection loop).
    # Use maxima of descendant chains to ensure all strict inequalities can be satisfied.
    t_conangles_viros = _pick("T_Conangles_Viros", -10**9, 10**9)
    t_bg05_bg04 = _pick("T_BG05_BG04", -10**9, 10**9)
    t_carlac = _pick("T_Carlac", -10**9, t_conangles_viros - 1)
    t_bg07 = _pick("T_BG07", -10**9, t_bg05_bg04 - 1)

    max_major_split = min(t_bg07 - 2, t_carlac - 2)
    t_major_split = _pick("T_MAJOR_SPLIT", -10**9, max_major_split)

    t_sauva = _pick("T_Sauva", t_major_split + 1, t_bg07 - 1)
    t_pyreenees = _pick("T_PYRENEES", t_major_split + 2, t_carlac - 1)
    t_montsenymid = _pick("T_Montsenymid", t_major_split + 1, t_pyreenees - 1)
    t_cimadal_coscollet = _pick("T_Cimadal_Coscollet", t_pyreenees + 1, 10**9)
    t_bg01 = _pick("T_BG01", -10**9, t_major_split - 1)
    t_p001 = _pick("T_P001", -10**9, t_bg01 - 1)

    return {
        "T_P001": t_p001,
        "T_BG01": t_bg01,
        "T_MAJOR_SPLIT": t_major_split,
        "T_Sauva": t_sauva,
        "T_BG07": t_bg07,
        "T_BG05_BG04": t_bg05_bg04,
        "T_Montsenymid": t_montsenymid,
        "T_PYRENEES": t_pyreenees,
        "T_Carlac": t_carlac,
        "T_Conangles_Viros": t_conangles_viros,
        "T_Cimadal_Coscollet": t_cimadal_coscollet,
    }


def is_fixed(scfg: Dict[str, Any]) -> bool:
    """Check if a prior specification is fixed (not free)."""
    return str(scfg.get("dist", "")).lower() == "fixed"


def get_fixed_value(scfg: Dict[str, Any]) -> float:
    """Get the fixed value from a prior specification."""
    return float(scfg["value"])


def _is_individual_pops_model(cfg: Dict[str, Any]) -> bool:
    """Detect if using individual populations model based on config."""
    template_path = cfg.get("simulation", {}).get("slim_template", "")
    return "individual" in str(template_path).lower()


def _get_population_list(cfg: Dict[str, Any]) -> List[str]:
    """Get list of population names based on model type."""
    if _is_individual_pops_model(cfg):
        # Individual populations model
        return ["P001", "BG01", "BG05", "BG04", "BG07", "Sauva", "Montsenymid",
                "Carlac", "Conangles", "Viros", "Cimadal", "Coscollet"]
    else:
        # Grouped populations model
        return ["BG01", "SOUTH_LOW", "SOUTH_MID", "EAST", "CENTRAL", "PYRENEES"]


def sample_from_prior(cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Sample a complete parameter set from priors.

    All parameters are in BIOLOGICAL (unscaled) units.
    Scaling for SLiM is handled downstream in slim.py.

    Fixed parameters (dist: fixed) are included in the output with their
    fixed value — SLiM needs them — but they will be excluded from the
    theta vector used for NPE training (see build_theta_keys).
    """
    pri = cfg["priors"]

    # Times (biological generations) - use appropriate function based on model
    if _is_individual_pops_model(cfg):
        times = sample_times_individual_pops(pri["times"], rng)
    else:
        times = sample_times_with_constraints(pri["times"], rng)

    # Population sizes (diploid) — supports fixed and loguniform
    sizes_out: Dict[str, int] = {}
    for name, scfg in pri["sizes"].items():
        if is_fixed(scfg):
            sizes_out[name] = int(round(get_fixed_value(scfg)))
        else:
            dist = scfg.get("dist", "loguniform")
            mn = float(scfg["min"])
            mx = float(scfg["max"])
            if dist != "loguniform":
                raise ValueError(f"sizes.{name}: only loguniform and fixed supported, got '{dist}'")
            sizes_out[name] = int(round(_loguniform(rng, mn, mx)))

    extras = pri.get("demography_extras", {})
    extras_enable = bool(extras.get("enable", False))

    out: Dict[str, Any] = {}
    out.update(times)
    out.update(sizes_out)

    if extras_enable:
        # Bottleneck - support both shared and per-population modes
        bn = extras.get("bottleneck", {})
        bn_mode = str(bn.get("mode", "shared")).lower()
        
        if bn_mode == "shared":
            # Original behavior: single bottleneck parameters for all populations
            out["BN_TIME_FRAC"] = _continuous_uniform(rng, float(bn["time_fraction"]["min"]), float(bn["time_fraction"]["max"]))
            out["BN_SIZE_FRAC"] = _loguniform(rng, float(bn["size_fraction"]["min"]), float(bn["size_fraction"]["max"]))
            out["BN_DUR"] = _discrete_uniform(rng, int(bn["duration_gens"]["min"]), int(bn["duration_gens"]["max"]))
        
        elif bn_mode == "per_population":
            # New: independent bottleneck parameters for each population
            # Get populations from config or use default based on model
            pops = bn.get("populations", _get_population_list(cfg))

            for pop in pops:
                out[f"BN_TIME_FRAC_{pop}"] = _continuous_uniform(rng, float(bn["time_fraction"]["min"]), float(bn["time_fraction"]["max"]))
                out[f"BN_SIZE_FRAC_{pop}"] = _loguniform(rng, float(bn["size_fraction"]["min"]), float(bn["size_fraction"]["max"]))
                out[f"BN_DUR_{pop}"] = _discrete_uniform(rng, int(bn["duration_gens"]["min"]), int(bn["duration_gens"]["max"]))
        
        else:
            raise ValueError(f"Unknown bottleneck mode '{bn_mode}'. Use 'shared' or 'per_population'")

        # Expansion (can be disabled independently)
        ex = extras.get("expansion", {})
        if bool(ex.get("enable", True)):
            out["EXP_START_FRAC"] = _continuous_uniform(rng, float(ex["start_fraction"]["min"]), float(ex["start_fraction"]["max"]))
            out["EXP_RATE"] = _continuous_uniform(rng, float(ex["rate"]["min"]), float(ex["rate"]["max"]))
        else:
            out["EXP_START_FRAC"] = 0.0
            out["EXP_RATE"] = 0.0

        # Migration (can be disabled independently)
        mig = extras.get("migration", {})
        if bool(mig.get("enable", False)):
            out["MIG_M"] = _loguniform(rng, float(mig["m"]["min"]), float(mig["m"]["max"]))
            out["MIG_START_FRAC"] = _continuous_uniform(rng, float(mig["start_fraction"]["min"]), float(mig["start_fraction"]["max"]))
        else:
            out["MIG_M"] = 0.0
            out["MIG_START_FRAC"] = 0.0
    else:
        # Disabled: set shared bottleneck parameters to zero
        out["BN_TIME_FRAC"] = 0.0
        out["BN_SIZE_FRAC"] = 0.0
        out["BN_DUR"] = 0
        out["EXP_START_FRAC"] = 0.0
        out["EXP_RATE"] = 0.0
        out["MIG_M"] = 0.0
        out["MIG_START_FRAC"] = 0.0

    return out


def build_theta_keys(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Build the list of FREE parameter names for NPE training.

    Fixed parameters are excluded from theta — they are still passed to
    SLiM but are not inferred by the neural network.
    """
    pri = cfg["priors"]
    keys: List[str] = []

    # All times are always free
    keys += list(pri["times"].keys())

    # Only free (non-fixed) sizes
    for name, scfg in pri["sizes"].items():
        if not is_fixed(scfg):
            keys.append(name)

    # Demography extras — only if enabled
    extras = pri.get("demography_extras", {})
    if bool(extras.get("enable", False)):
        # Bottleneck - check mode
        bn = extras.get("bottleneck", {})
        bn_mode = str(bn.get("mode", "shared")).lower()
        
        if bn_mode == "shared":
            # Original: single set of bottleneck parameters
            keys += ["BN_TIME_FRAC", "BN_SIZE_FRAC", "BN_DUR"]
        elif bn_mode == "per_population":
            # New: per-population bottleneck parameters
            # Get populations from config or use default based on model
            pops = bn.get("populations", _get_population_list(cfg))
            for pop in pops:
                keys += [f"BN_TIME_FRAC_{pop}", f"BN_SIZE_FRAC_{pop}", f"BN_DUR_{pop}"]

        # Expansion only if its sub-enable is true
        ex = extras.get("expansion", {})
        if bool(ex.get("enable", True)):
            keys += ["EXP_START_FRAC", "EXP_RATE"]

        # Migration only if its sub-enable is true
        mig = extras.get("migration", {})
        if bool(mig.get("enable", False)):
            keys += ["MIG_M", "MIG_START_FRAC"]

    return tuple(keys)


def theta_vector(param_dict: Dict[str, Any], theta_keys: Tuple[str, ...]) -> np.ndarray:
    return np.array([float(param_dict[k]) for k in theta_keys], dtype=np.float32)


def build_size_anchors(cfg: Dict[str, Any], theta_keys: Tuple[str, ...]) -> Dict[str, float]:
    """Build Ne anchors for ratio parameterization in transformed space.

    For each free size parameter `N_*`, use fixed `N_CORE` as anchor when
    available; otherwise no anchor is applied for that key.
    """
    anchors: Dict[str, float] = {}
    sizes = cfg.get("priors", {}).get("sizes", {})
    anchor = None
    core_cfg = sizes.get("N_CORE") if isinstance(sizes, dict) else None
    if isinstance(core_cfg, dict) and is_fixed(core_cfg):
        anchor = float(get_fixed_value(core_cfg))

    if anchor is None:
        return anchors

    for key in theta_keys:
        if key.startswith("N_"):
            anchors[key] = anchor
    return anchors
