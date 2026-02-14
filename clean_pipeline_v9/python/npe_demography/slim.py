from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


# Mapping from semantic group name to SLiM subpop name in template (GROUPED MODEL)
DEFAULT_GROUP_TO_SUBPOP = {
    "ANCESTRAL": "p1",     # BG01
    "SOUTH_LOW": "p3",     # Sauva
    "SOUTH_MID": "p4",     # Montsenymid
    "EAST": "p5",          # BG04/BG05/BG07
    "CENTRAL": "p7",       # Coscollet/Cimadal
    "PYRENEES": "p8",      # Carlac/Conangles/Viros
}

# Split-time parameter associated with each sampled subpop (GROUPED MODEL)
DEFAULT_SUBPOP_TO_TSPLIT = {
    "p1": "T_BG01",
    "p3": "T_SOUTH_LOW",
    "p4": "T_SOUTH_MID",
    "p5": "T_EAST",
    "p7": "T_CENTRAL",
    "p8": "T_PYRENEES",
}

# Baseline size parameter for each sampled subpop (GROUPED MODEL)
DEFAULT_SUBPOP_TO_N = {
    "p1": "N_BG01",
    "p3": "N_SOUTH_LOW",
    "p4": "N_SOUTH_MID",
    "p5": "N_EAST",
    "p7": "N_CENTRAL",
    "p8": "N_PYRENEES",
}

# All population size keys (including ghost/unsampled) - GROUPED MODEL
ALL_POP_KEYS = {"N0", "N_BG01", "N_CORE", "N_SOUTH_LOW", "N_SOUTH_MID",
                "N_EAST", "N_INT", "N_CENTRAL", "N_PYRENEES"}

# All time keys - GROUPED MODEL
ALL_TIME_KEYS = {"T_BG01", "T_CORE", "T_SOUTH_LOW", "T_SOUTH_MID",
                 "T_EAST", "T_INT", "T_CENTRAL", "T_PYRENEES"}

# ========== INDIVIDUAL POPULATIONS MODEL ==========
# Phylogeny: ((P001,(BG01,((((BG05,BG04),BG07),Sauva),(Montsenymid,((Carlac,(Conangles,Viros)),(Cimadal,Coscollet)))))))

# Mapping from individual population to SLiM subpop name (INDIVIDUAL MODEL)
INDIVIDUAL_GROUP_TO_SUBPOP = {
    "P001": "p1",
    "BG01": "p3",
    "BG05": "p8",
    "BG04": "p9",
    "BG07": "p10",
    "Sauva": "p11",
    "Montsenymid": "p13",
    "Carlac": "p16",
    "Conangles": "p18",
    "Viros": "p19",
    "Cimadal": "p21",
    "Coscollet": "p22",
}

# Split-time parameter for each sampled subpop (INDIVIDUAL MODEL)
INDIVIDUAL_SUBPOP_TO_TSPLIT = {
    "p1": "T_P001",
    "p3": "T_BG01",
    "p8": "T_BG05_BG04",
    "p9": "T_BG05_BG04",
    "p10": "T_BG07",
    "p11": "T_Sauva",
    "p13": "T_Montsenymid",
    "p16": "T_Carlac",
    "p18": "T_Conangles_Viros",
    "p19": "T_Conangles_Viros",
    "p21": "T_Cimadal_Coscollet",
    "p22": "T_Cimadal_Coscollet",
}

# Baseline size parameter for each sampled subpop (INDIVIDUAL MODEL)
INDIVIDUAL_SUBPOP_TO_N = {
    "p1": "N_P001",
    "p3": "N_BG01",
    "p8": "N_BG05",
    "p9": "N_BG04",
    "p10": "N_BG07",
    "p11": "N_Sauva",
    "p13": "N_Montsenymid",
    "p16": "N_Carlac",
    "p18": "N_Conangles",
    "p19": "N_Viros",
    "p21": "N_Cimadal",
    "p22": "N_Coscollet",
}

# All population size keys (including ghost/unsampled) - INDIVIDUAL MODEL
INDIVIDUAL_ALL_POP_KEYS = {
    "N0", "N_Node1", "N_Node2", "N_Node3", "N_Node4", "N_Node5",
    "N_Node6", "N_Node7", "N_Node8", "N_Node9", "N_Node10",
    "N_P001", "N_BG01", "N_BG05", "N_BG04", "N_BG07", "N_Sauva",
    "N_Montsenymid", "N_Carlac", "N_Conangles", "N_Viros", "N_Cimadal", "N_Coscollet"
}

# All time keys - INDIVIDUAL MODEL
INDIVIDUAL_ALL_TIME_KEYS = {
    "T_P001", "T_BG01", "T_MAJOR_SPLIT", "T_Sauva", "T_BG07", "T_BG05_BG04",
    "T_Montsenymid", "T_PYRENEES", "T_Carlac", "T_Conangles_Viros", "T_Cimadal_Coscollet"
}


def _get_model_mappings(cfg: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, set, set]:
    """Detect which model is being used and return appropriate mappings.

    Returns: (GROUP_TO_SUBPOP, SUBPOP_TO_TSPLIT, SUBPOP_TO_N, ALL_POP_KEYS, ALL_TIME_KEYS)
    """
    template_path = cfg.get("simulation", {}).get("slim_template", "")

    # Detect model type from template filename
    if "individual" in str(template_path).lower():
        return (INDIVIDUAL_GROUP_TO_SUBPOP, INDIVIDUAL_SUBPOP_TO_TSPLIT,
                INDIVIDUAL_SUBPOP_TO_N, INDIVIDUAL_ALL_POP_KEYS, INDIVIDUAL_ALL_TIME_KEYS)
    else:
        return (DEFAULT_GROUP_TO_SUBPOP, DEFAULT_SUBPOP_TO_TSPLIT,
                DEFAULT_SUBPOP_TO_N, ALL_POP_KEYS, ALL_TIME_KEYS)


def compute_scale_factor(params: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    """Compute the adaptive scale factor, maximised while keeping all pops >= safe_min.

    Scaling approach (preserves theta = 4*Ne*mu and rho = 4*Ne*r):
      Ne_slim = Ne / scale;  T_slim = T / scale;  R_slim = R * scale;  mu_overlay = mu * scale

    The scale factor is set to the MAXIMUM value such that:
      min(all_Ne) / scale >= safe_min_diploids

    Config keys:
      simulation.max_scale_factor:  upper bound on scaling (default 200)
      simulation.safe_min_diploids: minimum diploid Ne in SLiM (default 50)
      simulation.scale_factor:      legacy key, treated as max_scale_factor if present
    """
    # Support both new (max_scale_factor) and legacy (scale_factor) config keys
    max_scale = int(cfg["simulation"].get("max_scale_factor",
                        cfg["simulation"].get("scale_factor", 200)))
    safe_min = int(cfg["simulation"].get("safe_min_diploids", 50))

    if max_scale <= 1:
        return 1

    # Get model-specific population keys
    _, _, _, all_pop_keys, _ = _get_model_mappings(cfg)

    # Find the smallest population size among all pop keys present
    min_pop = min(int(params[k]) for k in all_pop_keys if k in params)

    # Adaptive: use the largest scale that keeps min_pop / scale >= safe_min
    max_safe_scale = max(1, min_pop // safe_min)
    return min(max_scale, max_safe_scale)


def scale_params_for_slim(params: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Scale biological parameters down for SLiM execution.

    Returns (slim_params_dict, scale_used).

    Scaling rules (matching manuscript):
      - Population sizes:  N_slim = max(safe_min, round(N / scale))
      - Split/event times: T_slim = max(1, round(T / scale))
      - Recombination rate: R_slim = R * scale  (compensates fewer gens)
      - GENS, BURNIN:      scaled down by scale
      - Mutation rate:      NOT passed to SLiM (overlay post-hoc with mu * scale)
    """
    scale = compute_scale_factor(params, cfg)
    safe_min = int(cfg["simulation"].get("safe_min_diploids", 50))

    # Get model-specific keys
    _, _, _, all_pop_keys, all_time_keys = _get_model_mappings(cfg)

    slim = {}

    for k, v in params.items():
        if k.startswith("_"):
            continue
        if k in all_pop_keys:
            slim[k] = max(safe_min, int(round(int(v) / scale)))
        elif k in all_time_keys:
            slim[k] = max(1, int(round(int(v) / scale)))
        elif k == "BN_DUR" or k.startswith("BN_DUR_"):
            # Bottleneck duration in generations: scale down (both shared and per-pop)
            slim[k] = max(1, int(round(int(v) / scale)))
        elif k in {"BN_TIME_FRAC", "BN_SIZE_FRAC", "EXP_START_FRAC", "EXP_RATE",
                    "MIG_M", "MIG_START_FRAC"} or k.startswith("BN_TIME_FRAC_") or k.startswith("BN_SIZE_FRAC_"):
            # Fractions and rates: pass through (they're relative or already adjusted)
            slim[k] = v
        else:
            slim[k] = v

    return slim, scale


def _inject_blocks(template_text: str, bottleneck_block: str, expansion_block: str, migration_block: str) -> str:
    out = template_text.replace("//__BOTTLENECK_BLOCK__", bottleneck_block)
    out = out.replace("//__EXPANSION_BLOCK__", expansion_block)
    out = out.replace("//__MIGRATION_BLOCK__", migration_block)
    return out


def _build_bottleneck_block(cfg: Dict[str, Any], slim_params: Dict[str, Any], bio_params: Dict[str, Any]) -> str:
    """Build bottleneck SLiM code using SCALED parameters for SLiM execution.

    Supports two modes:
    - shared: Single bottleneck parameters applied to all populations
    - per_population: Independent bottleneck parameters for each population
    """
    pri = cfg["priors"].get("demography_extras", {})
    if not pri.get("enable", False):
        return "// (bottlenecks disabled)"

    bn = pri.get("bottleneck", {})
    bn_mode = str(bn.get("mode", "shared")).lower()
    gens = int(slim_params.get("GENS", cfg["simulation"]["gens"]))

    # Get model-specific mappings
    _, subpop_to_tsplit, subpop_to_n, _, _ = _get_model_mappings(cfg)

    lines: List[str] = []

    # Get list of populations that can have bottlenecks
    if bn_mode == "per_population":
        # Use populations specified in config, or default to all sampled populations
        pops = bn.get("populations", list(subpop_to_n.values()))
        # Convert N_* to pop names by stripping N_ prefix
        pop_names = [p.replace("N_", "") if p.startswith("N_") else p for p in pops]
    else:
        pop_names = []

    for subpop, tkey in subpop_to_tsplit.items():
        nkey = subpop_to_n[subpop]
        if tkey not in slim_params or nkey not in slim_params:
            continue
        t_split = int(slim_params[tkey])
        n0 = int(slim_params[nkey])

        # Get bottleneck parameters for this population
        if bn_mode == "shared":
            # Shared parameters
            time_frac = float(bio_params.get("BN_TIME_FRAC", 0.0))
            size_frac = float(bio_params.get("BN_SIZE_FRAC", 0.0))
            dur = int(slim_params.get("BN_DUR", 0))
        elif bn_mode == "per_population":
            # Per-population parameters - extract population name from N key
            pop_name = nkey.replace("N_", "")
            time_frac = float(bio_params.get(f"BN_TIME_FRAC_{pop_name}", 0.0))
            size_frac = float(bio_params.get(f"BN_SIZE_FRAC_{pop_name}", 0.0))
            # Duration needs to be from slim_params (scaled)
            dur_key = f"BN_DUR_{pop_name}"
            dur = int(slim_params.get(dur_key, 0))
        else:
            raise ValueError(f"Unknown bottleneck mode '{bn_mode}'")

        if dur <= 0 or size_frac <= 0.0:
            continue

        # terminal branch length in scaled ticks
        br_len = max(1, gens - t_split)
        max_start = gens - dur - 1
        if max_start <= t_split + 1:
            continue
        t_start = int(round(t_split + time_frac * (max_start - t_split)))
        t_end = t_start + dur
        n_bn = max(2, int(round(n0 * size_frac)))

        lines.append(f"// Bottleneck in {subpop} (N={n_bn} for {dur} scaled gens)")
        lines.append(f"(BURNIN_slim + {t_start}) early() {{ {subpop}.setSubpopulationSize({n_bn}); }}")
        lines.append(f"(BURNIN_slim + {t_end}) early() {{ {subpop}.setSubpopulationSize({n0}); }}")

    return "\n".join(lines) if lines else "// (bottleneck skipped: incompatible times)"


def _build_expansion_block(cfg: Dict[str, Any], slim_params: Dict[str, Any], bio_params: Dict[str, Any]) -> str:
    """Build expansion SLiM code using SCALED parameters."""
    pri = cfg["priors"].get("demography_extras", {})
    if not pri.get("enable", False):
        return "// (expansion disabled)"

    # Check sub-enable flag for expansion
    ex_cfg = pri.get("expansion", {})
    if not ex_cfg.get("enable", True):
        return "// (expansion disabled in config)"

    gens = int(slim_params.get("GENS", cfg["simulation"]["gens"]))

    start_frac = float(bio_params.get("EXP_START_FRAC", 0.0))
    rate = float(bio_params.get("EXP_RATE", 0.0))
    if rate == 0.0:
        return "// (no expansion drawn)"

    # Get model-specific mappings
    _, subpop_to_tsplit, subpop_to_n, _, _ = _get_model_mappings(cfg)

    lines: List[str] = []
    for subpop, tkey in subpop_to_tsplit.items():
        nkey = subpop_to_n[subpop]
        if tkey not in slim_params or nkey not in slim_params:
            continue
        t_split = int(slim_params[tkey])
        n0 = int(slim_params[nkey])
        br_len = max(1, gens - t_split)
        t_start = int(round(t_split + start_frac * br_len))
        if t_start >= gens:
            continue

        # Cap rate to prevent overflow in scaled ticks
        max_duration = gens - t_start
        max_safe_rate = 13.8 / max(1, max_duration)
        capped_rate = max(-max_safe_rate, min(max_safe_rate, rate))

        # SLiM5: use community.tick for current tick
        lines.append(f"// Exponential size change in {subpop}")
        lines.append(f"(BURNIN_slim + {t_start}):(BURNIN_slim + GENS_slim) early() {{")
        lines.append(f"    dt = community.tick - (BURNIN_slim + {t_start});")
        lines.append(f"    newN = asInteger(round({n0} * exp({capped_rate} * dt)));")
        lines.append(f"    if (newN < 2) newN = 2;")
        lines.append(f"    if (newN > 1000000) newN = 1000000;")
        lines.append(f"    {subpop}.setSubpopulationSize(newN);")
        lines.append(f"}}")

    return "\n".join(lines) if lines else "// (expansion skipped)"


def _build_migration_block(cfg: Dict[str, Any], slim_params: Dict[str, Any], bio_params: Dict[str, Any]) -> str:
    """Build migration SLiM code using SCALED parameters."""
    pri = cfg["priors"].get("demography_extras", {})
    if not pri.get("enable", False):
        return "// (migration disabled)"

    mig = pri.get("migration", {})
    if not mig.get("enable", False):
        return "// (migration disabled)"

    m = float(bio_params.get("MIG_M", 0.0))
    start_frac = float(bio_params.get("MIG_START_FRAC", 0.0))
    if m <= 0.0:
        return "// (no migration drawn)"

    gens = int(slim_params.get("GENS", cfg["simulation"]["gens"]))

    # Get model-specific mappings
    group_to_subpop, subpop_to_tsplit, _, _, _ = _get_model_mappings(cfg)

    def grp_to_subpop(grp: str) -> str:
        if grp in group_to_subpop:
            return group_to_subpop[grp]
        if grp.startswith("p"):
            return grp
        raise KeyError(f"Unknown group '{grp}'. Add it to group_to_subpop mapping or use SLiM ids directly.")

    lines: List[str] = []
    pairs = mig.get("pairs", [])
    for pair in pairs:
        a, b = pair
        pa = grp_to_subpop(str(a))
        pb = grp_to_subpop(str(b))

        ts_a = int(slim_params[subpop_to_tsplit[pa]])
        ts_b = int(slim_params[subpop_to_tsplit[pb]])
        t0 = max(ts_a, ts_b)
        br_len = max(1, gens - t0)
        t_start = int(round(t0 + start_frac * br_len))
        if t_start >= gens:
            continue

        lines.append(f"// Symmetric migration between {pa} and {pb} (m={m})")
        lines.append(f"(BURNIN_slim + {t_start}) early() {{")
        lines.append(f"    {pa}.setMigrationRates({pb}, {m});")
        lines.append(f"    {pb}.setMigrationRates({pa}, {m});")
        lines.append("}")

    return "\n".join(lines) if lines else "// (migration skipped)"


def render_slim_script(template_path: str | Path, out_path: str | Path,
                       cfg: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Path, int]:
    """Render SLiM script with scaling and optional demography extras.

    Returns (script_path, scale_used).
    """
    template_path = Path(template_path)
    out_path = Path(out_path)

    # Scale parameters for SLiM
    slim_params, scale_used = scale_params_for_slim(params, cfg)

    # Ensure simulation horizon is available in *scaled ticks* for injected demography blocks
    simcfg = cfg['simulation']
    slim_params['GENS'] = max(1, int(round(int(simcfg['gens']) / scale_used)))
    slim_params['BURNIN'] = max(1, int(round(int(simcfg['burnin']) / scale_used)))

    txt = template_path.read_text(encoding="utf-8")

    # Build optional blocks using scaled SLiM params but biological fractions
    bottleneck_block = _build_bottleneck_block(cfg, slim_params, params)
    expansion_block = _build_expansion_block(cfg, slim_params, params)
    migration_block = _build_migration_block(cfg, slim_params, params)

    txt = _inject_blocks(txt, bottleneck_block, expansion_block, migration_block)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt, encoding="utf-8")
    return out_path, scale_used


def run_slim(
    slim_binary: str,
    slim_script: str | Path,
    tree_out: str | Path,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    timeout_s: float | None = None,
) -> int:
    """Run SLiM with -d parameter definitions. Returns scale_used.

    Applies the scaling transformation before passing parameters to SLiM:
      - Ne / scale, T / scale, R * scale, GENS / scale, BURNIN / scale
    """
    slim_script = str(slim_script)
    tree_out = str(tree_out)

    simcfg = cfg["simulation"]
    scale = compute_scale_factor(params, cfg)
    safe_min = int(simcfg.get("safe_min_diploids", 50))

    # Scale simulation-level params
    gens_scaled = max(1, int(round(int(simcfg["gens"]) / scale)))
    burnin_scaled = max(1, int(round(int(simcfg["burnin"]) / scale)))
    r_scaled = float(simcfg["recombination_rate"]) * scale

    cmd: List[str] = [
        slim_binary,
        "-d", f"TREE_OUT='{tree_out}'",
        "-d", f"R={r_scaled}",
        "-d", f"GENS={gens_scaled}",
        "-d", f"BURNIN={burnin_scaled}",
        "-d", f"LEN={int(float(simcfg['genome_length']))}",
    ]

    # Scale and pass demographic parameters
    slim_params, _ = scale_params_for_slim(params, cfg)
    for k, v in slim_params.items():
        if k.startswith("_"):
            continue
        # Skip keys already passed or not needed by SLiM template
        if k in {"R", "GENS", "BURNIN", "LEN", "TREE_OUT",
                  "BN_TIME_FRAC", "BN_SIZE_FRAC", "BN_DUR",
                  "EXP_START_FRAC", "EXP_RATE",
                  "MIG_M", "MIG_START_FRAC"}:
            continue
        if isinstance(v, (int, np.integer)):
            cmd += ["-d", f"{k}={int(v)}"]
        elif isinstance(v, (float, np.floating)):
            cmd += ["-d", f"{k}={float(v)}"]
        else:
            cmd += ["-d", f"{k}={v}"]

    cmd.append(slim_script)

    p_run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if p_run.returncode != 0:
        cmd_str = " ".join(cmd)
        raise RuntimeError(
            "SLiM simulation failed.\n"
            f"CMD: {cmd_str}\n"
            f"STDOUT:\n{p_run.stdout}\n"
            f"STDERR:\n{p_run.stderr}\n"
        )

    return scale
