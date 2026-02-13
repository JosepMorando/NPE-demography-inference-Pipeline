from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tskit
import msprime


def _get_population_name(ts: tskit.TreeSequence, pop_id: int) -> str:
    md = ts.population(pop_id).metadata
    try:
        if isinstance(md, (bytes, bytearray)):
            md = json.loads(md.decode("utf-8"))
        if isinstance(md, str):
            md = json.loads(md)
        if isinstance(md, dict) and "name" in md:
            return str(md["name"])
    except Exception:
        pass
    return str(pop_id)


def get_sample_nodes_by_popname(ts: tskit.TreeSequence) -> Dict[str, np.ndarray]:
    """Return mapping pop_name -> sample node IDs."""
    out: Dict[str, List[int]] = {}
    for pop_id in range(ts.num_populations):
        name = _get_population_name(ts, pop_id)
        nodes = ts.samples(population=pop_id)
        if len(nodes) == 0:
            continue
        out[name] = np.array(nodes, dtype=np.int64)
    return {k: v for k, v in out.items()}


def overlay_mutations_if_needed(ts: tskit.TreeSequence, cfg: Dict[str, Any],
                                 rng: np.random.Generator, scale_used: int = 1) -> tskit.TreeSequence:
    """Overlay mutations post-hoc with scaled mu."""
    mcfg = cfg["simulation"].get("mutation_overlay", {})
    if not mcfg.get("enable", True):
        return ts

    mu_bio = float(mcfg.get("mu", 7.77e-9))
    mu_scaled = mu_bio * scale_used

    # Optimization: Default to binary model for speed
    mts = msprime.sim_mutations(
        ts,
        rate=mu_scaled,
        model="binary",
        random_seed=int(rng.integers(1, 2**31 - 1)),
        keep=True,
    )
    return mts


def thin_variants(ts: tskit.TreeSequence, target_snps: int, rng: np.random.Generator) -> tskit.TreeSequence:
    if target_snps is None or target_snps <= 0:
        return ts
    n = ts.num_sites
    if n <= target_snps:
        return ts
    keep = np.sort(rng.choice(np.arange(n, dtype=int), size=target_snps, replace=False))
    all_sites = np.arange(n, dtype=int)
    delete = np.setdiff1d(all_sites, keep)
    return ts.delete_sites(delete)


def _run_poolfstat_summaries(
    alt_counts: np.ndarray,
    pop_order: List[str],
    n_hap_per_pop: int,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    workdir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    target_cov = int(cfg.get("observed", {}).get("target_cov", n_hap_per_pop))
    pool_sizes = np.full(len(pop_order), max(1, n_hap_per_pop // 2), dtype=np.int32)
    seed = int(rng.integers(1, 2**31 - 1))

    # Optimization: Use standard np.savez for faster writing than compressed here
    counts_path = workdir / "sim_counts.npz"
    out_dir = workdir / "poolfstat_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        counts_path,
        alt_counts=alt_counts.astype(np.int32),
        n_hap_per_pop=np.int32(n_hap_per_pop),
        group_order=np.array(pop_order, dtype=object),
        target_cov=np.int32(target_cov),
        pool_sizes=pool_sizes,
        seed=np.int32(seed),
    )

    repo_root = Path(__file__).resolve().parents[2]
    r_script = repo_root / "r" / "compute_sim_summaries.R"
    
    # Run R script
    subprocess.run(
        ["Rscript", str(r_script), "--counts", str(counts_path), "--out", str(out_dir)],
        check=True, capture_output=True
    )

    summary_path = out_dir / "sim_summaries.npz"
    data = np.load(summary_path, allow_pickle=True)
    x = np.array(data["x"], dtype=np.float32)
    
    meta = {
        "target_cov": target_cov,
        "pool_sizes": pool_sizes.tolist(),
        "group_order": pop_order,
        "seed": seed,
    }
    return x, meta


def compute_summaries_from_trees(
    trees_path: str | Path,
    pop_order: List[str],
    n_hap_per_pop: int,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    scale_used: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute summary vector from a SLiM TreeSeq file."""
    trees_path = Path(trees_path)
    ts = tskit.load(str(trees_path))
    ts = overlay_mutations_if_needed(ts, cfg, rng, scale_used=scale_used)

    target_snps = cfg["simulation"].get("mutation_overlay", {}).get("target_snps", None)
    ts = thin_variants(ts, None if target_snps is None else int(target_snps), rng)

    # Resolve samples efficiently
    pop_samples = get_sample_nodes_by_popname(ts)
    
    sampled_nodes: List[np.ndarray] = []
    for p in pop_order:
        nodes = pop_samples[p]
        if len(nodes) < n_hap_per_pop:
            choose = rng.choice(nodes, size=n_hap_per_pop, replace=True)
        else:
            choose = rng.choice(nodes, size=n_hap_per_pop, replace=False)
        sampled_nodes.append(np.array(choose, dtype=np.int64))

    # Build genotype matrix for the union of sampled nodes
    union_nodes = np.concatenate(sampled_nodes)
    gm = ts.genotype_matrix(samples=union_nodes).astype(np.int8) # (nvar, nsamp)

    # --- OPTIMIZATION: Vectorized Allele Counting ---
    # Reshapes (SNPs, Total_Samples) -> (SNPs, Num_Pops, Samples_Per_Pop)
    # Then sums along the last axis to get allele counts pop-by-pop instantly.
    alt_counts = gm.reshape(gm.shape[0], len(pop_order), n_hap_per_pop).sum(axis=2)

    # --- OPTIMIZATION: Force temporary files to RAM disk (/dev/shm) ---
    # This bypasses the HDD and speeds up R process communication.
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
        x_pool, pf_meta = _run_poolfstat_summaries(
            alt_counts=alt_counts,
            pop_order=pop_order,
            n_hap_per_pop=n_hap_per_pop,
            cfg=cfg,
            rng=rng,
            workdir=Path(tmpdir),
        )

    # LD is disabled as requested
    x_final = x_pool

    meta = {
        "pop_order": pop_order,
        "n_hap_per_pop": n_hap_per_pop,
        "n_sites": int(ts.num_sites),
        "n_variants": int(gm.shape[0]),
        "scale_used": scale_used,
        **pf_meta,
        "ld_stats_count": 0
    }
    return x_final, meta
