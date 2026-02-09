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
    """Overlay mutations post-hoc with scaled mu.

    The mutation rate passed to msprime is: mu_biological * scale_factor
    This compensates for the reduced number of generations in the scaled simulation,
    preserving theta = 4 * Ne * mu (since Ne is also scaled down by the same factor).
    """
    mcfg = cfg["simulation"].get("mutation_overlay", {})
    if not mcfg.get("enable", True):
        return ts

    mu_bio = float(mcfg.get("mu", 7.77e-9))
    mu_scaled = mu_bio * scale_used  # KEY: scale mu up to compensate scaled-down time/Ne

    model = str(mcfg.get("model", "binary"))

    if model == "binary":
        mut_model = msprime.BinaryMutationModel()
    elif model == "infinite_sites":
        mut_model = msprime.InfiniteSites(msprime.NUCLEOTIDES)
    elif model == "jc69":
        mut_model = msprime.JC69()
    elif model == "hky":
        mut_model = msprime.HKY(kappa=2.0)
    else:
        raise ValueError(f"Unknown mutation model: {model}")

    mts = msprime.sim_mutations(
        ts,
        rate=mu_scaled,
        model=mut_model,
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

    counts_path = workdir / "sim_counts.npz"
    out_dir = workdir / "poolfstat_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
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
    cmd = ["Rscript", str(r_script), "--counts", str(counts_path), "--out", str(out_dir)]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "poolfstat summary computation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    summary_path = out_dir / "sim_summaries.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing poolfstat output: {summary_path}")

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
    """Compute summary vector from a SLiM TreeSeq file.

    Parameters
    ----------
    scale_used : int
        The scaling factor that was applied to SLiM. Passed to overlay_mutations
        so that mu is correctly compensated.
    """
    trees_path = Path(trees_path)
    ts = tskit.load(str(trees_path))
    ts = overlay_mutations_if_needed(ts, cfg, rng, scale_used=scale_used)

    target_snps = cfg["simulation"].get("mutation_overlay", {}).get("target_snps", None)
    ts = thin_variants(ts, None if target_snps is None else int(target_snps), rng)

    pop_samples = get_sample_nodes_by_popname(ts)

    # Resolve requested pop_order
    missing = [p for p in pop_order if p not in pop_samples]
    if missing:
        raise KeyError(f"Tree sequence lacks populations {missing}. Available: {sorted(pop_samples.keys())}")

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
    gm = ts.genotype_matrix(samples=union_nodes)  # (nvar, nsamp) haploid 0/1
    gm = np.asarray(gm, dtype=np.int8)

    k = len(pop_order)
    alt_counts = np.zeros((gm.shape[0], k), dtype=np.int32)

    idx0 = 0
    for i in range(k):
        idx1 = idx0 + n_hap_per_pop
        g = gm[:, idx0:idx1]
        alt_counts[:, i] = g.sum(axis=1)
        idx0 = idx1

    with tempfile.TemporaryDirectory(dir=str(trees_path.parent)) as tmpdir:
        x, pf_meta = _run_poolfstat_summaries(
            alt_counts=alt_counts,
            pop_order=pop_order,
            n_hap_per_pop=n_hap_per_pop,
            cfg=cfg,
            rng=rng,
            workdir=Path(tmpdir),
        )

    meta = {
        "pop_order": pop_order,
        "n_hap_per_pop": n_hap_per_pop,
        "n_sites": int(ts.num_sites),
        "n_variants": int(gm.shape[0]),
        "scale_used": scale_used,
        **pf_meta,
    }
    return x, meta
