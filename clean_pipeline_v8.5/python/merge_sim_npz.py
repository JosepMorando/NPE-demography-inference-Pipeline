#!/usr/bin/env python3
"""Merge multiple simulation .npz files produced by python/simulate.py.

This is a thin utility used by the multi-node runner scripts.
It concatenates X and Theta across parts and preserves theta_keys/pop_order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def _load_meta(meta_json: object) -> List[object]:
    if meta_json is None:
        return []
    # meta_json is saved as JSON string
    return json.loads(str(meta_json))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output merged .npz path")
    p.add_argument("parts", nargs="+", help="Input part .npz files")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Xs = []
    Thetas = []
    metas_all: List[object] = []

    theta_keys_ref = None
    pop_order_ref = None

    for f in args.parts:
        z = np.load(f, allow_pickle=True)

        X = z["X"]
        Theta = z["Theta"]
        theta_keys = z["theta_keys"]
        pop_order = z["pop_order"]
        meta_json = z.get("meta_json", None)

        if theta_keys_ref is None:
            theta_keys_ref = theta_keys
            pop_order_ref = pop_order
        else:
            if not np.array_equal(theta_keys_ref, theta_keys):
                raise ValueError(f"theta_keys mismatch in {f}")
            if not np.array_equal(pop_order_ref, pop_order):
                raise ValueError(f"pop_order mismatch in {f}")

        Xs.append(X)
        Thetas.append(Theta)
        metas_all.extend(_load_meta(meta_json))

    X_merged = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
    Theta_merged = np.concatenate(Thetas, axis=0).astype(np.float32, copy=False)

    np.savez_compressed(
        out_path,
        X=X_merged,
        Theta=Theta_merged,
        theta_keys=np.array(theta_keys_ref, dtype=object),
        pop_order=np.array(pop_order_ref, dtype=object),
        meta_json=json.dumps(metas_all),
    )

    print(f"[merge_sim_npz] Wrote {out_path}  (X: {X_merged.shape}, Theta: {Theta_merged.shape}, meta: {len(metas_all)})")


if __name__ == "__main__":
    main()
