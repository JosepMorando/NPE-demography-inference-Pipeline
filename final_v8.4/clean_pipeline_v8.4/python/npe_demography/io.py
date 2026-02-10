from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_groups_csv(path: str | Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Read Pop->Group mapping.

    Returns
    -------
    group_order : list
        Groups in the order they appear in the CSV.
    group_to_pops : dict
        Mapping group -> list of populations belonging to it.

    The CSV must have headers: Pop,Group.
    """
    path = Path(path)
    group_order: List[str] = []
    group_to_pops: Dict[str, List[str]] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header. Expected Pop,Group.")
        required = {"Pop", "Group"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns Pop,Group. Found {reader.fieldnames}.")

        for row in reader:
            pop = (row.get("Pop") or "").strip()
            grp = (row.get("Group") or "").strip()
            if not pop or not grp:
                continue
            if grp not in group_to_pops:
                group_to_pops[grp] = []
                group_order.append(grp)
            group_to_pops[grp].append(pop)

    if not group_order:
        raise ValueError(f"No valid Pop,Group rows found in {path}.")

    return group_order, group_to_pops
