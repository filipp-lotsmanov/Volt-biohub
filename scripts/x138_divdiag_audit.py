#!/usr/bin/env python
"""Label x138's division candidates against GT: which gate refuses the true daughters.

Reads the dumps written by the ``volt-x138-divdiag-*`` builds (x138 run on train
movies with a read-only hook in front of ``add_safe_divisions_postlink``):

- ``{ds}_graph.npz``: the graph entering the safe-division pass (node ids, t, z/y/x
  in original voxels, edges);
- ``{ds}_cands.csv``: every (source, existing child, orphan) triple inside gates
  wider than any shipped one, with the value and pass flag of each gate;
- ``{ds}_added.csv``: the forks the pass actually added.

Every GT division (a GT node with two children) is traced through that graph and
put in exactly one bucket: a node was never detected, the divider does not have
exactly one child, the second daughter already has a parent, the triple lies outside
the logging gates, a named gate refused it, or it was added. Refusals are attributed
to the first failing gate in x138's own order: existing child, parent distance,
sister distance, mutual nearest neighbour, divergence, DeepCenter veto, symmetry.
The pass flags are computed independently in the dump, so "what else fails" is also
reported, and each gate's relaxation is priced as true against false admissions.

    uv run python scripts/x138_divdiag_audit.py <divdiag dir> [<divdiag dir> ...]

Train movies are in-sample for the public checkpoints, so this measures where the
pass loses divisions it was handed, not how many a third embryo would present.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from dataspec import DATASET_PATH

SCALE = np.array([1.625, 0.40625, 0.40625])  # um per original voxel (z, y, x)
MATCH_UM = 7.0
GATES = ["exist", "par", "sis", "nn", "div", "dc", "sym"]


def gt_graph(data_dir: Path, name: str):
    g = zarr.open_group(str(data_dir / f"{name}.geff"), mode="r")
    ids = np.asarray(g["nodes/ids"])
    t = np.asarray(g["nodes/props/t/values"]).astype(int)
    zyx = np.stack([np.asarray(g[f"nodes/props/{a}/values"]) for a in "zyx"], 1).astype(float)
    edges = np.asarray(g["edges/ids"])
    return ids, t, zyx, edges


def match_frame(gt_um: np.ndarray, pr_um: np.ndarray) -> dict[int, int]:
    """Optimal one-to-one assignment within MATCH_UM, as the scorer does."""
    if not len(gt_um) or not len(pr_um):
        return {}
    d = np.linalg.norm(gt_um[:, None, :] - pr_um[None, :, :], axis=2)
    cost = np.where(d <= MATCH_UM, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(r, c, strict=True) if d[i, j] <= MATCH_UM}


def audit_movie(ddir: Path, name: str, data_dir: Path, cand_rows: list, buckets: Counter):
    graph = np.load(ddir / f"{name}_graph.npz")
    p_ids, p_t, p_zyx = graph["node_id"], graph["t"], graph["zyx"]
    children: dict[int, list[int]] = {}
    parent: dict[int, int] = {}
    for s, d in zip(graph["edge_src"], graph["edge_tgt"], strict=True):
        children.setdefault(int(s), []).append(int(d))
        parent[int(d)] = int(s)
    added = set()
    added_path = ddir / f"{name}_added.csv"
    if added_path.exists():
        with open(added_path) as fh:
            added = {(int(r["source_id"]), int(r["target_id"])) for r in csv.DictReader(fh)}
    cands: dict[tuple[int, int, int], dict] = {}
    with open(ddir / f"{name}_cands.csv") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["source_id"]), int(r["child_id"]), int(r["cand_id"]))
            cands[key] = r

    g_ids, g_t, g_zyx, g_edges = gt_graph(data_dir, name)
    g_index = {int(i): k for k, i in enumerate(g_ids)}
    g_children: dict[int, list[int]] = {}
    for s, d in g_edges:
        g_children.setdefault(int(s), []).append(int(d))
    divisions = [(s, kids) for s, kids in g_children.items() if len(kids) == 2]
    if not divisions:
        return
    # GT -> prediction matches, only for the frames that hold a division.
    frames = {int(g_t[g_index[s]]) for s, _ in divisions}
    frames |= {f + 1 for f in frames}
    match: dict[int, int] = {}
    for f in frames:
        gi = np.nonzero(g_t == f)[0]
        pi = np.nonzero(p_t == f)[0]
        m = match_frame(g_zyx[gi] * SCALE, p_zyx[pi] * SCALE)
        match.update({int(g_ids[gi[a]]): int(p_ids[pi[b]]) for a, b in m.items()})

    true_keys = set()
    for s, (d1, d2) in divisions:
        ps, pa, pb = match.get(s), match.get(d1), match.get(d2)
        if ps is None or pa is None or pb is None:
            buckets["node never detected"] += 1
            continue
        kids = children.get(ps, [])
        if pa in kids and pb in kids:
            buckets["fork already in graph (ILP)"] += 1
            continue
        if len(kids) != 1 or kids[0] not in (pa, pb):
            buckets["divider not linked to exactly one daughter"] += 1
            continue
        exist, other = (pa, pb) if kids[0] == pa else (pb, pa)
        if other in parent:
            buckets["second daughter parented elsewhere"] += 1
            continue
        if (ps, other) in added:
            buckets["ADDED (true fork)"] += 1
            true_keys.add((ps, exist, other))
            continue
        row = cands.get((ps, exist, other))
        if row is None:
            buckets["outside the logging gates"] += 1
            continue
        true_keys.add((ps, exist, other))
        failing = [g for g in GATES if row[f"pass_{g}"] == "0"]
        buckets[f"refused by {failing[0]}" if failing else "passes every gate, not added (cap)"] += 1
    for key, row in cands.items():
        cand_rows.append({**row, "movie": name, "true": int(key in true_keys),
                          "added": int((key[0], key[2]) in added)})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", type=Path, nargs="+")
    p.add_argument("--data-dir", type=Path, default=Path(DATASET_PATH))
    p.add_argument("--out", type=Path, default=None, help="write the labelled candidates here")
    a = p.parse_args(argv)
    buckets: Counter = Counter()
    rows: list[dict] = []
    movies = 0
    for ddir in a.dirs:
        for f in sorted(ddir.glob("*_cands.csv")):
            name = f.name[: -len("_cands.csv")]
            if not (a.data_dir / f"{name}.geff").exists():
                continue
            audit_movie(ddir, name, a.data_dir, rows, buckets)
            movies += 1
    total = sum(buckets.values())
    print(f"{movies} movies, {total} GT divisions")
    for k, v in buckets.most_common():
        print(f"  {k:45s} {v:5d}  {v / max(total, 1):6.1%}")
    true = [r for r in rows if r["true"]]
    false = [r for r in rows if not r["true"]]
    print(f"\ncandidates logged: {len(rows)} ({len(true)} true, {len(false)} false)")
    # Price each gate: how many true / false candidates it alone refuses, i.e. would
    # be admitted if only that gate were opened.
    for g in GATES:
        others = [x for x in GATES if x != g]
        t_n = sum(r[f"pass_{g}"] == "0" and all(r[f"pass_{o}"] == "1" for o in others) for r in true)
        f_n = sum(r[f"pass_{g}"] == "0" and all(r[f"pass_{o}"] == "1" for o in others) for r in false)
        print(f"  opening only '{g}' admits {t_n} true and {f_n} false")
    for col in ("d_par", "d_sis", "symmetry", "div_margin", "dc_score"):
        tv = np.array([float(r[col]) for r in true if r[col] not in ("", "nan")])
        if len(tv):
            print(f"  true daughters {col:10s} p10/p50/p90 "
                  f"{np.percentile(tv, 10):.2f}/{np.percentile(tv, 50):.2f}/{np.percentile(tv, 90):.2f}")
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
