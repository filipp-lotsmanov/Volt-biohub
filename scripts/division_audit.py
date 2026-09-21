#!/usr/bin/env python
"""Why do we miss 73 % of real divisions? Decompose the FNs, don't guess.

The division term is worth 0.1 of the score and we hold ~0.07 of it, so ~0.09
is unclaimed -- more than every remaining port item combined. The 2026-09-18
decomposition (4,053 forks emitted, 40 charged FP, TP 7, FN 19) said *how many*
we miss but not *why*, and the two answers imply opposite fixes:

* the daughters were never detected  -> detection-limited, opening the division
  gates cannot help and will only add false forks;
* the daughters were detected but no fork was emitted -> admission-limited, and
  the gates, the orphan requirement and the caps are the things to move.

That distinction has never been measured, and every division experiment since
has been a threshold sweep in the dark. This labels each GT division with the
official scorer and then, for the ones scored FN, reports which of the three
nodes (divider, both daughters) a prediction actually matched.

The match test here is nearest-neighbour within the 7 um tolerance, *not* the
scorer's optimal bipartite assignment -- so it is an upper bound on what was
detected. A node this calls missing is definitely missing; one it calls found
may still have lost the assignment to a competitor. That asymmetry is the right
way round for the question being asked.

Usage
-----
    uv run python scripts/division_audit.py --csv submission.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle"))

from scipy.spatial import cKDTree

from csv_to_geffs import build_graph_from_rows
from dataspec import DATASET_PATH
from volt_biohub.division_metrics import score_divisions
from volt_biohub.scoring import ORIGINAL_SCALE, load_gt


def _nodes_um(graph) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(t, xyz_um, ids) for every node in *graph*.

    ``node_id`` must be named in *attr_keys* -- ``node_attrs`` omits it
    otherwise, and GT ids are sparse (68000000011), so falling back to a
    positional range would silently mismatch every division.
    """
    df = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    ids = df["node_id"].to_list()
    t = df["t"].to_numpy().astype(np.int64)
    zyx = np.stack([df["z"].to_numpy(), df["y"].to_numpy(), df["x"].to_numpy()], axis=1)
    return t, zyx.astype(np.float64) * np.asarray(ORIGINAL_SCALE), ids


def _match(pred_t, pred_um, trees, tree_rows, t, um, tol=7.0):
    """Row index of the nearest predicted node within *tol* um, or None."""
    tree = trees.get(int(t))
    if tree is None:
        return None
    d, k = tree.query(um, k=1)
    return tree_rows[int(t)][int(k)] if d <= tol else None


def audit_one(name: str, nodes: pl.DataFrame, edges: pl.DataFrame,
              data_dir: Path) -> list[dict]:
    gt = load_gt(data_dir, name)
    # score_divisions mutates its inputs, so the graph it grades and the graph
    # the detection test reads must be different objects.
    scored = score_divisions(build_graph_from_rows(nodes, edges),
                             gt, ORIGINAL_SCALE, 7.0)
    fresh = build_graph_from_rows(nodes, edges)
    pred_t, pred_um, _ids = _nodes_um(fresh)
    trees, tree_rows = {}, {}
    for t in np.unique(pred_t):
        sel = np.nonzero(pred_t == t)[0]
        trees[int(t)] = cKDTree(pred_um[sel])
        tree_rows[int(t)] = sel

    gt_t, gt_um, gt_ids = _nodes_um(gt)
    pos = {i: (gt_t[k], gt_um[k]) for k, i in enumerate(gt_ids)}

    # Predicted degrees, keyed by the node's index in pred_um, so a matched GT
    # node can be asked whether it was already parented.
    indeg: dict[int, int] = {}
    outdeg: dict[int, int] = {}
    parent_of: dict[int, int] = {}
    id_to_row = {i: k for k, i in enumerate(_ids)}
    for a, b in fresh.edge_attrs(attr_keys=[]).select(
            ["source_id", "target_id"]).iter_rows():
        ra, rb = id_to_row.get(a), id_to_row.get(b)
        if ra is None or rb is None:
            continue
        outdeg[ra] = outdeg.get(ra, 0) + 1
        indeg[rb] = indeg.get(rb, 0) + 1
        parent_of[rb] = ra

    rows = []
    for div_node, ok in scored.scores.items():
        kids = list(gt.successors(div_node))
        parts = {"divider": div_node, **{f"daughter{k}": c
                                         for k, c in enumerate(kids, 1)}}
        matched = {k: _match(pred_t, pred_um, trees, tree_rows, *pos[v])
                   for k, v in parts.items() if v in pos}
        det = {k: v is not None for k, v in matched.items()}
        # Which structural gate could have rejected this fork. The orphan rule
        # in add_safe_divisions requires the second daughter to have NO parent
        # at all, so a daughter the edge head already linked to something else
        # can never become a candidate however wide the distance gates are.
        d_rows = [matched[k] for k in matched if k.startswith("daughter")]
        div_row = matched.get("divider")
        parented = sum(1 for r in d_rows if r is not None and indeg.get(r, 0) > 0)
        adopted = sum(1 for r in d_rows
                      if r is not None and parent_of.get(r) == div_row)
        # Is the true sister actually the candidate add_safe_divisions would
        # pick? The rule is `tree.query(pos[ch])` -- the SINGLE nearest orphan
        # to the existing child -- so a spurious detection nearer than the true
        # sister makes her unreachable by any distance gate. This is the column
        # that separates "a gate rejected it" from "the selection never
        # proposed it", and the divergence sweep (-2.0/0.0/1.0 all worse than
        # 2.25, divJ falling monotonically as the gate opens) says it is the
        # latter.
        sister_rank = -1
        linked = [r for r in d_rows if r is not None and parent_of.get(r) == div_row]
        free = [r for r in d_rows if r is not None and indeg.get(r, 0) == 0]
        if linked and free:
            ch, sister = linked[0], free[0]
            t_sis = int(pred_t[sister])
            orphan_rows = [r for r in tree_rows.get(t_sis, [])
                           if indeg.get(int(r), 0) == 0]
            if orphan_rows and sister in orphan_rows:
                d = np.linalg.norm(pred_um[orphan_rows] - pred_um[ch], axis=1)
                order = [orphan_rows[k] for k in np.argsort(d)]
                sister_rank = order.index(sister)

        rows.append({
            "dataset": name, "gt_div_node": int(div_node),
            "scored_tp": bool(ok), "n_daughters": len(kids),
            "detected": sum(det.values()), "of": len(det),
            "missing": ",".join(sorted(k for k, v in det.items() if not v)) or "-",
            "divider_outdeg": outdeg.get(div_row, 0) if div_row is not None else -1,
            "daughters_parented": parented,
            "daughters_linked_to_divider": adopted,
            "sister_orphan_rank": sister_rank,
        })
    return rows, len(scored.fp_forks)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("reports/division_audit.csv"))
    args = p.parse_args(argv)
    data_dir = args.data_dir or DATASET_PATH

    df = pl.read_csv(args.csv, columns=[
        "dataset", "row_type", "node_id", "t", "z", "y", "x",
        "source_id", "target_id"])
    all_rows, fp_total = [], 0
    for (name,), group in df.group_by("dataset"):
        if not (data_dir / f"{name}.geff").exists():
            continue
        rows, n_fp = audit_one(
            name,
            group.filter(pl.col("row_type") == "node"),
            group.filter(pl.col("row_type") == "edge"),
            data_dir,
        )
        all_rows.extend(rows)
        fp_total += n_fp
        print(f"  {name}: {sum(r['scored_tp'] for r in rows)}/{len(rows)} TP, "
              f"{n_fp} FP forks", flush=True)

    if not all_rows:
        print("no GT divisions in these datasets")
        return 1
    out = pl.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)

    tp = int(out["scored_tp"].sum())
    fn_rows = out.filter(~pl.col("scored_tp"))
    fn = fn_rows.height
    print(f"\nGT divisions {out.height}   TP {tp}   FN {fn}   FP forks {fp_total}")
    print(f"division Jaccard {tp / max(tp + fn + fp_total, 1):.4f}")
    print("\nFN decomposition -- how many of the three nodes were detected:")
    for k, v in sorted(Counter(
            f"{r['detected']}/{r['of']}" for r in fn_rows.iter_rows(named=True)
    ).items(), reverse=True):
        print(f"  {k} detected : {v}")
    full = fn_rows.filter(pl.col("detected") == pl.col("of"))
    print(f"\nof the {full.height} FNs with EVERY node detected -- what blocked the fork:")
    print(f"  divider out-degree 0 : {full.filter(pl.col('divider_outdeg') == 0).height}")
    print(f"  divider out-degree 1 : {full.filter(pl.col('divider_outdeg') == 1).height}"
          "   <- add_safe_divisions' candidate pool")
    print(f"  divider out-degree 2 : {full.filter(pl.col('divider_outdeg') >= 2).height}"
          "   <- already forked, so the fork exists but did not score")
    # The distinction that matters. A divider with out-degree 1 necessarily
    # parents ONE daughter, so "parented" on its own is the normal case. What
    # blocks admission is a daughter parented by someone ELSE: the orphan rule
    # requires no parent at all, so no distance gate can ever reach it.
    foreign = (pl.col("daughters_parented") - pl.col("daughters_linked_to_divider"))
    print(f"  second daughter parented by a FOREIGN node : "
          f"{full.filter(foreign >= 1).height}"
          "   <- orphan rule; no distance gate can reach these")
    print(f"  second daughter genuinely an orphan        : "
          f"{full.filter(foreign == 0).height}"
          "   <- admissible, so a distance/symmetry/divergence gate rejected it")
    ranked = full.filter(pl.col("sister_orphan_rank") >= 0)
    if ranked.height:
        print(f"\nof the {ranked.height} where the sister's rank is computable -- is she "
              "the nearest orphan to the existing child?")
        print(f"  rank 0 (she IS the candidate) : "
              f"{ranked.filter(pl.col('sister_orphan_rank') == 0).height}"
              "   <- a gate rejected her; loosening could reach these")
        print(f"  rank 1-2                      : "
              f"{ranked.filter(pl.col('sister_orphan_rank').is_between(1, 2)).height}")
        print(f"  rank >2                       : "
              f"{ranked.filter(pl.col('sister_orphan_rank') > 2).height}"
              "   <- mutual-nearest never proposes her, at ANY gate")
        print(f"  median rank {ranked['sister_orphan_rank'].median()}")
    print("\nwhich node is missing, among FNs:")
    for k, v in Counter(fn_rows["missing"].to_list()).most_common():
        print(f"  {k:24s} {v}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
