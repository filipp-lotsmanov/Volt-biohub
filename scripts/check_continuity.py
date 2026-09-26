"""Verify temporal continuity of ground-truth annotation.

Answers, per dataset and in aggregate:
  - Does every edge span exactly one timepoint?
  - Are there empty timepoints inside a dataset's annotated span?
  - How long are tracks, and where do they start and end?
  - Do track ends coincide with the volume boundary (cell left the field) or
    sit mid-volume (annotation stopped)?

Reads only .geff metadata and node/edge arrays. No image voxels.

Run:
    uv run python scripts/check_continuity.py
    uv run python scripts/check_continuity.py --root data/train
    uv run python scripts/check_continuity.py --dump 44b6_0113de3b
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zarr

N_FRAMES = 100
VOLUME_ZYX = (64, 256, 256)
BORDER_MARGIN = 3


def load(geff_path: Path) -> dict:
    g = zarr.open_group(str(geff_path), mode="r")
    node_ids = np.asarray(g["nodes/ids"])
    props = {k: np.asarray(g[f"nodes/props/{k}/values"]) for k in ("t", "z", "y", "x")}
    edges = np.asarray(g["edges/ids"])
    meta = dict(g.attrs).get("geff", {})
    return {"node_ids": node_ids, "props": props, "edges": edges, "meta": meta}


def near_border(z: int, y: int, x: int) -> bool:
    limits = VOLUME_ZYX
    for val, lim in zip((z, y, x), limits):
        if val <= BORDER_MARGIN or val >= lim - 1 - BORDER_MARGIN:
            return True
    return False


def walk_branches(node_ids: np.ndarray, edges: np.ndarray) -> list[list[int]]:
    """Split the graph into unbranched chains, cutting at every fork and merge."""
    succ: dict[int, list[int]] = defaultdict(list)
    pred: dict[int, list[int]] = defaultdict(list)
    for s, d in edges.tolist():
        succ[s].append(d)
        pred[d].append(s)

    starts = [int(n) for n in node_ids.tolist() if len(pred.get(int(n), ())) != 1]
    fork_children = [c for n in node_ids.tolist()
                     if len(succ.get(int(n), ())) > 1
                     for c in succ[int(n)]]
    seeds = list(dict.fromkeys(starts + [int(c) for c in fork_children]))

    branches: list[list[int]] = []
    for seed in seeds:
        chain = [seed]
        cur = seed
        while len(succ.get(cur, ())) == 1:
            nxt = succ[cur][0]
            if len(pred.get(nxt, ())) != 1:
                break
            chain.append(nxt)
            cur = nxt
        branches.append(chain)
    return branches


def analyse(name: str, geff_path: Path) -> tuple[dict, list[dict]]:
    d = load(geff_path)
    node_ids, edges = d["node_ids"], d["edges"]
    t = d["props"]["t"]
    tmap = {int(n): int(tv) for n, tv in zip(node_ids.tolist(), t.tolist())}
    pos = {int(n): (int(z), int(y), int(x)) for n, z, y, x in
           zip(node_ids.tolist(), d["props"]["z"].tolist(),
               d["props"]["y"].tolist(), d["props"]["x"].tolist())}

    row: dict = {"dataset": name, "n_nodes": int(node_ids.size),
                 "n_edges": int(edges.size and edges.shape[0])}

    if edges.size:
        dt = np.array([tmap[int(dst)] - tmap[int(src)] for src, dst in edges.tolist()])
        row["edge_dt_min"] = int(dt.min())
        row["edge_dt_max"] = int(dt.max())
        row["n_edges_dt_not_1"] = int((dt != 1).sum())
    else:
        row["edge_dt_min"] = row["edge_dt_max"] = row["n_edges_dt_not_1"] = 0

    present = sorted(set(int(v) for v in t.tolist()))
    row["t_min"], row["t_max"] = present[0], present[-1]
    span = row["t_max"] - row["t_min"] + 1
    row["span"] = span
    row["n_covered"] = len(present)
    inner_missing = sorted(set(range(row["t_min"], row["t_max"] + 1)) - set(present))
    row["n_inner_gaps"] = len(inner_missing)
    row["inner_gap_frames"] = "|".join(str(v) for v in inner_missing[:20])
    row["frac_of_recording"] = span / N_FRAMES
    row["starts_at_zero"] = row["t_min"] == 0
    row["ends_at_last"] = row["t_max"] == N_FRAMES - 1

    per_frame = Counter(int(v) for v in t.tolist())
    counts = np.array([per_frame.get(f, 0) for f in range(row["t_min"], row["t_max"] + 1)])
    row["nodes_per_frame_min"] = int(counts.min())
    row["nodes_per_frame_median"] = float(np.median(counts))
    row["nodes_per_frame_max"] = int(counts.max())

    branches = walk_branches(node_ids, edges)
    row["n_branches"] = len(branches)
    lengths = np.array([len(b) for b in branches])
    row["branch_len_min"] = int(lengths.min()) if lengths.size else 0
    row["branch_len_median"] = float(np.median(lengths)) if lengths.size else 0.0
    row["branch_len_max"] = int(lengths.max()) if lengths.size else 0

    succ_count = Counter(int(s) for s, _ in edges.tolist()) if edges.size else Counter()
    terminal = [b[-1] for b in branches if succ_count.get(b[-1], 0) == 0]
    row["n_terminal"] = len(terminal)
    row["n_terminal_at_last_frame"] = sum(1 for n in terminal
                                          if tmap[n] == N_FRAMES - 1)
    row["n_terminal_at_border"] = sum(1 for n in terminal if near_border(*pos[n]))
    row["n_terminal_interior_midtime"] = sum(
        1 for n in terminal
        if tmap[n] != N_FRAMES - 1 and not near_border(*pos[n]))

    detail = [{"dataset": name, "branch_index": i, "length": len(b),
               "t_start": tmap[b[0]], "t_end": tmap[b[-1]],
               "ends_at_last_frame": tmap[b[-1]] == N_FRAMES - 1,
               "ends_at_border": near_border(*pos[b[-1]]),
               "has_successor": succ_count.get(b[-1], 0) > 0}
              for i, b in enumerate(branches)]
    return row, detail


def describe(label: str, vals: np.ndarray) -> str:
    if vals.size == 0:
        return f"{label:<34} no data"
    return (f"{label:<34} min={vals.min():<10.4g} median={np.median(vals):<10.4g} "
            f"max={vals.max():<10.4g}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("reports/continuity.csv"))
    ap.add_argument("--branches-out", type=Path,
                    default=Path("reports/continuity_branches.csv"))
    ap.add_argument("--dump", default=None,
                    help="Print the per-frame node count for one dataset and exit.")
    args = ap.parse_args(argv)

    root = args.root
    if root is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dataspec import DATA_ROOT  # noqa: PLC0415
        root = Path(DATA_ROOT) / args.split

    if args.dump:
        d = load(root / f"{args.dump}.geff")
        t = d["props"]["t"]
        per_frame = Counter(int(v) for v in t.tolist())
        print(f"{args.dump}: per-frame annotated node count")
        for f in range(N_FRAMES):
            n = per_frame.get(f, 0)
            print(f"  t={f:<3} {n:<4} {'#' * n if n <= 60 else '#' * 60 + '...'}")
        return 0

    geffs = sorted(root.glob("*.geff"))
    if not geffs:
        print(f"No .geff found under {root}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    details: list[dict] = []
    errors: list[tuple[str, str]] = []
    for p in geffs:
        try:
            row, detail = analyse(p.stem, p)
        except Exception as exc:
            errors.append((p.stem, str(exc)))
            continue
        rows.append(row)
        details.extend(detail)

    if not rows:
        print("Nothing analysed.", file=sys.stderr)
        for name, err in errors[:10]:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with args.branches_out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(details[0].keys()))
        w.writeheader()
        w.writerows(details)

    def col(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)

    print(f"\n{len(rows)} datasets analysed, {len(details)} branches total\n")

    print("Edge dt")
    bad_dt = [r for r in rows if r["n_edges_dt_not_1"] > 0]
    print(f"  datasets with any edge dt != 1: {len(bad_dt)}")
    if bad_dt:
        for r in bad_dt[:10]:
            print(f"    {r['dataset']}: {r['n_edges_dt_not_1']} edges, "
                  f"dt range {r['edge_dt_min']}..{r['edge_dt_max']}")

    print("\nGaps inside the annotated span")
    gapped = [r for r in rows if r["n_inner_gaps"] > 0]
    print(f"  datasets with an empty timepoint inside span: {len(gapped)}")
    for r in gapped[:15]:
        print(f"    {r['dataset']}: {r['n_inner_gaps']} gaps, span "
              f"{r['t_min']}..{r['t_max']}, frames [{r['inner_gap_frames']}]")

    print("\nSpan coverage")
    print(describe("  span (frames)", col("span")))
    print(describe("  frac of 100-frame recording", col("frac_of_recording")))
    print(f"  datasets starting at t=0:  {sum(r['starts_at_zero'] for r in rows)}")
    print(f"  datasets ending at t=99:   {sum(r['ends_at_last'] for r in rows)}")

    print("\nPer-frame annotated node count, within span")
    print(describe("  min over frames", col("nodes_per_frame_min")))
    print(describe("  median over frames", col("nodes_per_frame_median")))
    print(describe("  max over frames", col("nodes_per_frame_max")))

    print("\nBranches (chains cut at forks)")
    print(describe("  branches per dataset", col("n_branches")))
    bl = np.array([d["length"] for d in details], dtype=float)
    print(describe("  branch length (nodes)", bl))
    print(f"  branches of length 1: {int((bl == 1).sum())}")

    print("\nWhy branches end")
    term = col("n_terminal").sum()
    print(f"  terminal branches:              {int(term)}")
    print(f"  ending at t=99:                 {int(col('n_terminal_at_last_frame').sum())}")
    print(f"  ending near volume border:      {int(col('n_terminal_at_border').sum())}")
    print(f"  interior and before t=99:       {int(col('n_terminal_interior_midtime').sum())}")

    if errors:
        print(f"\n{len(errors)} datasets failed:", file=sys.stderr)
        for name, err in errors[:10]:
            print(f"  {name}: {err}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
