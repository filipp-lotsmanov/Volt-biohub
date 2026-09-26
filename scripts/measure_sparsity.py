"""Measure ground-truth annotation sparsity against estimated_number_of_nodes.

Each dataset is two directories: <name>.zarr (image) and <name>.geff (graph).
This reads only the .geff metadata and node/edge index arrays, so it is cheap.

Run:
    uv run python scripts/measure_sparsity.py
    uv run python scripts/measure_sparsity.py --root data/train --out reports/sparsity.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import zarr


def measure(name: str, geff_path: Path) -> dict:
    row: dict = {
        "dataset": name, "n_nodes": None, "n_edges": None, "t_true": None,
        "sparsity": None, "t_min": None, "t_max": None, "n_timepoints_covered": None,
        "n_divisions": None, "n_track_starts": None, "max_out_degree": None,
        "note": "",
    }
    try:
        g = zarr.open_group(str(geff_path), mode="r")
    except Exception as exc:
        row["note"] = f"open failed: {exc}"
        return row

    meta = dict(g.attrs).get("geff", {})
    extra = meta.get("extra") or {}
    t_true = extra.get("estimated_number_of_nodes")
    if t_true is None:
        row["note"] = "estimated_number_of_nodes absent"
    else:
        row["t_true"] = float(t_true)

    try:
        node_ids = np.asarray(g["nodes/ids"])
        t = np.asarray(g["nodes/props/t/values"])
        edge_ids = np.asarray(g["edges/ids"])
    except Exception as exc:
        row["note"] = (row["note"] + f"; array read failed: {exc}").lstrip("; ")
        return row

    row["n_nodes"] = int(node_ids.shape[0])
    row["n_edges"] = int(edge_ids.shape[0])
    if t.size:
        row["t_min"] = int(t.min())
        row["t_max"] = int(t.max())
        row["n_timepoints_covered"] = int(np.unique(t).size)

    if row["t_true"] and row["t_true"] > 0:
        row["sparsity"] = row["n_nodes"] / row["t_true"]

    if edge_ids.size:
        out_deg = Counter(edge_ids[:, 0].tolist())
        in_deg = Counter(edge_ids[:, 1].tolist())
        row["max_out_degree"] = max(out_deg.values())
        row["n_divisions"] = sum(1 for v in out_deg.values() if v >= 2)
        row["n_track_starts"] = int(sum(1 for nid in node_ids.tolist()
                                        if in_deg.get(nid, 0) == 0))
    else:
        row["max_out_degree"] = 0
        row["n_divisions"] = 0
        row["n_track_starts"] = row["n_nodes"]

    return row


def describe(label: str, vals: np.ndarray) -> str:
    if vals.size == 0:
        return f"{label:<28} no data"
    return (f"{label:<28} n={vals.size:<4} min={vals.min():<12.6g} "
            f"median={np.median(vals):<12.6g} max={vals.max():<12.6g}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("reports/sparsity.csv"))
    args = ap.parse_args(argv)

    root = args.root
    if root is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dataspec import DATA_ROOT  # noqa: PLC0415
        root = Path(DATA_ROOT) / args.split

    geffs = sorted(root.glob("*.geff"))
    zarrs = sorted(root.glob("*.zarr"))
    if not geffs:
        print(f"No .geff found under {root}", file=sys.stderr)
        return 1

    geff_names = {p.stem for p in geffs}
    zarr_names = {p.stem for p in zarrs}
    if geff_names != zarr_names:
        print(f"Pairing mismatch: {len(zarr_names - geff_names)} images without a "
              f"graph, {len(geff_names - zarr_names)} graphs without an image",
              file=sys.stderr)
        for n in sorted(zarr_names - geff_names)[:10]:
            print(f"  image only: {n}", file=sys.stderr)

    rows = [measure(p.stem, p) for p in geffs]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "n_nodes", "n_edges", "t_true", "sparsity", "t_min",
              "t_max", "n_timepoints_covered", "n_divisions", "n_track_starts",
              "max_out_degree", "note"]
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    def col(key: str, subset: list[dict] | None = None) -> np.ndarray:
        src = rows if subset is None else subset
        return np.array([r[key] for r in src if r.get(key) is not None], dtype=float)

    sp = col("sparsity")
    print(f"\n{len(rows)} graphs read, {sp.size} measurable\n")
    print(describe("sparsity (n_nodes/t_true)", sp))
    print(describe("n_nodes", col("n_nodes")))
    print(describe("n_edges", col("n_edges")))
    print(describe("t_true", col("t_true")))
    print(describe("timepoints covered", col("n_timepoints_covered")))
    print(describe("track starts", col("n_track_starts")))
    print(describe("divisions", col("n_divisions")))

    if sp.size:
        print(f"\nsparsity spread max/min: {sp.max() / sp.min():.1f}x")
        print(f"sparsity above 1.0:      {(sp > 1.0).sum()} datasets")
        print(f"total nodes:             {int(col('n_nodes').sum())}")
        print(f"total edges:             {int(col('n_edges').sum())}")
        print(f"total divisions:         {int(col('n_divisions').sum())}")
        print(f"datasets with divisions: {int((col('n_divisions') > 0).sum())}")

    mx = col("max_out_degree")
    if mx.size and mx.max() > 2:
        print(f"\nWARNING: max out-degree {mx.max():.0f} exceeds 2", file=sys.stderr)

    with_div = [r for r in rows if r.get("n_divisions")]
    without = [r for r in rows if r.get("n_divisions") == 0]
    print("\nSplit by division presence")
    print(describe("  sparsity, with div", col("sparsity", with_div)))
    print(describe("  sparsity, no div", col("sparsity", without)))
    print(describe("  n_nodes, with div", col("n_nodes", with_div)))
    print(describe("  n_nodes, no div", col("n_nodes", without)))

    failed = [r for r in rows if r["sparsity"] is None]
    if failed:
        print(f"\n{len(failed)} unmeasurable:", file=sys.stderr)
        for r in failed[:10]:
            print(f"  {r['dataset']}: {r['note']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())