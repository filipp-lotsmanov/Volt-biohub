#!/usr/bin/env python
"""Split the train movies into --train-as-test lists for the Kaggle division diagnosis.

A Kaggle session is capped at 12 h, and x138 took ~3.5 min per movie on the visible
run (2 x T4) before the diagnosis's extra DeepCenter scoring. So the 199 train movies
go into several kernels. Movies with a GT division come first: they carry every
missed division the diagnosis buckets, so if GPU quota runs out, the runs already done
still answer the main question. Movies without one only add false forks (the cost
side of a gate change), so they come last.

    uv run python scripts/x138_divdiag_lists.py --out-dir /home/y2b/divdiag_lists
    # -> divdiag_1.txt, divdiag_2.txt, ... (division movies first), and a summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MIN_PER_MOVIE = 4.5   # visible run ~3.5 min/movie + the diagnosis's DeepCenter pass


def gt_divisions(geff: Path) -> int:
    """Number of GT dividing nodes (out-degree >= 2) in one movie."""
    import tracksdata as td

    r = td.graph.IndexedRXGraph.from_geff(geff)
    g = r[0] if isinstance(r, tuple) else r
    src = g.edge_attrs(attr_keys=[])["source_id"]
    return int((src.value_counts()["count"] >= 2).sum())


def make_lists(counts: dict[str, int], max_per_list: int) -> list[list[str]]:
    """Division movies first (most divisions first), then the rest, in chunks.

    Chunks are filled round-robin within each group so every list has a similar mix of
    embryos and sizes, and the division movies end up in the fewest lists.
    """
    with_div = sorted((s for s in counts if counts[s] > 0), key=lambda s: (-counts[s], s))
    without = sorted(s for s in counts if counts[s] == 0)
    lists: list[list[str]] = []
    for group in (with_div, without):
        if not group:
            continue
        n = -(-len(group) // max_per_list)
        lists += [group[i::n] for i in range(n)]
    return lists


def main(argv=None) -> int:
    from dataspec import DATASET_PATH

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None, help="train .geff directory")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-per-list", type=int, default=100,
                   help="movies per kernel; 100 x ~4.5 min is ~7.5 h, inside the 12 h cap")
    a = p.parse_args(argv)
    data_dir = a.data_dir or DATASET_PATH

    stems = sorted(q.name[:-5] for q in data_dir.iterdir() if q.name.endswith(".zarr"))
    if not stems:
        print(f"no .zarr movies under {data_dir}")
        return 1
    counts = {}
    for i, s in enumerate(stems, 1):
        geff = data_dir / f"{s}.geff"
        counts[s] = gt_divisions(geff) if geff.exists() else 0
        if i % 25 == 0:
            print(f"  read {i}/{len(stems)} GT graphs", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    lists = make_lists(counts, a.max_per_list)
    for k, chunk in enumerate(lists, 1):
        path = a.out_dir / f"divdiag_{k}.txt"
        path.write_text("\n".join(chunk) + "\n")
        n_div = sum(counts[s] for s in chunk)
        embryos = sorted({s.split("_", 1)[0] for s in chunk})
        print(f"{path}: {len(chunk)} movies, {n_div} GT divisions, embryos {embryos}, "
              f"~{len(chunk) * MIN_PER_MOVIE / 60:.1f} h")
    print(f"{sum(counts.values())} GT divisions in {sum(v > 0 for v in counts.values())} of "
          f"{len(stems)} movies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
