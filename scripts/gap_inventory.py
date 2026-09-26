"""Inventory every gap in the ground-truth annotation timeline.

A gap is a run of consecutive timepoints with zero annotated nodes, lying
strictly inside a dataset's annotated span (t_min..t_max). For each gap this
reports its length and the mean image intensity of the gap frames against the
frames immediately either side, so dropped image frames can be told apart from
annotation that simply stopped.

Output is sorted by gap length, longest first.

Cost: reads one frame (8 MB) per gap frame plus two flanking frames per gap.

Run:
    uv run python scripts/gap_inventory.py
    uv run python scripts/gap_inventory.py --max-gap-frames 400
    uv run python scripts/gap_inventory.py --no-intensity
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import zarr

N_FRAMES = 100
DEAD_FRAME_RATIO = 0.5


def find_gaps(t_values: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) inclusive runs of empty timepoints inside the span."""
    present = set(int(v) for v in t_values.tolist())
    if not present:
        return []
    lo, hi = min(present), max(present)
    missing = sorted(set(range(lo, hi + 1)) - present)
    runs: list[tuple[int, int]] = []
    for f in missing:
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    return runs


def frame_mean(img: zarr.Array, t: int, cache: dict[int, float]) -> float:
    if t not in cache:
        cache[t] = float(np.asarray(img[t]).mean())
    return cache[t]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("reports/gap_inventory.csv"))
    ap.add_argument("--no-intensity", action="store_true",
                    help="Skip image reads. Sizes only, runs in seconds.")
    ap.add_argument("--max-gap-frames", type=int, default=600,
                    help="Abort intensity checks after this many frames read.")
    args = ap.parse_args(argv)

    root = args.root
    if root is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dataspec import DATA_ROOT  # noqa: PLC0415
        root = Path(DATA_ROOT) / args.split

    geffs = sorted(root.glob("*.geff"))
    if not geffs:
        print(f"No .geff found under {root}", file=sys.stderr)
        return 1

    gaps: list[dict] = []
    for p in geffs:
        try:
            g = zarr.open_group(str(p), mode="r")
            t = np.asarray(g["nodes/props/t/values"])
        except Exception as exc:
            print(f"  {p.stem}: graph read failed: {exc}", file=sys.stderr)
            continue
        present = set(int(v) for v in t.tolist())
        for start, end in find_gaps(t):
            gaps.append({
                "dataset": p.stem, "gap_start": start, "gap_end": end,
                "gap_len": end - start + 1,
                "span_start": min(present), "span_end": max(present),
                "n_nodes": int(t.size),
                "mean_before": None, "mean_in_gap_min": None,
                "mean_in_gap_max": None, "mean_after": None,
                "gap_over_flank": None, "verdict": "",
            })

    if not gaps:
        print("No gaps found.")
        return 0

    gaps.sort(key=lambda r: (-r["gap_len"], r["dataset"], r["gap_start"]))

    total_gap_frames = sum(r["gap_len"] for r in gaps)
    print(f"{len(gaps)} gaps across "
          f"{len({r['dataset'] for r in gaps})} datasets, "
          f"{total_gap_frames} gap frames total\n")

    if not args.no_intensity:
        if total_gap_frames > args.max_gap_frames:
            print(f"Gap frames ({total_gap_frames}) exceeds --max-gap-frames "
                  f"({args.max_gap_frames}). Checking intensity for the shortest "
                  f"gaps first, which are the dropped-frame candidates.\n")
        budget = args.max_gap_frames
        by_len = sorted(gaps, key=lambda r: r["gap_len"])
        img_cache: dict[str, zarr.Array] = {}
        frame_cache: dict[str, dict[int, float]] = {}

        for row in by_len:
            if row["gap_len"] + 2 > budget:
                continue
            name = row["dataset"]
            try:
                if name not in img_cache:
                    img_cache[name] = zarr.open_group(
                        str(root / f"{name}.zarr"), mode="r")["0"]
                    frame_cache[name] = {}
                img = img_cache[name]
                cache = frame_cache[name]

                before = row["gap_start"] - 1
                after = row["gap_end"] + 1
                inside = [frame_mean(img, f, cache)
                          for f in range(row["gap_start"], row["gap_end"] + 1)]
                budget -= row["gap_len"]

                flanks = []
                if before >= 0:
                    row["mean_before"] = frame_mean(img, before, cache)
                    flanks.append(row["mean_before"])
                    budget -= 1
                if after < N_FRAMES:
                    row["mean_after"] = frame_mean(img, after, cache)
                    flanks.append(row["mean_after"])
                    budget -= 1

                row["mean_in_gap_min"] = min(inside)
                row["mean_in_gap_max"] = max(inside)
                if flanks:
                    flank = float(np.mean(flanks))
                    if flank > 0:
                        ratio = min(inside) / flank
                        row["gap_over_flank"] = ratio
                        row["verdict"] = ("dead frame" if ratio < DEAD_FRAME_RATIO
                                          else "image fine")
            except Exception as exc:
                row["verdict"] = f"image read failed: {exc}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gaps[0].keys()))
        w.writeheader()
        w.writerows(gaps)

    def fmt(v: float | None, width: int = 9) -> str:
        return f"{v:>{width}.1f}" if isinstance(v, float) else f"{'-':>{width}}"

    print(f"{'dataset':<20}{'gap':>10}{'len':>5}{'before':>9}{'in gap':>9}"
          f"{'after':>9}{'ratio':>7}  verdict")
    for r in gaps:
        gap_str = (f"{r['gap_start']}" if r["gap_len"] == 1
                   else f"{r['gap_start']}-{r['gap_end']}")
        ratio = (f"{r['gap_over_flank']:>7.2f}"
                 if isinstance(r["gap_over_flank"], float) else f"{'-':>7}")
        print(f"{r['dataset']:<20}{gap_str:>10}{r['gap_len']:>5}"
              f"{fmt(r['mean_before'])}{fmt(r['mean_in_gap_min'])}"
              f"{fmt(r['mean_after'])}{ratio}  {r['verdict']}")

    lens = np.array([r["gap_len"] for r in gaps])
    print(f"\ngap length: min={lens.min()} median={np.median(lens):.0f} "
          f"max={lens.max()}  total frames={lens.sum()}")
    for bucket, lo, hi in (("1 frame", 1, 1), ("2-5", 2, 5),
                           ("6-15", 6, 15), ("16+", 16, 10**6)):
        n = int(((lens >= lo) & (lens <= hi)).sum())
        frames = int(lens[(lens >= lo) & (lens <= hi)].sum())
        print(f"  {bucket:<10} {n:>3} gaps, {frames:>4} frames")

    dead = [r for r in gaps if r["verdict"] == "dead frame"]
    fine = [r for r in gaps if r["verdict"] == "image fine"]
    if dead or fine:
        print(f"\ndead frames: {len(dead)} gaps "
              f"({sum(r['gap_len'] for r in dead)} frames)")
        print(f"image fine:  {len(fine)} gaps "
              f"({sum(r['gap_len'] for r in fine)} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
