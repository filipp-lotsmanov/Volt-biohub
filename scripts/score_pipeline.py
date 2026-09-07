#!/usr/bin/env python
"""Score a full detect -> link -> graph pipeline with the competition metric.

One harness, a swappable detection front-end. ``--detector dog`` is the
classical floor; ``--detector unet`` runs a trained checkpoint's detection head.
Everything after detection is identical, which is the only way a comparison
between them means anything.

The learned *linker* is deliberately not used. Measured over all 199 samples,
the half-distance to a cell's nearest true neighbour beats its per-frame motion
by 4.9x at the median and the typical case is unambiguous in 199/199, so a
geometric linker is close to optimal by construction — and the transformer
currently emits zero edges, so it cannot serve as a fixed downstream stage while
we ask questions about detection.

Writes one ``experiments.csv`` row per run, per the project's experiment
discipline, and prints per-sample rows so a regression is attributable.

Examples
--------
    uv run python scripts/score_pipeline.py --detector dog --split 0 --limit 20
    CUDA_VISIBLE_DEVICES=$(volt_gpu) uv run python scripts/score_pipeline.py \
        --detector unet --split 0 --weights weights/modelA/split_0/edge_predictor_best.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dataspec import DATASET_PATH
from volt_biohub.linking import DIVISION_GATE_UM, MOTION_GATE_UM
from volt_biohub.metrics import summarise
from volt_biohub.scoring import DEFAULT_SEPARATION_UM, DOWNSAMPLE, score_video


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def append_experiment(path: Path, record: dict) -> None:
    """One row per run. Created with a header if absent, appended otherwise."""
    fields = [
        "timestamp", "git_sha", "machine", "run", "detector", "split", "n_datasets",
        "budget", "separation_um", "gate_um", "division_gate_um",
        "score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard",
        "node_recall", "node_count_ratio", "seconds", "notes",
    ]
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(record)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", choices=("dog", "unet"), default="dog")
    p.add_argument("--weights", type=Path, default=None,
                   help="Checkpoint for --detector unet.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--splits", type=Path, default=None)
    p.add_argument("--split", type=int, default=0,
                   help="Fold index in the splits file; its 'test' list is scored.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Explicit dataset names, overriding --split.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--budget", type=float, default=1.0,
                   help="Target node count as a multiple of t_true (default 1.0).")
    p.add_argument("--separation-um", type=float, default=DEFAULT_SEPARATION_UM,
                   help="Minimum peak separation. Default 2*sigma; the hard floor "
                        "is the 2.87 um closest division sister pair.")
    p.add_argument("--gate-um", type=float, default=MOTION_GATE_UM)
    p.add_argument("--division-gate-um", type=float, default=DIVISION_GATE_UM)
    p.add_argument("--refine", action="store_true",
                   help="Response-weighted centroid within each peak's neighbourhood. "
                        "Aimed at temporal jitter, not matching precision: the U-Net's "
                        "saturated map puts each cell on a plateau whose winning voxel "
                        "moves between frames, and the linker resolves 1.82um motion.")
    p.add_argument("--no-divisions", action="store_true",
                   help="Forbid a second child. The control arm for the division term.")
    p.add_argument("--run", type=str, default="",
                   help="Short run name recorded in experiments.csv.")
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--experiments", type=Path, default=Path("experiments.csv"))
    args = p.parse_args(argv)

    data_dir = args.data_dir or Path(DATASET_PATH)
    if args.datasets:
        names = list(args.datasets)
    else:
        splits = args.splits or data_dir / "dataset_splits.json"
        names = json.loads(splits.read_text())[args.split]["test"]
    if args.limit:
        names = names[:args.limit]

    model = device = None
    if args.detector == "unet":
        if args.weights is None:
            p.error("--detector unet needs --weights")
        import torch

        from predict_unet_transformer import load_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, ds = load_model(args.weights, device)
        if tuple(ds) != DOWNSAMPLE:
            p.error(f"checkpoint trained at downsample {tuple(ds)}, harness assumes {DOWNSAMPLE}")

    print(f"{args.detector} · {len(names)} datasets · budget {args.budget}x t_true · "
          f"sep {args.separation_um} um · gate {args.gate_um} um · "
          f"div gate {'off' if args.no_divisions else args.division_gate_um} · "
          f"refine {args.refine}",
          flush=True)

    rows, t0 = [], time.monotonic()
    for i, name in enumerate(names, 1):
        r = score_video(
            data_dir, name, model=model, device=device, budget=args.budget,
            separation_um=args.separation_um, gate_um=args.gate_um,
            division_gate_um=None if args.no_divisions else args.division_gate_um,
            refine=args.refine,
        )
        rows.append(r)
        print(f"  [{i:3d}/{len(names)}] {name:16s} adjJ={r['adj_edge_jaccard']:.4f} "
              f"J={r['edge_jaccard']:.4f} rec={r['node_recall']:.4f} "
              f"n_pred={r['n_pred']:6d} ratio={r['total_node_ratio']:+.2f}", flush=True)

    s = summarise(rows)
    dt = time.monotonic() - t0
    mean_recall = float(np.mean([r["node_recall"] for r in rows])) if rows else float("nan")
    mean_ratio = float(np.nanmean([r["total_node_ratio"] for r in rows])) if rows else float("nan")

    print(f"\n=== {s['n']} datasets · {dt:.0f}s ({dt / max(len(names), 1):.1f}s each) ===")
    for k in ("score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard"):
        print(f"  {k:18s} {s[k]:.4f}")
    print(f"  {'node_recall':18s} {mean_recall:.4f}")
    print(f"  {'node_count_ratio':18s} {mean_ratio:+.4f}")

    append_experiment(args.experiments, {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(), "machine": "linux-server",
        "run": args.run or args.detector, "detector": args.detector,
        "split": args.split if not args.datasets else "explicit",
        "n_datasets": s["n"], "budget": args.budget,
        "separation_um": args.separation_um, "gate_um": args.gate_um,
        "division_gate_um": "off" if args.no_divisions else args.division_gate_um,
        "score": f"{s['score']:.4f}", "adj_edge_jaccard": f"{s['adj_edge_jaccard']:.4f}",
        "edge_jaccard": f"{s['edge_jaccard']:.4f}",
        "division_jaccard": f"{s['division_jaccard']:.4f}",
        "node_recall": f"{mean_recall:.4f}", "node_count_ratio": f"{mean_ratio:+.4f}",
        "seconds": f"{dt:.0f}", "notes": args.notes,
    })
    print(f"\nappended to {args.experiments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
