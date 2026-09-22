#!/usr/bin/env python
"""Score the SUBMISSION PATH end to end, not a parallel reimplementation of it.

Why this exists. ``scripts/score_pipeline.py`` builds a tracksdata graph straight
from the linker's output and scores that; ``kaggle/submit.py`` writes a CSV and
Kaggle scores *that*. Two code paths, and they disagreed: CV read 0.7366 while the
leaderboard returned 0.588. Some of that gap was a real defect (the harness took
the node budget from the ground truth, which does not exist at test time), but the
structural problem is that nothing ever scored the artefact we actually ship.

So this runs ``submit.run`` — the real writer, with its real node-budget predictor,
its real isolated-node pruning and its real validator — over held-out *training*
videos, reads the CSV back, and scores it with the official metric. Anything that
corrupts coordinates, ids, edge direction or node counts between the linker and the
file shows up here rather than on the leaderboard.

The local ``test/`` folder is four byte-identical copies of train samples, so their
ground truth is readable; any train sample works the same way.

Examples
--------
    uv run python scripts/score_submission.py --split 1 --limit 20
    CUDA_VISIBLE_DEVICES=$(volt_gpu) uv run python scripts/score_submission.py \
        --split 1 --limit 20 --weights weights/E1_fix/split_1/edge_predictor_best.pth
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle"))

from csv_to_geffs import build_graph_from_rows
from dataspec import DATASET_PATH
from score_pipeline import append_experiment, git_sha
from volt_biohub.linking import DIVISION_GATE_UM, MOTION_GATE_UM
from volt_biohub.metrics import evaluate as score_pair
from volt_biohub.metrics import per_sample_metrics, summarise
from volt_biohub.scoring import (
    ORIGINAL_SCALE,
    load_gt,
    read_t_true,
    safe_node_recall,
)
from volt_biohub.submission import SubmissionError


def staged_test_dir(data_dir: Path, names: list[str], root: Path) -> Path:
    """A directory that looks like the competition's ``test/`` to ``submit.run``.

    Symlinks rather than copies: a video is ~411 MiB and the box is kept under
    100 GB. Only the ``.zarr`` is linked, never the ``.geff`` — if the ground
    truth were visible here, ``submit.run`` could in principle read the node count
    it is supposed to predict, and this harness would flatter itself exactly the
    way the one it replaces did.
    """
    test_dir = root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        link = test_dir / f"{n}.zarr"
        if not link.exists():
            link.symlink_to(data_dir / f"{n}.zarr")
    return test_dir


def score_csv(csv_path: Path, data_dir: Path, expected: list[str]) -> list[dict]:
    """Score a submission CSV against the ground truth, per dataset.

    *expected* is reconciled against what the CSV actually contains. Grouping by
    dataset and scoring whatever turns up silently drops a missing dataset from
    both the numerator and the denominator, so a run that lost a video reports a
    clean average over the survivors — the banner prints the requested count and
    the summary prints a smaller one, with nothing to reconcile them. On the real
    leaderboard a missing dataset is not averaged away, so this must be loud.
    """
    df = pl.read_csv(csv_path, columns=[
        "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
    rows: list[dict] = []
    for (name,), group in df.group_by("dataset"):
        nodes = group.filter(pl.col("row_type") == "node")
        edges = group.filter(pl.col("row_type") == "edge")
        # Coordinates in the CSV are already in original anisotropic voxel space,
        # which is the space the scorer works in -- so no upsample here. A missing
        # upsample in submit.py would show up as a collapsed score, which is the
        # point of scoring the file rather than the linker's output.
        pred = build_graph_from_rows(nodes, edges)
        gt = load_gt(data_dir, name)
        er = score_pair(pred, gt, scale=ORIGINAL_SCALE, max_distance=7.0)
        r = per_sample_metrics(er, read_t_true(data_dir, name), safe_node_recall(pred, gt))
        r["name"] = name
        r["n_pred"] = nodes.height
        r["n_edges_pred"] = edges.height
        rows.append(r)

    missing = sorted(set(expected) - {r["name"] for r in rows})
    if missing:
        raise SubmissionError(
            f"{len(missing)} requested dataset(s) absent from the CSV: {missing}. "
            "Every test dataset must appear in a submission, so scoring the rest "
            "would report a number the leaderboard will not give you."
        )
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--splits", type=Path, default=None)
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--weights", type=Path, default=None,
                   help="U-Net checkpoint. Omitted -> the classical DoG detector.")
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument("--separation-um", type=float, default=None,
                   help="Peak suppression radius. Default None means submit.py's own "
                        "DEFAULT_SEPARATION_UM (5.86 um) -- which is NOT the 4.0 every "
                        "tuned CV number in this repo was measured at. Not passing this "
                        "is how the harness built to catch CV/submission divergence "
                        "reproduced it.")
    p.add_argument("--gate-um", type=float, default=MOTION_GATE_UM)
    p.add_argument("--division-gate-um", type=float, default=DIVISION_GATE_UM)
    p.add_argument("--no-divisions", action="store_true")
    p.add_argument("--det-threshold", type=float, default=None,
                   help="Sigmoid threshold replacing the node budget.")
    p.add_argument("--learned-edges", action="store_true",
                   help="Use the checkpoint's trained transformer edge head.")
    p.add_argument("--edge-threshold", type=float, default=0.5)
    p.add_argument("--postprocess", action="store_true",
                   help="Full post-detection stack; replaces gate linking.")
    p.add_argument("--min-track-len", type=int, default=6)
    p.add_argument("--no-smooth", action="store_true",
                   help="Turn OFF line-fit trajectory smoothing. It had no CLI "
                        "path until now, so it has only ever run at its "
                        "defaults. The reference's clean ablation prices it at "
                        "+0.0072, the largest of their unambiguously positive "
                        "passes -- but every other component of theirs we "
                        "ported failed to reproduce its published delta here, "
                        "in both directions, so it is worth one arm.")
    p.add_argument("--smooth-window", type=int, default=None,
                   help="Half-width of the neighbourhood the linear fit uses "
                        "(default 2). Only nodes with an unambiguous chain on "
                        "both sides move, so a wider window reaches fewer "
                        "nodes, not more.")
    p.add_argument("--smooth-weight", type=float, default=None,
                   help="How far each node is pulled toward its fit, 0-1 "
                        "(default 0.8). Smoothing already pulled a border node "
                        "outside the grid once, which the validator caught and "
                        "the score would not have.")
    p.add_argument("--ilp", action="store_true",
                   help="Solve the RAW learned candidate graph with tracksdata's "
                        "ILP instead of assigning per frame pair. The reference's "
                        "own clean-split ablation prices this at +0.037, the "
                        "largest single component; our adjJ 0.858 sits exactly on "
                        "their 'ILP off' arm of 0.8587. Needs --weights.")
    p.add_argument("--ilp-prune", action="store_true",
                   help="Solve the link set the relink already resolved, so the "
                        "solver only decides what to KEEP. This is what the "
                        "reference does; --ilp hands it the raw candidate graph "
                        "instead, and that difference is -0.05 against +0.047 "
                        "(volt_biohub.ilp_link, measured 2026-09-21).")
    p.add_argument("--learned-bonus", type=float, default=0.0,
                   help="Subtract this x the edge-head probability from the "
                        "relink's Hungarian cost, in micrometres. The reference "
                        "ships 1.0 and we shipped nothing. Geometry still decides "
                        "the assignment; the model only breaks ties -- which is "
                        "the one way to spend these probabilities that we have not "
                        "measured losing (greedy edges 0.7632, ILP 0.8605).")
    p.add_argument("--edge-temperature", type=float, default=1.0,
                   help="Divide the edge logits before the softmax: below 1 "
                        "sharpens, above 1 smooths. Two smoothing operations "
                        "lost here (fusion -0.001, feature TTA -0.011), which "
                        "says the cost reads the RANKING; this tests the "
                        "opposite direction. Not the same as the bonus size.")
    p.add_argument("--secondary-weights", type=Path, default=None,
                   help="A second detector checkpoint to blend with. The public "
                        "cluster runs two (support-pack-50ep primary + seed314159 "
                        "secondary); we shipped one. Measured separately the two "
                        "are a dead heat but disagree in their errors.")
    p.add_argument("--secondary-weight", type=float, default=0.0,
                   help="Blend weight for the secondary, moment-matched in logit "
                        "space with a per-frame retention guard. Reference 0.80.")
    p.add_argument("--feature-tta", action="store_true",
                   help="Average the U-Net FEATURES over the 8 D4 views while "
                        "leaving detection sharp. Detection TTA loses here "
                        "(0.8609 vs 0.8687) by blurring daughters into one peak; "
                        "this touches only what the edge head reads, and is the "
                        "public chain's largest single step (0.941 -> 0.946).")
    p.add_argument("--secondary-edge-weight", type=float, default=0.0,
                   help="Ensemble the SECONDARY checkpoint's transformer edge "
                        "head at this weight (reference path only). Separate "
                        "from --secondary-weight, which blends detection maps: "
                        "the two halves of the same checkpoint are worth "
                        "different amounts and compose independently.")
    p.add_argument("--bidirectional-weight", type=float, default=0.0,
                   help="Fuse forward and reverse edge probabilities by a "
                        "weighted harmonic mean (reference 0.15, measured "
                        "+0.0053). Touches no detection and no geometry, so the "
                        "daughter-blurring that killed detection TTA cannot "
                        "occur -- it only sharpens what the bonus spends.")
    p.add_argument("--candidate-threshold", type=float, default=0.48,
                   help="Probability above which a pair becomes an ILP candidate. "
                        "No distance gate. Reference 0.48; 0.44 measured +0.003.")
    p.add_argument("--ilp-disappearance", type=float, default=None,
                   help="Cost of a track ending. Reference 2.0; 3.0 measured "
                        "+0.004 on 18 of 20 volumes.")
    p.add_argument("--ilp-appearance", type=float, default=None,
                   help="Cost of a track starting. Reference 0.0; 1.0 measured "
                        "-0.0055 and suppresses divisions.")
    p.add_argument("--ilp-node", type=float, default=None,
                   help="Reward for keeping a node (negative rewards, the solver "
                        "minimises). The reference leaves it 0.0, which makes "
                        "deleting a node free and costs us node recall.")
    p.add_argument("--ilp-edge", type=float, default=None,
                   help="Edge weight multiplying the probability. Reference -1.0, "
                        "so an edge is worth at most 1.0 against a disappearance "
                        "cost of 2.0 -- scale it up to make links pay.")
    p.add_argument("--ilp-division", type=float, default=None,
                   help="Cost of a fork. INERT at any value >= 1.0, so the "
                        "shipped 1.2 means prune_dp never emits a fork at all "
                        "and every division we score comes from "
                        "add_safe_divisions downstream. Proof: a fork is taken "
                        "iff w_div + g2 < 0 where "
                        "g = edge_weight*p + (on[c] - w_app) - free[c]; with "
                        "APPEARANCE_WEIGHT 0.0, EDGE_WEIGHT -1.0 and "
                        "free[c] = min(on[c], base) <= on[c], that is g >= -p "
                        ">= -1, so the branch is unreachable for any input at "
                        "w_div >= 1. Confirmed: 1.2 and 1.6 return bit-identical "
                        "metrics. Only below 1.0 does it move anything (0.8 "
                        "gives divJ 0.0902 -> 0.0939, about four divisions, and "
                        "+0.0004 of score, i.e. noise). The earlier help here "
                        "said \"0.8 and 1.6 both lose\", which was measured "
                        "against the tracksdata SOLVER, not prune_dp.")
    p.add_argument("--no-gaps", action="store_true",
                   help="Skip single-frame gap closing, which the reference's "
                        "ablation prices at +0.0004 once a solver picks topology.")
    p.add_argument("--div-symmetry-tau", type=float, default=None,
                   help="Raise to demand more even splits. Full-fold decomposition "
                        "at the default 0.6: TP 7, FP 40, FN 19 -- FP is 6x TP, so "
                        "the precision side may already be past its optimum.")
    p.add_argument("--div-diverge-um", type=float, default=None,
                   help="Minimum extra separation of the grandchildren over the "
                        "sisters. Raise to demand stronger post-mitotic divergence.")
    p.add_argument("--div-frame-frac", type=float, default=None,
                   help="Per-frame fork cap as a fraction of that frame's nodes. "
                        "It interacts with the ILP prune, which cuts node counts "
                        "by about a fifth -- so the cap tightens by the same "
                        "fifth without anyone asking it to, and divJ fell "
                        "0.1034 -> 0.0822 on fold 1 and 0.0738 -> 0.0597 on "
                        "fold 0 when the prune landed.")
    p.add_argument("--div-global-frac", type=float, default=None,
                   help="Cap on forks as a fraction of edges. A blunt count limit: "
                        "it drops candidates by order, not by quality.")
    p.add_argument("--div-orphan-probs", action="store_true",
                   help="Attach the edge head's DISCARDED probabilities to the "
                        "division pass. The greedy resolution keeps a pair only "
                        "if it clears 0.5 and wins the <=1-parent pass, so an "
                        "unparented target -- exactly what a second daughter is "
                        "-- reaches post-processing with no learned evidence. "
                        "Diagnostic for now: it only populates the learned_prob "
                        "column of VOLT_DIV_REJECT, and changes no decision.")
    p.add_argument("--div-n-cand", type=int, default=None,
                   help="How many of the nearest orphans to try as the second "
                        "daughter, as a FALLBACK: the first to clear every gate "
                        "wins, so the pool size and out-degree are unchanged. "
                        "1 is the reference's mutual-nearest rule. The measured "
                        "case for raising it: the per-frame cap holds a median "
                        "of 1 candidate against a cap of 2 and binds in only "
                        "12.8 % of frames, every true fork in the pool already "
                        "ranks first and is accepted (6/6), and ~1 % of emitted "
                        "forks are charged as FP at all -- so the term is "
                        "recall-limited and the single-nearest rule, not any "
                        "threshold, is what keeps the true sister out.")
    p.add_argument("--div-parent-um", type=float, default=None)
    p.add_argument("--div-sister-um", type=float, default=None)
    p.add_argument("--div-no-grandchildren", action="store_true",
                   help="Stop REQUIRING both daughters to have a child before a "
                        "fork is admitted. The forward-divergence DISTANCE is "
                        "measured as already optimal (loosening it to 1.0 and "
                        "-2.0 both lose monotonically), but its structural half "
                        "was unreachable from any knob and demands two further "
                        "frames of correct linking on both branches. Absence of "
                        "a grandchild is not evidence against a division.")
    p.add_argument("--div-rise-weight", type=float, default=None,
                   help="Weight on the parent's own two-frame intensity rise in "
                        "the fork ranking, micrometre-equivalent. The only fork "
                        "feature measured to separate: AUC 0.632 over all 151 GT "
                        "divisions against 1,729 controls, where every geometric "
                        "feature sits at 0.50. A dividing parent DIMS (median "
                        "rise 0.901 against 1.002), so a positive weight "
                        "penalises brightening candidates.")
    p.add_argument("--tta", action="store_true",
                   help="8-view D4 TTA in (y, x), averaged in logit space.")
    p.add_argument("--tta-views", type=int, default=8, choices=(4, 8),
                   help="8 is the full D4 group (what our -0.015 measured); 4 is "
                        "the flip subgroup, which is what the reference's "
                        "det_tta actually does and what its +0.009 refers to. "
                        "Rotating a 90-degree turn through an anisotropic PSF is "
                        "a much stronger perturbation than a mirror.")
    p.add_argument("--reference", action="store_true",
                   help="Run volt_biohub.reference_predict -- the upstream "
                        "inference path copied verbatim -- instead of this "
                        "repo's detect-and-link. On the public notebooks' own "
                        "4-movie proxy set it reaches adjusted edge Jaccard "
                        "0.9299 where the rewrite reaches 0.8888, and 0.9505 "
                        "total once our post-processing runs on top.")
    p.add_argument("--real-pairs", action="store_true",
                   help="Compute the edge head's trunk features from a REAL "
                        "consecutive pair instead of one frame duplicated. The "
                        "fake pair is free for detection (node recall 0.9973 "
                        "against 0.9967) and is the leading suspect for the edge "
                        "head scoring 0.86 here against 0.93 through the "
                        "reference's own code on the same weights.")
    p.add_argument("--switch-at", type=float, default=None)
    p.add_argument("--refine", action="store_true", default=True)
    p.add_argument("--keep-csv", type=Path, default=None)
    p.add_argument("--run", type=str, default="")
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--experiments", type=Path, default=Path("experiments.csv"))
    args = p.parse_args(argv)

    import submit

    data_dir = args.data_dir or Path(DATASET_PATH)
    if args.datasets:
        names = list(args.datasets)
    else:
        splits = args.splits or data_dir / "dataset_splits.json"
        names = json.loads(splits.read_text())[args.split]["test"]
    if args.limit:
        names = names[:args.limit]

    print(f"submission path · {len(names)} datasets · "
          f"{'unet' if args.weights else 'dog'} · gate {args.gate_um} um · "
          f"div gate {'off' if args.no_divisions else args.division_gate_um}", flush=True)

    t0 = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="volt-subscore-"))
    try:
        test_dir = staged_test_dir(data_dir, names, tmp)
        csv_path = tmp / "submission.csv"
        extra = {} if args.separation_um is None else {"separation_um": args.separation_um}
        submit.run(
            test_dir, csv_path, **extra,
            weights=args.weights, budget=args.budget, gate_um=args.gate_um,
            division_gate_um=None if args.no_divisions else args.division_gate_um,
            refine=args.refine, switch_at=args.switch_at,
            det_threshold=args.det_threshold,
            postprocess=args.postprocess,
            min_track_len=args.min_track_len,
            learned_edges=args.learned_edges,
            edge_threshold=args.edge_threshold,
            tta=args.tta,
            tta_views=args.tta_views,
            ilp=args.ilp,
            ilp_prune=args.ilp_prune,
            learned_bonus=args.learned_bonus,
            secondary_edge_weight=args.secondary_edge_weight,
            bidirectional_weight=args.bidirectional_weight,
            feature_tta=args.feature_tta,
            secondary_weights=args.secondary_weights,
            secondary_weight=args.secondary_weight,
            temperature=args.edge_temperature,
            candidate_threshold=args.candidate_threshold,
            real_pairs=args.real_pairs,
            reference=args.reference,
            gaps=not args.no_gaps,
            ilp_kwargs={k: v for k, v in {
                "disappearance_weight": args.ilp_disappearance,
                "appearance_weight": args.ilp_appearance,
                "division_weight": args.ilp_division,
                "node_weight": args.ilp_node,
                "edge_weight": args.ilp_edge,
            }.items() if v is not None} or None,
            div_orphan_probs=args.div_orphan_probs,
            smooth=not args.no_smooth,
            smooth_kwargs={k: v for k, v in {
                "window": args.smooth_window,
                "weight": args.smooth_weight,
            }.items() if v is not None} or None,
            div_kwargs={k: v for k, v in {
                "symmetry_tau": args.div_symmetry_tau,
                "diverge_um": args.div_diverge_um,
                "global_frac": args.div_global_frac,
                "frame_frac": args.div_frame_frac,
                "parent_um": args.div_parent_um,
                "sister_um": args.div_sister_um,
                "n_cand": args.div_n_cand,
                "rise_weight": args.div_rise_weight,
                # False is meaningful here, so it cannot ride the
                # `v is not None` filter as a plain flag default.
                "require_grandchildren": (
                    False if args.div_no_grandchildren else None),
            }.items() if v is not None} or None,
        )
        rows = score_csv(csv_path, data_dir, names)
        if args.keep_csv:
            shutil.copy(csv_path, args.keep_csv)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    s = summarise(rows)
    dt = time.monotonic() - t0
    rec = float(np.mean([r["node_recall"] for r in rows]))
    ratio = float(np.nanmean([r["total_node_ratio"] for r in rows]))

    for r in sorted(rows, key=lambda r: r["name"]):
        print(f"  {r['name']:16s} adjJ={r['adj_edge_jaccard']:.4f} "
              f"J={r['edge_jaccard']:.4f} rec={r['node_recall']:.4f} "
              f"n={r['n_pred']:6d} ratio={r['total_node_ratio']:+.2f}")
    print(f"\n=== {s['n']} datasets, SUBMISSION PATH · {dt:.0f}s ===")
    for k in ("score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard"):
        print(f"  {k:18s} {s[k]:.4f}")
    print(f"  {'node_recall':18s} {rec:.4f}")
    print(f"  {'node_count_ratio':18s} {ratio:+.4f}")

    append_experiment(args.experiments, {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(), "machine": "linux-server",
        "run": args.run or "submission_path",
        "detector": "unet" if args.weights else "dog",
        "split": args.split if not args.datasets else "explicit",
        "n_datasets": s["n"], "budget": args.budget,
        "separation_um": "", "gate_um": args.gate_um,
        "division_gate_um": "off" if args.no_divisions else args.division_gate_um,
        "score": f"{s['score']:.4f}", "adj_edge_jaccard": f"{s['adj_edge_jaccard']:.4f}",
        "edge_jaccard": f"{s['edge_jaccard']:.4f}",
        "division_jaccard": f"{s['division_jaccard']:.4f}",
        "node_recall": f"{rec:.4f}", "node_count_ratio": f"{ratio:+.4f}",
        "seconds": f"{dt:.0f}",
        "notes": args.notes or "end-to-end through submit.run + CSV readback",
    })
    print(f"\nappended to {args.experiments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
