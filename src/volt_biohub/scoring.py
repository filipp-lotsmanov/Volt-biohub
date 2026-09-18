"""Score a detector end-to-end with the competition metric.

Exists so the *training loop* and the *standalone scorer* run identical code.
Checkpoint selection previously used ``test_acc x test_recall`` on the edge
matrix, which is worthless: that matrix is overwhelmingly zeros, so predicting
no edges at all scores 0.989–0.999, and the figure *rises* as detection density
grows. It read 0.9943/0.9408 after twenty iterations and froze at epoch 2 of 8
while the detection loss kept improving. Anything selected by it is arbitrary.

The fix is to score epochs with the real thing. That needs whole videos rather
than the training loop's frame-pair windows, so this module owns the video-level
path: read frames, detect, link, assemble, score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tracksdata as td
import zarr

from volt_biohub.detectors import (
    budget_for,
    detect_dog,
    detect_from_map,
    predict_n_true,
)
from volt_biohub.linking import (
    DIVISION_GATE_UM,
    MOTION_GATE_UM,
    build_graph,
    link_sequence,
)
from volt_biohub.metrics import evaluate as score_pair
from volt_biohub.metrics import node_recall, per_sample_metrics, summarise

DOWNSAMPLE = (1, 4, 4)
ISO_UM = 1.625
"""The 1,4,4 grid is cubic: 0.40625 * 4 == 1.625 exactly."""
ORIGINAL_SCALE = (1.625, 0.40625, 0.40625)
"""Micrometres per voxel in the ORIGINAL grid, where the scorer works."""
DEFAULT_SEPARATION_UM = 5.86
"""2 x the measured nucleus sigma. Hard floor is 2.87 um, the closest sister pair."""


def load_volumes(data_dir: Path, name: str) -> np.ndarray:
    """Normalised frames on the isotropic grid, exactly as train and predict do.

    Both of those bypass ``io.open_dataset(normalize=True)`` and normalise from
    the stored 0.001/0.999 quantiles with a lower clamp and no upper clamp. A
    detector scored under a different normalisation than it trained under would
    measure the mismatch, so this must not drift from
    ``train_unet_transformer.py`` or ``predict_unet_transformer.py``.
    """
    g = zarr.open_group(str(data_dir / f"{name}.zarr"), mode="r")
    q = g.attrs["image_statistics"]["quantiles"]
    lo, hi = float(q["0.001"]), float(q["0.999"])
    raw = np.asarray(
        g["0"][:, ::DOWNSAMPLE[0], ::DOWNSAMPLE[1], ::DOWNSAMPLE[2]]
    ).astype(np.float32)
    return np.clip((raw - lo) / (hi - lo + 1e-6), 0, None)


def read_t_true(data_dir: Path, name: str) -> float:
    """``estimated_number_of_nodes`` — the node penalty's denominator.

    Only readable in CV: it lives in the ground-truth geff, and the test split
    ships images alone. Use it for the *scoring* denominator always, and for the
    *detector budget* only when you deliberately want an oracle upper bound —
    see ``score_video(predicted_budget=...)``.
    """
    g = zarr.open_group(str(data_dir / f"{name}.geff"), mode="r")
    return float(g.attrs["geff"]["extra"]["estimated_number_of_nodes"])


def read_q90(data_dir: Path, name: str) -> float:
    """The stored 0.9 intensity quantile — available on train *and* test."""
    g = zarr.open_group(str(data_dir / f"{name}.zarr"), mode="r")
    return float(g.attrs["image_statistics"]["quantiles"]["0.9"])


def load_gt(data_dir: Path, name: str):
    r = td.graph.IndexedRXGraph.from_geff(data_dir / f"{name}.geff")
    return r[0] if isinstance(r, tuple) else r


def detect_video(
    volumes: np.ndarray,
    n_per_frame: int,
    separation_um: float,
    model=None,
    device=None,
    refine: bool = False,
) -> list[np.ndarray]:
    """Peaks per frame, from the U-Net head when *model* is given, else from DoG.

    The learned path peaks the raw **logits**, not a sigmoid. The map is
    saturated — median sigmoid 1.0000 at GT centroids, ~6,000 voxels per frame
    above 0.99 — so a sigmoid flattens exactly the ranking the peak finder needs
    while the logits keep it.
    """
    if model is None:
        return [
            detect_dog(v, n_per_frame, ISO_UM, min_separation_um=separation_um,
                       refine=refine)
            for v in volumes
        ]

    import torch

    out: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for frame in volumes:
                logits = model.detect(torch.from_numpy(frame).to(device))
                out.append(detect_from_map(
                    logits.float().cpu().numpy(), n_per_frame, ISO_UM, separation_um,
                    refine=refine,
                ))
    finally:
        if was_training:
            model.train()
    return out


def safe_node_recall(pred, gt) -> float:
    """``metrics.node_recall``, but 0.0 instead of a crash on an edgeless graph.

    ``metrics.evaluate`` returns before matching when the prediction has no edges,
    so ``MATCHED_NODE_ID`` is never written and ``node_recall`` raises ``KeyError``.
    That is exactly the graph ``submit.run`` emits for a video where detection found
    nothing — its one deliberate degradation path — so an unguarded call discards a
    whole inference run at the last step. The reference harness guards it the same
    way. Guarded here rather than in ``metrics.py``, whose semantics are frozen.
    """
    try:
        return node_recall(pred, gt)
    except KeyError:
        return 0.0


def score_video(
    data_dir: Path,
    name: str,
    *,
    model=None,
    device=None,
    budget: float = 1.0,
    separation_um: float = DEFAULT_SEPARATION_UM,
    gate_um: float = MOTION_GATE_UM,
    division_gate_um: float | None = DIVISION_GATE_UM,
    refine: bool = False,
    predicted_budget: bool = True,
) -> dict:
    """One per-sample metric row, as :func:`metrics.per_sample_metrics` returns.

    A fresh graph is built every call: ``metrics.evaluate`` mutates the predicted
    graph in place and warns-and-resets if handed the same object twice.

    *predicted_budget* decides where the detector's node budget comes from, and it
    defaults to ``True`` because the alternative silently made every CV number in
    this project optimistic. ``estimated_number_of_nodes`` exists only in the
    ground-truth geff; the submission path has to predict it from the stored 0.9
    quantile at 25 % median error. Measured on the four samples whose GT we can
    check, the predictor lands at **0.54x, 1.54x, 1.12x and 0.50x** of truth — two
    of four videos were handed half the node budget they should have had. Scoring
    with the oracle value hid that completely, and CV 0.7366 met LB 0.588.

    Pass ``False`` only to measure the oracle upper bound deliberately.

    Note the asymmetry: the *scoring* denominator below is always the true value,
    because that is what the leaderboard's own scorer uses. Only the budget moves.
    """
    volumes = load_volumes(data_dir, name)
    t_true = read_t_true(data_dir, name)
    budget_base = predict_n_true(read_q90(data_dir, name)) if predicted_budget else t_true
    n_per_frame = budget_for(budget_base, len(volumes), budget)

    coords = detect_video(volumes, n_per_frame, separation_um, model, device, refine)
    links = link_sequence(coords, (ISO_UM, ISO_UM, ISO_UM),
                          gate_um=gate_um, division_gate_um=division_gate_um)
    pred = build_graph(coords, links, upsample=DOWNSAMPLE)
    gt = load_gt(data_dir, name)

    er = score_pair(pred, gt, scale=ORIGINAL_SCALE, max_distance=7.0)
    row = per_sample_metrics(er, t_true, safe_node_recall(pred, gt))
    row["name"] = name
    row["n_pred"] = int(sum(len(c) for c in coords))
    row["n_edges_pred"] = int(sum(len(lk) for lk in links))
    row["t_true"] = t_true
    return row


def score_videos(
    data_dir: Path, names: list[str], **kwargs
) -> tuple[dict, list[dict]]:
    """Score several videos and aggregate exactly as the leaderboard does.

    Returns ``(summary, rows)``. Use ``summarise``'s ``score`` — the weighted
    *adjusted* Jaccard plus 0.1 x division Jaccard — never
    ``evaluate_datasets``, whose micro-average carries no node-count adjustment
    and is not the leaderboard's number.
    """
    rows = [score_video(data_dir, n, **kwargs) for n in names]
    return summarise(rows), rows


def epoch_selection_score(
    data_dir: Path, names: list[str], model, device, **kwargs
) -> tuple[float, dict]:
    """The number to select checkpoints on: the competition score itself.

    Deliberately takes a *subsample* of the fold's test videos. Scoring all 128
    videos of the heavy fold every epoch would cost more than the epoch does,
    and a fixed subsample is a consistent ranking signal even though its
    absolute value is noisier than the full fold's.
    """
    summary, _ = score_videos(data_dir, names, model=model, device=device, **kwargs)
    return float(summary["score"]), summary
