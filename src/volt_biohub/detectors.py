"""Interchangeable cell detectors, so classical and learned paths score alike.

Every detector here takes one normalised volume on the isotropic training grid
and returns ``(N, 3)`` integer ``(z, y, x)`` peaks, ordered strongest first.
Nothing downstream — linking, graph assembly, scoring — knows which one ran.
That is the point: the only honest way to ask whether a learned detector beats
a classical one is to hold the rest of the pipeline fixed, and the learned
linker cannot serve as that fixture while it emits zero edges.

Both detectors take a *node budget* rather than a probability threshold. The
metric's node-count penalty responds to how many nodes you emit, not to how
confident you were, and a budget is directly comparable across detectors whose
score distributions are nothing alike — the U-Net's map is saturated (median
sigmoid 1.0000 at GT centroids, ~6,000 voxels/frame above 0.99), so a shared
threshold would mean two different things.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

NUCLEUS_FWHM_UM = 6.9
"""Measured from radial intensity profiles around 208 GT centroids."""

DEFAULT_SIGMA_UM = NUCLEUS_FWHM_UM / 2.3548
"""~2.93 um. Matches detection_targets.DEFAULT_SIGMA_UM by construction."""

DOG_RATIO = 1.6
"""Second Gaussian's width. 1.6 approximates the Laplacian of Gaussian."""


def _peak_window(min_separation_um: float, iso_um: float) -> int:
    """Odd ``maximum_filter`` size enforcing at least *min_separation_um*.

    A voxel survives ``maximum_filter(size=k)`` only if it is the maximum of a
    ``(k, k, k)`` box centred on it, so two survivors are at least
    ``k // 2 + 1`` voxels apart along any axis. That filter *is* a suppression
    radius. Missing this is how a suppression sweep silently measures nothing:
    with sigma 2.93 um on the 1.625 um grid the window is already 5 voxels, so
    an explicit NMS below 4.88 um can never bind, and a sweep over 2.5–4.0 um
    returns a dead-flat score.
    """
    need_vox = max(1, math.ceil(min_separation_um / iso_um))
    return max(3, (2 * need_vox - 1) | 1)


def detect_dog(
    volume: np.ndarray,
    n_keep: int,
    iso_um: float,
    sigma_um: float = DEFAULT_SIGMA_UM,
    min_separation_um: float | None = None,
    refine: bool = False,
) -> np.ndarray:
    """Difference-of-Gaussians blob detection, local maxima, top *n_keep*.

    Parameter-poor on purpose: sigma comes from the measured nucleus size and
    the separation from the division sister-separation floor (min 2.87 um over
    all 151 events), so nothing here is fitted to the data being scored.
    """
    if min_separation_um is None:
        min_separation_um = 2.0 * sigma_um
    s = sigma_um / iso_um
    dog = gaussian_filter(volume, s) - gaussian_filter(volume, s * DOG_RATIO)
    peaks = _top_peaks(dog, n_keep, _peak_window(min_separation_um, iso_um),
                       positive_only=True)
    return refine_centres(dog, peaks) if refine else peaks


def detect_from_map(
    score_map: np.ndarray,
    n_keep: int,
    iso_um: float,
    min_separation_um: float,
    refine: bool = False,
    threshold: float | None = None,
) -> np.ndarray:
    """Peaks of an arbitrary score map — the learned detector's head output.

    Takes the map rather than the model so the caller owns batching and device
    placement, and so a cached map can be re-peaked at several budgets without
    re-running the network.

    **Pass *threshold* to escape the node budget, and read this before choosing.**
    With *threshold* None the map is truncated to the top *n_keep* peaks, where
    ``n_keep`` comes from ``predict_n_true``. That makes detection *purely*
    budget-limited: the model decides which nodes, never how many. Measured on the
    four test samples, emitted-vs-asked was 13900/13851, 50600/50641, 7100/7105 and
    35200/35168 — **exactly the budget every time** — while the predictor itself
    landed at 0.54x, 1.54x, 1.12x and 0.50x of the true count. Two of four videos
    were therefore capped near half the node recall they could have reached, and no
    detector improvement can lift that cap.

    It is worse on the hidden test set than in CV, because the predictor is a
    log-linear fit on the *two* training embryos applied to an unseen third, and
    nothing on that set can check it. This is the most likely reason three
    submissions scored 0.588 / 0.578 / 0.581 while the detector under them changed
    from a 2,400-iteration model to a 400-epoch one: the clamp sits downstream of
    the thing we kept changing.

    A threshold removes ``N_true`` from the inference path entirely, which is what
    the reference implementation does (sigmoid > 0.5 with a 3-voxel window); its
    node counts then land at 0.94–1.53x of true by themselves.
    """
    window = _peak_window(min_separation_um, iso_um)
    if threshold is not None:
        pooled = maximum_filter(score_map, size=window, mode="nearest")
        # Sigmoid on the logits, matching the reference. Comparing the threshold in
        # probability space rather than logit space keeps it transferable between
        # checkpoints whose logit ranges differ.
        prob = 1.0 / (1.0 + np.exp(-score_map.astype(np.float64)))
        idx = np.argwhere((score_map >= pooled) & (prob > threshold))
        if len(idx) == 0:
            return np.empty((0, 3), dtype=np.int64)
        peaks = idx.astype(np.int64)
        return refine_centres(score_map, peaks) if refine else peaks

    peaks = _top_peaks(score_map, n_keep, window, positive_only=False)
    return refine_centres(score_map, peaks) if refine else peaks


def refine_centres(
    field: np.ndarray, peaks: np.ndarray, radius: int = 1
) -> np.ndarray:
    """Response-weighted centroid in a small box around each peak.

    CLAUDE.md rules out sub-voxel localisation because the 7 um match gate is
    ~4 z-voxels wide, and that is right *for node matching*. It is wrong for
    **linking**, which is a different job: the linker resolves a median motion
    of 1.82 um, so a one-voxel wobble (1.625 um on this grid) is nearly a whole
    frame of apparent movement. Measured on four videos, the U-Net's link
    displacements sit at a median 2.30 um against 1.62 um for DoG and 1.82 um
    for ground truth, and 17.2% of its links exceed the GT motion p99 versus
    10.5% for DoG -- because a saturated map puts each cell on a *plateau* and
    which voxel of it wins changes frame to frame.

    Refining in the downsampled grid is cheap precision: the output is scaled
    back by 4 in y and x, so +-0.25 voxel here is +-1 voxel in the original
    space the submission wants, and rounding stays clean.
    """
    if len(peaks) == 0:
        return peaks.astype(np.float64)
    shape = np.asarray(field.shape)
    offs = np.stack(np.meshgrid(
        *[np.arange(-radius, radius + 1)] * 3, indexing="ij"
    ), -1).reshape(-1, 3)
    pos = peaks[:, None, :] + offs[None, :, :]
    inside = ((pos >= 0) & (pos < shape)).all(-1)
    clipped = np.clip(pos, 0, shape - 1)
    w = field[clipped[..., 0], clipped[..., 1], clipped[..., 2]].astype(np.float64)
    # Only positive response may pull the centroid; a negative lobe would push
    # the centre away from the cell rather than toward it.
    w = np.where(inside, np.maximum(w, 0.0), 0.0)
    tot = w.sum(1, keepdims=True)
    flat = tot[:, 0] <= 0
    out = (w[..., None] * clipped).sum(1) / np.where(tot == 0, 1.0, tot)
    out[flat] = peaks[flat]
    return out


def _top_peaks(
    field: np.ndarray, n_keep: int, window: int, *, positive_only: bool
) -> np.ndarray:
    """Local maxima of *field*, strongest first, truncated to *n_keep*.

    Ties matter here and are easy to get wrong. A saturated map has broad
    plateaus where every voxel equals the window maximum, so ``field == pooled``
    marks the whole plateau. Ranking by value alone then returns an arbitrary
    contiguous blob rather than one peak per cell, which is exactly how a
    confident detector loses node recall. Breaking ties by a fixed jitter keeps
    the choice deterministic and spreads it over the plateau instead of taking
    its first voxels in raster order.
    """
    if n_keep <= 0:
        return np.empty((0, 3), dtype=np.int64)

    pooled = maximum_filter(field, size=window, mode="nearest")
    is_peak = field >= pooled
    if positive_only:
        is_peak &= field > 0
    idx = np.argwhere(is_peak)
    if len(idx) == 0:
        return np.empty((0, 3), dtype=np.int64)

    vals = field[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
    if len(idx) > n_keep:
        rng = np.random.default_rng(0)
        vals = vals + rng.uniform(0, 1e-9, size=len(vals))
    order = np.argsort(-vals)[:n_keep]
    return idx[order].astype(np.int64)


N_TRUE_LOG_SLOPE = 0.79920777
N_TRUE_LOG_INTERCEPT = 4.57252748
"""``log(N_true) = 0.7992*log(q0.9 + 1) + 4.5725``, least squares over all 199
training samples. R^2 0.722 in log space, 25.3 % median relative error. Refit with
``scripts/check_t_true.py``."""


def predict_n_true(q90: float) -> float:
    """Estimate ``estimated_number_of_nodes`` from the stored 0.9 quantile.

    **This lives here, not in `kaggle/submit.py`, because having two copies of the
    node budget was a real and expensive bug.** The submission path predicted
    `N_true` while `scoring.score_video` read the true value out of the ground-truth
    geff, so every CV number in `experiments.csv` was measured with information the
    leaderboard run does not have. Measured on the four samples whose GT we can
    check, the predictor lands at **0.54x, 1.54x, 1.12x and 0.50x** of truth — two
    of four videos were given half the node budget they should have had, which the
    CV could not see. One definition, one import, both callers.

    It is a weak per-sample predictor, but the node penalty is soft in both
    directions: this fit is worth an edge-weighted multiplier of 0.9927 against
    0.936 for the best global constant.
    """
    return float(math.exp(N_TRUE_LOG_SLOPE * math.log(q90 + 1.0) + N_TRUE_LOG_INTERCEPT))


def budget_for(t_true: float, n_frames: int, multiplier: float = 1.0) -> int:
    """Detections per frame for a target total of ``multiplier * t_true``.

    ``t_true`` is ``estimated_number_of_nodes``, the denominator of the node
    penalty. It is absent from the test split but predictable from the stored
    intensity quantiles (Spearman 0.856 on q0.9; a p90-only log-linear fit
    reaches R^2 0.72 and a 0.993 weighted multiplier against 0.936 for the best
    global constant), so the budget stays available at inference.

    Measured response on 14 samples with the classical detector: 0.5x scores
    0.560, 0.75x 0.642, 1.0x 0.640, 1.5x 0.615, 2.0x 0.588 — an interior
    optimum near 0.75–1.0x, and edge Jaccard flat from 1.0x upward, so past
    that point extra detections are pure penalty with no recall to show for it.
    """
    return max(1, round(multiplier * t_true / max(n_frames, 1)))
