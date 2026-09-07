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
    return _top_peaks(dog, n_keep, _peak_window(min_separation_um, iso_um), positive_only=True)


def detect_from_map(
    score_map: np.ndarray,
    n_keep: int,
    iso_um: float,
    min_separation_um: float,
) -> np.ndarray:
    """Peaks of an arbitrary score map — the learned detector's head output.

    Takes the map rather than the model so the caller owns batching and device
    placement, and so a cached map can be re-peaked at several budgets without
    re-running the network.
    """
    return _top_peaks(score_map, n_keep, _peak_window(min_separation_um, iso_um),
                      positive_only=False)


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
