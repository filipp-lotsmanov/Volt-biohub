"""Geometric frame-to-frame linking, and the graph assembly the scorer wants.

Why a purely geometric linker is the default rather than a fallback: measured
over all 199 training samples, the half-distance from a cell to its nearest
*true* neighbour exceeds that cell's per-frame displacement by **4.9x at the
median** (true pitch from ``estimated_number_of_nodes``: 18.45 um median against
1.82 um median motion). The typical case is unambiguous in **199 of 199**
samples and the p99 tail in 157; the samples where it is genuinely ambiguous
hold only ~10% of the metric's edge weight. Assignment on micrometre distance is
therefore close to optimal by geometry, and a learned edge cost has very little
left to buy.

One caveat on that geometric argument, learned the hard way: it holds for the
*typical* case, which is not where the score is lost. Edge precision measured
0.755 for the U-Net against 0.873 for the classical detector, so the atypical
cases carry the remaining loss -- which is what a learned edge cost is for. The
geometric linker is the right default, not the last word.

Both constants below were **swept, not derived**. Deriving them from motion
percentiles alone produced 12.0 um and an enabled division pass, and both were
wrong once measured against the metric -- see each docstring. The lesson
generalises: a parameter read off the data distribution still has to be checked
against the objective, because the objective only grades part of what you emit.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import tracksdata as td
from scipy.optimize import linear_sum_assignment

MOTION_GATE_UM = 8.0
"""Linking gate, swept rather than derived -- 12.0 was wrong for a reason worth knowing.

Ordinary motion says 12 um: p99.9 over all 128,883 GT edges is 13.8 um, so 12 forfeits
0.18% of real edges. That reasoning ignores the metric. An edge is scored only when an
endpoint matched a GT node that has GT edges, so a *loose* gate on a detector with high
node recall manufactures scored false positives. Measured on 30 samples of fold 0, same
detector and budget: 12 um -> 0.5849, **8 um -> 0.6672**, 5 um -> 0.6143, with node recall
identical at 0.8601 throughout. The classical detector prefers 8 um too (0.6435 ->
0.6614), but gains only +0.018 against the U-Net's +0.082 -- the stronger the detector,
the more a loose gate costs.

So this is a per-detector knob. Sweep it whenever the detector changes; do not carry one
detector's value to another.
"""

DIVISION_GATE_UM: float | None = None
"""Second-child gate. **Off by default: measured as a net loss of ~0.025.**

Turning the second pass off raised edge Jaccard 0.6656 -> 0.6914 while giving up 0.0058
of division Jaccard -- worth 0.00058 of score at the 0.1 division weight. It pays 0.026
to buy 0.0006. This is metric fact 2's over-detection/division trade running in reverse:
forcing a second child manufactures edge false positives far faster than it finds real
forks.

Set it to a distance (far-daughter p90 is 10.05 um, max 13.53) to turn divisions back on
-- worth retrying once edge Jaccard plateaus, since 0.1 of the score is unreachable
without it.
"""

_BIG = 1.0e9


def link_frames(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    scale_um: tuple[float, float, float],
    gate_um: float = MOTION_GATE_UM,
    division_gate_um: float | None = DIVISION_GATE_UM,
) -> list[tuple[int, int, float]]:
    """Link detections in frame *a* to detections in frame *b*.

    Parameters
    ----------
    coords_a, coords_b
        ``(N, 3)`` and ``(M, 3)`` integer or float ``(z, y, x)`` coordinates on
        the *same* grid. Physical distance is ``|a - b| * scale_um``, never a
        voxel distance — the raw grid is 4x anisotropic in z, so a voxel-space
        gate would be wrong by 4x on one axis.
    scale_um
        Micrometres per voxel on this grid, ``(z, y, x)``.
    gate_um
        Pairs farther apart than this are never linked.
    division_gate_um
        When not None, a second pass attaches one extra child to a parent that
        already has exactly one, if the orphaned target sits within this
        distance. Pass None to forbid divisions entirely, which is the right
        control arm when measuring what the division term is worth.

    Returns
    -------
    list of ``(index_in_a, index_in_b, distance_um)``
        At most one parent per child (enforced by the assignment), at most two
        children per parent (enforced here). Both are hard scorer requirements:
        out-degree > 2 is silently truncated to the two lowest edge ids, and
        merge duplicates are dropped.
    """
    if len(coords_a) == 0 or len(coords_b) == 0:
        return []

    s = np.asarray(scale_um, dtype=np.float64)
    d = np.linalg.norm(
        (np.asarray(coords_a, dtype=np.float64)[:, None, :]
         - np.asarray(coords_b, dtype=np.float64)[None, :, :]) * s,
        axis=-1,
    )
    cost = np.where(d <= gate_um, d, _BIG)

    rows, cols = linear_sum_assignment(cost)
    links = [(int(i), int(j), float(d[i, j]))
             for i, j in zip(rows, cols, strict=True) if cost[i, j] < _BIG]

    if division_gate_um is None or not links:
        return links

    # Second pass: the far daughter of a division loses the 1-to-1 assignment to
    # the near one, so it arrives here as an orphaned target. Attach it to the
    # nearest parent that still has room, closest pair first so the strongest
    # evidence claims its parent before a weaker one can.
    n_children = {}
    for i, _, _ in links:
        n_children[i] = n_children.get(i, 0) + 1
    claimed = {j for _, j, _ in links}

    cand = [
        (d[i, j], i, j)
        for j in range(len(coords_b)) if j not in claimed
        for i in n_children
        if d[i, j] <= division_gate_um
    ]
    cand.sort()
    for dist, i, j in cand:
        if j in claimed or n_children.get(i, 0) >= 2:
            continue
        links.append((i, j, float(dist)))
        claimed.add(j)
        n_children[i] += 1

    return links


def build_graph(
    coords_per_t: list[np.ndarray],
    links_per_t: list[list[tuple[int, int, float]]],
    upsample: tuple[int, int, int] = (1, 1, 1),
) -> td.graph.InMemoryGraph:
    """Assemble a tracksdata graph the scorer can consume.

    *coords_per_t* is one array per timepoint, in whatever downsampled grid the
    detector ran on; *upsample* multiplies back to **original anisotropic voxel
    space**, which is the coordinate system both the scorer and the submission
    format require. Getting this wrong is the highest-risk mistake available
    here — off by 4x in z, no exception raised, score near zero.

    ``add_node_attr_key`` is called explicitly for z/y/x to sidestep a
    tracksdata/Polars issue where a float default is mistaken for a dtype.
    """
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)

    up = np.asarray(upsample, dtype=np.float64)
    ids_per_t: list[list[int]] = []
    for t, coords in enumerate(coords_per_t):
        scaled = np.asarray(coords, dtype=np.float64) * up
        ids_per_t.append(g.bulk_add_nodes([
            {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
            for z, y, x in scaled
        ]))

    g.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
    rows = [
        {"source_id": ids_per_t[t][i], "target_id": ids_per_t[t + 1][j],
         "edge_dist": dist}
        for t, links in enumerate(links_per_t)
        for i, j, dist in links
    ]
    if rows:
        g.bulk_add_edges(rows)
    return g


def link_sequence(
    coords_per_t: list[np.ndarray],
    scale_um: tuple[float, float, float],
    gate_um: float = MOTION_GATE_UM,
    division_gate_um: float | None = DIVISION_GATE_UM,
) -> list[list[tuple[int, int, float]]]:
    """Run :func:`link_frames` across every consecutive pair.

    Only ``t -> t+1`` edges are produced. The scorer deletes any edge spanning
    more than one timepoint without warning, so gap closing must be expressed by
    *inserting an interpolated node*, never as a longer edge.
    """
    return [
        link_frames(coords_per_t[t], coords_per_t[t + 1], scale_um,
                    gate_um=gate_um, division_gate_um=division_gate_um)
        for t in range(len(coords_per_t) - 1)
    ]
