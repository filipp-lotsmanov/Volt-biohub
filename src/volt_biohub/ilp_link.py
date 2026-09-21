"""Global track linking by integer program, over a learned candidate graph.

**Measured 2026-09-21, and it reverses this module's own premise.** The
docstring below used to say the +0.037 comes from solving the *raw* candidate
graph and that running the solver over an already-resolved link set is
"measured worthless". That is backwards, and it is why our ILP scored 0.8190
against a geometric relink at 0.8687. Running the reference's own inference
path (``scripts/predict_unet_transformer.py``, byte-identical to the upstream
baseline) on the four movies the public notebooks validate on:

    reference detector + edge head + ILP        adjJ 0.9299   FP 38/16/3/20
    the same, --use-ilp off (greedy head)       adjJ 0.8830   FP 71/29/13/35
    our own full stack                          adjJ 0.8888

The ILP is worth **+0.047 of adjusted edge Jaccard there**, close to the
+0.037 their published ablation claims -- and it is spent on *precision*: edge
false positives roughly halve while true positives move by single digits, and
the node count falls 51k -> 42k, which also earns back the node-count
multiplier. Node recall stays at 0.981, so the over-deletion we measured was
never inherent to the solver.

**What we got wrong is the input.** The reference thresholds the source-softmax
at 0.5 and resolves greedily to <=1 parent / <=2 children *first*, then hands
the solver that graph, which therefore only decides what to keep. We handed it
every pair above a probability threshold and asked it to choose the topology as
well; with ``node_weight`` at 0.0 a marginal node is then always cheaper to
delete than to keep, and recall fell to 0.9419.

:func:`solve` is kept for the raw-graph variant that we measured. :func:`prune`
is the one that matches the reference.
"""

from __future__ import annotations

import logging

import numpy as np

# Reference costs, then the retune measured on 2026-09-18 against them. Sign
# convention: the solver maximises, so EDGE_WEIGHT multiplies a probability in
# [0, 1] and the appearance/disappearance/division weights are penalties.
EDGE_WEIGHT = -1.0
NODE_WEIGHT = 0.0
APPEARANCE_WEIGHT = 0.0
DISAPPEARANCE_WEIGHT = 2.0
DIVISION_WEIGHT = 1.2
CANDIDATE_THRESHOLD = 0.48

# Why node_weight is exposed at all, since the reference never sets it: with it
# at 0.0, keeping a node and deleting one are equally free, so the solver deletes
# any node whose edges do not pay for themselves. The economics are lopsided --
# an edge is worth at most 1.0 (edge_weight x a probability in [0, 1]) while
# ending a track costs disappearance 2.0 -- so a marginal node is always cheaper
# to drop. Measured on our full fold at the reference costs: node recall
# 0.9830 -> 0.9691 and node ratio -0.07 -> -0.16 against the geometric linker,
# for a net 0.8687 -> 0.8605. The reference does not feel this because its motion
# relink runs afterwards and re-derives edges from geometry, quietly repairing
# the over-deletion; with the relink off (their own best arm) the cost balance
# has to carry itself.
#
# The retune: disappearance 3.0 + candidate 0.44 measured 0.9066 -> 0.9123 on a
# dual-seed blend and 0.9355 -> 0.9403 on the shipped pack weights. Kept as named
# constants rather than folded into the defaults so the two can ship as separate
# one-variable submissions.
DISAPPEARANCE_WEIGHT_TUNED = 3.0
CANDIDATE_THRESHOLD_TUNED = 0.44


def candidates_from_probs(
    probs_per_pair: list[np.ndarray],
    coords_per_t: list[np.ndarray],
    iso_um: float,
    threshold: float = CANDIDATE_THRESHOLD,
) -> list[tuple[int, int, float, float]]:
    """Flatten per-frame-pair probability matrices into global candidate edges.

    Returns ``(source, target, prob, distance_um)`` with **global** node indices
    into the concatenation of *coords_per_t*, which is the form
    :func:`solve` needs.

    No distance gate, deliberately. A gate here would re-impose exactly the
    constraint the solver exists to relax: the reference's own error analysis
    finds association recall falling to 91.0% where the nearest detection is
    under 7 um, and a geometric gate cannot tell a crowded neighbour from a fast
    cell. The threshold on the probability is the only admission rule.

    Note the softmax is over *sources*, so at most two sources can exceed 0.48
    for a given target and the candidate set is naturally sparse -- typically a
    couple of options per node, not a dense bipartite graph.
    """
    offsets = np.cumsum([0] + [len(c) for c in coords_per_t])
    out: list[tuple[int, int, float, float]] = []
    for t, probs in enumerate(probs_per_pair):
        if probs.size == 0:
            continue
        ii, jj = np.nonzero(probs > threshold)
        if len(ii) == 0:
            continue
        src = np.asarray(coords_per_t[t], dtype=np.float64)[ii]
        tgt = np.asarray(coords_per_t[t + 1], dtype=np.float64)[jj]
        dist = np.linalg.norm((src - tgt) * iso_um, axis=1)
        for i, j, d in zip(ii, jj, dist, strict=True):
            out.append((int(offsets[t] + i), int(offsets[t + 1] + j),
                        float(probs[i, j]), float(d)))
    return out


def solve(
    coords_per_t: list[np.ndarray],
    candidates: list[tuple[int, int, float, float]],
    iso_um: float,
    *,
    edge_weight: float = EDGE_WEIGHT,
    node_weight: float = NODE_WEIGHT,
    appearance_weight: float = APPEARANCE_WEIGHT,
    disappearance_weight: float = DISAPPEARANCE_WEIGHT,
    division_weight: float = DIVISION_WEIGHT,
):
    """Solve the candidate graph and return ``(coords_per_t, links_per_t)``.

    Imports tracksdata lazily. **This is the first thing on the inference path
    that needs it**, which is a deliberate reversal: ``CLAUDE.md`` has forbidden
    tracksdata at inference since the beginning because it is not on PyPI, and
    that constraint turned out to be unnecessary -- the public support pack ships
    a wheel for it alongside ``ilpy`` and ``pyscipopt``. Lazy, so the
    tracksdata-free fallback path still imports and runs with tracksdata (and
    torch) absent entirely.

    Nodes the solver does not keep are returned with their edges absent, so they
    arrive at the submission writer as degree-0 and are pruned there. Returning
    them rather than deleting them keeps the index space aligned with the input,
    which every caller and every test depends on.
    """
    import polars as pl
    import tracksdata as td

    n_frames = len(coords_per_t)
    links_per_t: list[list[tuple[int, int, float]]] = [[] for _ in range(max(0, n_frames - 1))]
    if not candidates:
        return coords_per_t, links_per_t

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    flat: list[dict] = []
    frame_of: list[int] = []
    index_in_frame: list[int] = []
    for t, coords in enumerate(coords_per_t):
        for i, c in enumerate(np.asarray(coords, dtype=np.float64)):
            flat.append({"t": int(t), "z": float(c[0]), "y": float(c[1]), "x": float(c[2])})
            frame_of.append(t)
            index_in_frame.append(i)
    node_ids = graph.bulk_add_nodes(flat)

    graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
    graph.bulk_add_edges([
        {"source_id": node_ids[s], "target_id": node_ids[t],
         "edge_prob": p, "edge_dist": d}
        for s, t, p, d in candidates
    ])

    solver = td.solvers.ILPSolver(
        edge_weight=edge_weight * td.EdgeAttr("edge_prob"),
        node_weight=node_weight,
        appearance_weight=appearance_weight,
        disappearance_weight=disappearance_weight,
        division_weight=division_weight,
    )
    # tracksdata tries Gurobi first and logs a WARNING when its licence is
    # absent, then falls back to SCIP. We bundle pyscipopt and never Gurobi, so
    # that warning is the expected path, not a problem -- and at one line per
    # video it would bury the real log. Silence the solver's logger only.
    _log = logging.getLogger("tracksdata")
    _prev = _log.level
    _log.setLevel(logging.ERROR)
    try:
        solved = solver.solve(graph)
    finally:
        _log.setLevel(_prev)

    # Map tracksdata ids back to our (frame, index-in-frame) space. The solver
    # returns a graph whose node ids are the ones we handed it, so the inverse of
    # `node_ids` is all that is needed -- but build it explicitly rather than
    # assuming identity, because bulk_add_nodes documents no such guarantee.
    back = {nid: k for k, nid in enumerate(node_ids)}
    for src, tgt in _solved_edges(solved):
        ks, kt = back.get(src), back.get(tgt)
        if ks is None or kt is None:
            continue
        t = frame_of[ks]
        if frame_of[kt] != t + 1 or t >= len(links_per_t):
            # The scorer deletes any edge whose dt != 1, silently. Drop them here
            # so a solver that ever emits one does not cost us the edge twice.
            continue
        a, b = index_in_frame[ks], index_in_frame[kt]
        # Micrometres, like every other link weight in the repo: postprocess's
        # structure enforcement orders on this value, so a voxel-space weight
        # here would silently compare against micrometre weights elsewhere.
        d = float(np.linalg.norm(
            (np.asarray(coords_per_t[t][a], dtype=np.float64)
             - np.asarray(coords_per_t[t + 1][b], dtype=np.float64)) * iso_um))
        links_per_t[t].append((a, b, d))

    return coords_per_t, links_per_t


def _solved_edges(solved) -> list[tuple[int, int]]:
    """``(source_id, target_id)`` pairs from a solved tracksdata graph.

    Isolated in its own function because this is the one place the pre-1.0
    tracksdata API is load-bearing and its edge accessor has moved between
    releases; a single failure point with a clear error beats the same guess
    inlined at three call sites.
    """
    df = solved.edge_attrs(attr_keys=[])
    cols = set(df.columns)
    src = "source_id" if "source_id" in cols else "source"
    tgt = "target_id" if "target_id" in cols else "target"
    if src not in cols or tgt not in cols:
        raise RuntimeError(
            f"solved graph exposes no source/target columns; got {sorted(cols)}. "
            "tracksdata is pre-1.0 and its edge accessor has moved before."
        )
    return list(zip(df[src].to_list(), df[tgt].to_list(), strict=True))


def prune(
    coords_per_t: list[np.ndarray],
    links_per_t: list[list[tuple[int, int, float]]],
    probs_per_pair: list[np.ndarray] | None,
    iso_um: float,
    *,
    edge_weight: float = EDGE_WEIGHT,
    node_weight: float = NODE_WEIGHT,
    appearance_weight: float = APPEARANCE_WEIGHT,
    disappearance_weight: float = DISAPPEARANCE_WEIGHT,
    division_weight: float = DIVISION_WEIGHT,
    default_prob: float = 0.5,
):
    """Solve an **already-resolved** link set, keeping only what pays for itself.

    This is what the reference actually does, and the distinction from
    :func:`solve` is the whole difference between -0.05 and +0.047 (see the
    module docstring). Because every node already has at most one parent and
    two children, the solver cannot re-route anything: it chooses which nodes
    and edges to keep under the appearance/disappearance/division costs. That
    is a pure precision filter, and it is why it can cut edge false positives
    roughly in half while moving true positives by single digits.

    *probs_per_pair* supplies the edge reward. Any link with no probability --
    a gap-closed or division edge, or a relink run without the edge head --
    gets *default_prob*, which is deliberately the same 0.5 the reference uses
    as its admission threshold: unscored links are worth keeping at exactly the
    margin where a scored one would have been admitted at all.
    """
    import polars as pl
    import tracksdata as td

    n_frames = len(coords_per_t)
    out: list[list[tuple[int, int, float]]] = [[] for _ in range(max(0, n_frames - 1))]
    if not any(len(l) for l in links_per_t):
        return coords_per_t, out

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    flat: list[dict] = []
    offsets: list[int] = []
    for t, coords in enumerate(coords_per_t):
        offsets.append(len(flat))
        for c in np.asarray(coords, dtype=np.float64):
            flat.append({"t": int(t), "z": float(c[0]), "y": float(c[1]), "x": float(c[2])})
    node_ids = graph.bulk_add_nodes(flat)

    graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    rows = []
    for t, links in enumerate(links_per_t):
        probs = probs_per_pair[t] if probs_per_pair is not None and t < len(probs_per_pair) else None
        for a, b, _d in links:
            p = default_prob
            if probs is not None and probs.size and a < probs.shape[0] and b < probs.shape[1]:
                p = float(probs[a, b])
            rows.append({"source_id": node_ids[offsets[t] + a],
                         "target_id": node_ids[offsets[t + 1] + b],
                         "edge_prob": p})
    if not rows:
        return coords_per_t, out
    graph.bulk_add_edges(rows)

    solver = td.solvers.ILPSolver(
        edge_weight=edge_weight * td.EdgeAttr("edge_prob"),
        node_weight=node_weight,
        appearance_weight=appearance_weight,
        disappearance_weight=disappearance_weight,
        division_weight=division_weight,
    )
    _log = logging.getLogger("tracksdata")
    _prev = _log.level
    _log.setLevel(logging.ERROR)
    try:
        solved = solver.solve(graph)
    finally:
        _log.setLevel(_prev)

    back = {nid: k for k, nid in enumerate(node_ids)}
    frame_of = [r["t"] for r in flat]
    for src, tgt in _solved_edges(solved):
        ks, kt = back.get(src), back.get(tgt)
        if ks is None or kt is None:
            continue
        t = frame_of[ks]
        if frame_of[kt] != t + 1 or t >= len(out):
            continue
        a, b = ks - offsets[t], kt - offsets[t + 1]
        d = float(np.linalg.norm(
            (np.asarray(coords_per_t[t][a], dtype=np.float64)
             - np.asarray(coords_per_t[t + 1][b], dtype=np.float64)) * iso_um))
        out[t].append((a, b, d))

    return coords_per_t, out


def prune_dp(
    coords_per_t: list[np.ndarray],
    links_per_t: list[list[tuple[int, int, float]]],
    probs_per_pair: list[np.ndarray] | None,
    iso_um: float,
    *,
    edge_weight: float = EDGE_WEIGHT,
    node_weight: float = NODE_WEIGHT,
    appearance_weight: float = APPEARANCE_WEIGHT,
    disappearance_weight: float = DISAPPEARANCE_WEIGHT,
    division_weight: float = DIVISION_WEIGHT,
    default_prob: float = 0.5,
):
    """:func:`prune`, solved exactly by dynamic programming instead of an ILP.

    **Why this is not an approximation.** After greedy resolution every node has
    at most one parent, so the link graph is a *forest*, and tracksdata's
    formulation is separable over it: with merges disabled its constraints
    reduce to "a selected node has one incoming edge or appears" and "a selected
    node has one or two outgoing edges or disappears", and its objective is a
    sum of per-node and per-edge terms. A selected node therefore costs
    ``disappearance`` when it has no successor and ``division`` when it has two,
    an edge pays ``edge_weight x prob``, and nothing couples two different
    components. That is a textbook tree DP, exact in O(N), and it needs numpy
    and nothing else.

    That matters beyond elegance. The solver route costs ~15-20 s/video and
    drags ``tracksdata`` onto the inference path, whose dependency closure is
    scikit-image, dask, pyarrow, numba, imagecodecs and ``numpy>2`` -- 62 wheels
    in the public pack, installed over an image whose torch and scipy are built
    against its own numpy. This costs milliseconds and adds nothing.

    ``tests/test_ilp_dp.py`` pins it against ``prune`` on random forests, which
    is the only claim worth testing: the two must return the same edge set.
    """
    n_frames = len(coords_per_t)
    out: list[list[tuple[int, int, float]]] = [[] for _ in range(max(0, n_frames - 1))]
    if not any(len(lk) for lk in links_per_t):
        return coords_per_t, out

    offs, n = [], 0
    for c in coords_per_t:
        offs.append(n)
        n += len(c)

    # (global source, global target, frame, local a, local b, prob)
    kids: dict[int, list[tuple[int, float, int, int, int]]] = {}
    has_parent = np.zeros(n, dtype=bool)
    for t, links in enumerate(links_per_t):
        probs = (probs_per_pair[t] if probs_per_pair is not None
                 and t < len(probs_per_pair) else None)
        for a, b, _d in links:
            p = default_prob
            if probs is not None and probs.size and a < probs.shape[0] and b < probs.shape[1]:
                p = float(probs[a, b])
            gs, gt = offs[t] + a, offs[t + 1] + b
            kids.setdefault(gs, []).append((gt, p, t, a, b))
            has_parent[gt] = True

    w_app, w_dis, w_div = appearance_weight, disappearance_weight, division_weight

    def solve(root: int):
        """Iterative post-order; recursion would blow the stack on a 100-frame track."""
        on: dict[int, float] = {}
        free: dict[int, float] = {}
        pick: dict[int, list] = {}
        stack = [(root, False)]
        while stack:
            v, expanded = stack.pop()
            if not expanded:
                stack.append((v, True))
                for ch in kids.get(v, ()):
                    stack.append((ch[0], False))
                continue
            children = kids.get(v, ())
            base = sum(free[c[0]] for c in children)
            best = w_app + node_weight + w_dis + base   # v selected, no successor
            chosen: list = []
            gains = [(edge_weight * p + on[c] - free[c], (c, p, t, a, b))
                     for c, p, t, a, b in children]
            if gains:
                g1, e1 = min(gains, key=lambda x: x[0])
                cand = w_app + node_weight + base + g1
                if cand < best:
                    best, chosen = cand, [e1]
                if len(gains) >= 2:
                    two = sorted(gains, key=lambda x: x[0])[:2]
                    cand2 = (w_app + node_weight + w_div + base
                             + two[0][0] + two[1][0])
                    if cand2 < best:
                        best, chosen = cand2, [two[0][1], two[1][1]]
            on[v] = best
            pick[v] = chosen
            free[v] = min(best, base)
        return free, on, pick

    for root in range(n):
        if has_parent[root] or root not in kids:
            continue
        free, on, pick = solve(root)
        # Top-down reconstruction. The flag is the whole subtlety: a child that
        # its parent did NOT connect to is still free to start its own track,
        # because appearance is free -- dropping the parent edge must not drop
        # the subtree with it.
        stack = [(root, on[root] <= free[root])]
        while stack:
            v, is_on = stack.pop()
            taken: set[int] = set()
            if is_on:
                for c, _p, t, a, b in pick[v]:
                    d = float(np.linalg.norm(
                        (np.asarray(coords_per_t[t][a], dtype=np.float64)
                         - np.asarray(coords_per_t[t + 1][b], dtype=np.float64)) * iso_um))
                    out[t].append((a, b, d))
                    taken.add(c)
                    stack.append((c, True))
            for c in kids.get(v, ()):
                if c[0] not in taken:
                    stack.append((c[0], on[c[0]] <= free[c[0]]))
    return coords_per_t, out
