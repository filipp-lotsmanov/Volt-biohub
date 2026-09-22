"""The tree DP must agree with the ILP it replaces, edge for edge.

That is the only claim worth testing about ``prune_dp``: it exists to avoid
putting ``tracksdata`` (62 wheels, numpy>2, scikit-image, dask) on the
inference path, and it is only allowed to do that if it is exact. These build
random forests -- which is what greedy resolution always produces, since a node
can have at most one parent -- and compare the two edge sets.
"""
from __future__ import annotations

import numpy as np
import pytest

from volt_biohub.ilp_link import prune_dp

td = pytest.importorskip("tracksdata")
from volt_biohub.ilp_link import prune  # noqa: E402

ISO = 1.625


def random_forest(rng, n_frames=6, per_frame=5, p_link=0.7):
    coords = [rng.random((per_frame, 3)) * 40 for _ in range(n_frames)]
    probs, links = [], []
    for _t in range(n_frames - 1):
        m = rng.random((per_frame, per_frame)).astype(np.float32)
        probs.append(m)
        used_t: set[int] = set()
        lk = []
        for i in range(per_frame):
            for j in rng.permutation(per_frame):
                if j in used_t or rng.random() > p_link:
                    continue
                lk.append((i, int(j), 0.0))
                used_t.add(int(j))
                break
        links.append(lk)
    return coords, links, probs


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("weights", [
    {},                                        # the reference costs
    {"appearance_weight": 1.0},                # the term the DP nearly got wrong
    {"disappearance_weight": 3.0},             # the retune their ablation prices
    {"division_weight": 0.4, "appearance_weight": 0.5},
])
def test_dp_matches_the_solver(seed, weights):
    rng = np.random.default_rng(seed)
    coords, links, probs = random_forest(rng)
    _, a = prune(coords, links, probs, ISO, **weights)
    _, b = prune_dp(coords, links, probs, ISO, **weights)
    got = {(t, i, j) for t, lk in enumerate(a) for i, j, _ in lk}
    exp = {(t, i, j) for t, lk in enumerate(b) for i, j, _ in lk}
    assert got == exp, f"solver {len(got)} edges, DP {len(exp)}, symmetric diff {got ^ exp}"


def test_dp_keeps_a_confident_long_track_and_drops_a_lone_pair():
    """The economics in one fixture: ending a track costs 2.0, an edge earns at
    most 1.0, so a two-node fragment cannot pay for itself and a long one can."""
    coords = [np.array([[0.0, 0.0, float(t)], [0.0, 50.0, 0.0]]) for t in range(6)]
    links, probs = [], []
    for _t in range(5):
        m = np.zeros((2, 2), dtype=np.float32)
        m[0, 0] = 0.99          # the long track
        m[1, 1] = 0.99
        probs.append(m)
        links.append([(0, 0, 0.0)] + ([(1, 1, 0.0)] if _t == 0 else []))
    _, kept = prune_dp(coords, links, probs, ISO)
    flat = {(t, i, j) for t, lk in enumerate(kept) for i, j, _ in lk}
    assert all((t, 0, 0) in flat for t in range(5)), "the five-edge track should survive"
    assert (0, 1, 1) not in flat, "a single edge earns 0.99 against a 2.0 disappearance"
