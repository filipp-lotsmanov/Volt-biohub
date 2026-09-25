"""Invariants for the two things added for the combined submission.

The TTA test is worth more than it looks. A symmetry average is *exactly*
equivariant for any model whatsoever -- averaging ``g^-1 f(g x)`` over a whole
group commutes with the group -- so the identity below holds regardless of what
the network has learned, and it fails the moment an inverse transform is wrong,
missing, or applied in the wrong order. A "does it run" test would pass with
the inverses omitted, which is precisely the bug that silently destroys a
detection map by averaging misaligned views.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle"))

torch = pytest.importorskip("torch")

_CONFIG = {
    "unet_layers": [8, 16],
    "unet_out_channels": 8,
    "downsample": [1, 4, 4],
}


def _checkpoint(tmp_path: Path) -> Path:
    """A tiny randomly-initialised detector, saved the way the real one is."""
    from volt_biohub.models import TemporalUNet3D

    torch.manual_seed(0)
    unet = TemporalUNet3D(in_channels=1, out_channels=_CONFIG["unet_out_channels"],
                          layers=_CONFIG["unet_layers"])
    head = torch.nn.Conv3d(_CONFIG["unet_out_channels"], 1, kernel_size=1)
    state = {f"unet.{k}": v for k, v in unet.state_dict().items()}
    state.update({f"detect_head.{k}": v for k, v in head.state_dict().items()})
    d = tmp_path / "w"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(_CONFIG))
    path = d / "edge_predictor_best.pth"
    torch.save(state, path)
    return path


def test_tta_average_is_exactly_d4_equivariant(tmp_path: Path) -> None:
    """Rotating the input must rotate the TTA map, for an arbitrary model.

    This is the property that pins the inverse transforms. Without them the
    average is over misaligned views and this identity breaks badly.
    """
    import submit

    detect = submit.load_detector(_checkpoint(tmp_path), torch.device("cpu"), tta=True)
    rng = np.random.default_rng(0)
    frame = rng.random((16, 16, 16), dtype=np.float32)

    plain = detect(frame)
    rotated = detect(np.rot90(frame, 1, axes=(-2, -1)).copy())

    np.testing.assert_allclose(rotated, np.rot90(plain, 1, axes=(-2, -1)),
                               atol=1e-5)


def test_tta_falls_back_to_flips_when_y_and_x_differ(tmp_path: Path) -> None:
    """Rotation needs a square (y, x); a non-square sample must not raise.

    Every measured sample is 64 x 64 after the 1,4,4 stride, so this path is
    latent -- but a hidden test sample of another shape would otherwise crash
    at minute one of a twelve-hour run.
    """
    import submit

    detect = submit.load_detector(_checkpoint(tmp_path), torch.device("cpu"), tta=True)
    rng = np.random.default_rng(1)
    out = detect(rng.random((16, 16, 8), dtype=np.float32))
    assert out.shape == (16, 16, 8)


def test_clamp_keeps_smoothed_nodes_inside_the_volume() -> None:
    """Smoothing can pull a border node out of the grid; -1 fails validation."""
    from volt_biohub.postprocess import _clamp_to_grid

    coords = [np.array([[-0.4, 0.2, 70.0], [10.0, 10.0, 10.0]])]
    out = _clamp_to_grid(coords, (64, 64, 64))
    assert out[0][0].tolist() == [0.0, 0.2, 63.0]
    assert out[0][1].tolist() == [10.0, 10.0, 10.0]


def test_structure_enforcement_keeps_one_parent_and_two_children() -> None:
    """Two passes each add edges under a local guard; neither sees the other.

    The scorer silently collapses merge duplicates rather than erroring, so an
    in-degree of 2 is a free false positive that never surfaces as a failure.
    """
    from volt_biohub.postprocess import _enforce_structure

    # Two sources claim target 5; three targets claim source 0.
    links = [[(0, 5, 3.0), (1, 5, 1.0), (0, 6, 2.0), (0, 7, 4.0), (0, 8, 5.0)]]
    kept = _enforce_structure(links)[0]
    targets = [b for _, b, _ in kept]
    assert len(set(targets)) == len(targets)
    assert (1, 5, 1.0) in kept and (0, 5, 3.0) not in kept  # shortest parent wins
    assert sum(1 for a, _, _ in kept if a == 0) == 2        # at most two children
    assert (0, 8, 5.0) not in kept                          # the longest is dropped


@pytest.mark.parametrize(("label", "frames"), [
    ("empty middle frame", [1, 0, 1, 1, 1, 1]),
    ("all frames empty", [0, 0, 0, 0, 0]),
    ("single frame video", [1]),
    ("empty first and last", [0, 1, 1, 0]),
])
def test_postprocess_survives_degenerate_videos(label: str, frames: list[int]) -> None:
    """A threshold detector can emit nothing for a frame; a crash costs 12 hours.

    Every stage indexes by frame and several build a KD-tree, so an empty frame
    is the shape that breaks them. Kept as a parametrised case rather than one
    combined video because the first and last frame are special-cased.
    """
    from volt_biohub.postprocess import run_all

    coords = [
        np.array([[10.0, 10.0, 10.0 + i]]) if n else np.empty((0, 3))
        for i, n in enumerate(frames)
    ]
    out_coords, out_links = run_all(coords, 1.625, shape=(64, 64, 64))
    assert len(out_coords) == len(frames)
    assert len(out_links) == max(0, len(frames) - 1)


def test_writer_enforces_degrees_after_the_duplicate_collapse() -> None:
    """Collapsing duplicate positions can merge two parents into one with four kids.

    The scorer does not reject that -- it truncates out-degree above 2 to the two
    lowest edge ids and drops merge duplicates keeping the lowest, both in CSV row
    order. So the choice is made either way; making it here on link distance is
    the difference between a criterion and an accident. Found when separation 3.0
    um put detections close enough to collide after rounding, producing an
    out-degree of 3 that the validator rejected.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle"))
    from submission import dataset_rows

    # Two sources one voxel apart in the DOWNSAMPLED grid collapse to the same
    # original voxel once multiplied by (1, 4, 4) and rounded... so place them at
    # identical positions directly, which is what the collapse keys on.
    coords = [
        np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]),   # duplicates
        np.array([[10.0, 11.0, 10.0], [10.0, 11.0, 14.0],
                  [10.0, 11.0, 18.0], [10.0, 11.0, 22.0]]),
    ]
    # Each source claims two distinct children: four out-edges on one survivor.
    links = [[(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0), (1, 3, 4.0)]]

    rows = list(dataset_rows("ds", coords, links, upsample=(1, 4, 4)))
    edges = [r for r in rows if r["row_type"] == "edge"]
    out_deg: dict[int, int] = {}
    in_deg: dict[int, int] = {}
    for e in edges:
        out_deg[e["source_id"]] = out_deg.get(e["source_id"], 0) + 1
        in_deg[e["target_id"]] = in_deg.get(e["target_id"], 0) + 1

    assert out_deg and max(out_deg.values()) <= 2, f"out-degree not capped: {out_deg}"
    assert not in_deg or max(in_deg.values()) <= 1, f"merge emitted: {in_deg}"
    # Shortest links win, so the two kept are the 1.0 and 2.0 edges.
    kept_targets = {e["target_id"] for e in edges}
    assert len(kept_targets) == len(edges), "a target was claimed twice"


def _fork_fallback_fixture():
    """A parent whose NEAREST orphan is a duplicate and whose true sister is behind it.

    Laid out along x on the isotropic 1.625 um grid, four frames:

        t0  grandparent x=10      -> gives `p` a parent, which the pass requires
        t1  p          x=10
        t2  ch x=8   dup x=8.5   sister x=13
        t3  g1 x=6                g2 x=15

    `dup` is nearer to `ch` (0.81 um) than `sister` is (8.13 um), so the
    single-nearest rule proposes it and nothing else. It clears the distance and
    symmetry gates and fails only the divergence clause, having no child of its
    own -- which is the measured shape of the dominant false candidate, a second
    detection of a cell that is already linked.
    """
    coords = [
        np.array([[0.0, 0.0, 10.0]]),                       # t0: grandparent
        np.array([[0.0, 0.0, 10.0]]),                       # t1: p
        np.array([[0.0, 0.0, 8.0],                          # t2: ch
                  [0.0, 0.0, 8.5],                          #     dup   (orphan)
                  [0.0, 0.0, 13.0]]),                       #     sister(orphan)
        np.array([[0.0, 0.0, 6.0],                          # t3: g1 (child of ch)
                  [0.0, 0.0, 15.0]]),                       #     g2 (child of sister)
    ]
    links = [
        [(0, 0, 0.0)],                                      # grandparent -> p
        [(0, 0, 3.25)],                                     # p -> ch
        [(0, 0, 3.25), (2, 1, 3.25)],                       # ch -> g1, sister -> g2
    ]
    return coords, links


def test_single_nearest_orphan_hides_the_true_sister() -> None:
    """n_cand=1 reproduces the reference rule, and it finds nothing here.

    This is the failure no gate sweep could reach: every threshold in
    `add_safe_divisions` was swept individually and none binds, because the
    candidate that would pass them is never proposed. The pool audit
    (`scripts/division_candidate_audit.py`) measured the consequence -- a median
    of 1 candidate per frame against a cap of 2, and every true fork already
    ranking first -- so the term is proposal-limited, not ranking-limited.
    """
    from volt_biohub.postprocess import add_safe_divisions

    coords, links = _fork_fallback_fixture()
    _c, out = add_safe_divisions(coords, links, 1.625, n_cand=1)
    assert sum(1 for a, _, _ in out[1] if a == 0) == 1, "no fork should be added"


def test_fallback_reaches_the_sister_the_duplicate_masked() -> None:
    """n_cand=2 falls through the duplicate to the real second daughter."""
    from volt_biohub.postprocess import add_safe_divisions

    coords, links = _fork_fallback_fixture()
    _c, out = add_safe_divisions(coords, links, 1.625, n_cand=2)
    children_of_p = sorted(b for a, _b, _w in out[1] for b in [_b] if a == 0)
    assert children_of_p == [0, 2], "p should parent ch and the sister, not the dup"


def test_fallback_never_exceeds_two_children() -> None:
    """The fallback stops at the first candidate that clears the gates.

    Out-degree > 2 is truncated by the scorer silently, by lowest edge id, so a
    multi-propose variant of this knob could lose edges with no visible failure.
    Asking for more candidates than there are orphans also exercises
    `cKDTree.query`'s habit of padding the result with an out-of-range index.
    """
    from volt_biohub.postprocess import add_safe_divisions

    coords, links = _fork_fallback_fixture()
    _c, out = add_safe_divisions(coords, links, 1.625, n_cand=10)
    for t_links in out:
        counts: dict[int, int] = {}
        for a, _b, _w in t_links:
            counts[a] = counts.get(a, 0) + 1
        assert not counts or max(counts.values()) <= 2


def test_reject_diagnostic_agrees_with_the_live_gate_order(tmp_path, monkeypatch) -> None:
    """The diagnostic re-implements the gates; drift would misattribute the loss.

    It is a re-implementation on purpose -- the live loop must stay exactly the
    shipped behaviour and must not carry a diagnostic in its hot path -- so the
    thing worth pinning is that the two agree on the same fixture. The pool
    audit measured 19 of 25 GT divisions never reaching the pool at all, and
    this is the column that says which gate loses them; a wrong gate order here
    would point the next fix at the wrong clause.
    """
    import csv

    from volt_biohub.postprocess import add_safe_divisions

    out_csv = tmp_path / "reject.csv"
    monkeypatch.setenv("VOLT_DIV_REJECT", str(out_csv))
    coords, links = _fork_fallback_fixture()
    add_safe_divisions(coords, links, 1.625, n_cand=1)

    rows = list(csv.DictReader(out_csv.read_text().splitlines()))
    by_rank = {int(r["k_rank"]): r["reason"] for r in rows}
    # The duplicate is nearest and has no child of its own; the true sister is
    # behind it and clears everything. That is exactly why n_cand=1 finds
    # nothing here while n_cand=2 finds the fork.
    assert by_rank[0] == "no_grandchild"
    assert by_rank[1] == "pass"
