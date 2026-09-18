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
