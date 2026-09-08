"""Round-trip tests for every coordinate-space boundary in the pipeline.

CLAUDE.md calls this the highest-risk code in the project: "Isotropic <->
original-anisotropic round-trips ... wrong by 4x in z, no exception raised,
score near zero." Four spaces exist and a coordinate crosses between them
several times per prediction:

1. **original anisotropic voxels** — ``(z, y, x)`` on the ``(64, 256, 256)``
   grid at ``(1.625, 0.40625, 0.40625)`` um/voxel. The scorer and the
   submission CSV both live here.
2. **the downsampled isotropic grid** — ``--downsample 1,4,4`` strided-reads
   the zarr to ``64**3`` at a cubic ``1.625`` um. Detection, linking and the
   network all live here.
3. **micrometres** — every distance (7 um match tolerance, 8 um link gate,
   2.93 um nucleus sigma, suppression radii) is physical, never voxel.
4. **an upsampled isotropic grid** — ``img_proc.rescale_coords_to_isotropic``
   normalises to ``min(scale)``, i.e. it *interpolates z up* by 4 instead of
   binning y/x down. That is the opposite convention to space 2 and the two
   disagree by 4x in y and x, so ``io.rescale_graph_to_original`` cannot have a
   default: it takes the isotropic scale it is inverting and the tests at the
   bottom of this file pin *both* conventions through the one function. (It used
   to hardcode ``min(scale)``, which silently returned garbage for the grid
   ``open_dataset(target_scale=(1.625,) * 3)`` actually produces.)

The tests are written so that a 4x z error cannot pass. Several assert that
the *wrong* transform collapses the score, because a round-trip test that only
checks ``f_inv(f(x)) == x`` passes happily when ``f`` and ``f_inv`` are wrong in
the same direction — which is exactly what the pre-existing
``test_rescale_coords_roundtrip`` does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tracksdata as td
import zarr

from volt_biohub.img_proc import rescale_coords_to_isotropic
from volt_biohub.io import (
    DEFAULT_SCALE,
    _parse_scale,
    open_dataset,
    rescale_graph_to_original,
    save_graph,
)
from volt_biohub.linking import build_graph, link_frames
from volt_biohub.metrics import evaluate
from volt_biohub.scoring import DOWNSAMPLE, ISO_UM, ORIGINAL_SCALE

FIXTURE_DIR = Path(__file__).parent / "data" / "division_clip"
FIXTURE_NAME = "division_clip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_geff(path):
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def _point_graph(rows: list[tuple[int, float, float, float]],
                 edges: list[tuple[int, int]]) -> td.graph.InMemoryGraph:
    """A minimal graph from explicit ``(t, z, y, x)`` rows and index-pair edges."""
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes(
        [{"t": int(t), "z": float(z), "y": float(y), "x": float(x)} for t, z, y, x in rows]
    )
    if edges:
        g.bulk_add_edges([{"source_id": ids[a], "target_id": ids[b]} for a, b in edges])
    return g


def _edge_jaccard(pred, gt, **kw) -> float:
    er = evaluate(pred, gt, **kw)
    denom = er.edge_tp + er.edge_fp + er.edge_fn
    return er.edge_tp / denom if denom else float("nan")


@pytest.fixture
def gt_on_downsampled_grid(division_clip_fixture):
    """GT re-expressed on the ``1,4,4`` grid, as a detector would emit it.

    Returns ``(gt_graph, coords_per_t, links_per_t)``. ``coords_per_t`` is the
    detector's coordinate system — integer indices into the strided read — so
    round-tripping it through :func:`build_graph` and scoring it is the exact
    journey a real prediction makes. Rounding is deliberate: a detector cannot
    emit ``y = 33.5``, and the resulting <=2-voxel (0.81 um) error must stay
    invisible against the 7 um match tolerance.
    """
    gt = _load_geff(FIXTURE_DIR / f"{FIXTURE_NAME}.geff")
    nodes = gt.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    edges = gt.edge_attrs(attr_keys=["source_id", "target_id"])

    ds = np.asarray(DOWNSAMPLE, dtype=np.float64)
    n_t = int(nodes["t"].max()) + 1

    coords_per_t: list[np.ndarray] = []
    where: dict[int, tuple[int, int]] = {}
    for t in range(n_t):
        frame = nodes.filter(pl.col("t") == t)
        arr = frame.select("z", "y", "x").to_numpy().astype(np.float64) / ds
        coords_per_t.append(np.round(arr).reshape(-1, 3))
        for k, node_id in enumerate(frame["node_id"].to_list()):
            where[node_id] = (t, k)

    links_per_t: list[list[tuple[int, int, float]]] = [[] for _ in range(n_t - 1)]
    for src, tgt in zip(edges["source_id"].to_list(), edges["target_id"].to_list(), strict=True):
        (t_src, i), (_, j) = where[src], where[tgt]
        links_per_t[t_src].append((i, j, 0.0))

    return gt, coords_per_t, links_per_t


# ---------------------------------------------------------------------------
# 1. The grid contract: constants that must agree or every distance is wrong
# ---------------------------------------------------------------------------

def test_downsample_grid_is_cubic():
    """``ORIGINAL_SCALE * DOWNSAMPLE`` must be ``ISO_UM`` on all three axes.

    Everything downstream of detection treats the training grid as cubic and
    passes the scalar ``ISO_UM`` where a 3-tuple belongs — ``detectors._peak_window``,
    ``link_sequence((ISO_UM,)*3)``. That is only sound because
    ``0.40625 * 4 == 1.625`` exactly. Change ``DOWNSAMPLE`` without changing
    ``ISO_UM`` and every physical distance in the pipeline is silently wrong by
    the ratio, with no exception anywhere.
    """
    assert tuple(s * d for s, d in zip(ORIGINAL_SCALE, DOWNSAMPLE, strict=True)) == (ISO_UM,) * 3
    # The z axis is the one that is never downsampled, so ISO_UM *is* the z pitch.
    assert ORIGINAL_SCALE[0] == ISO_UM
    assert ORIGINAL_SCALE == DEFAULT_SCALE


def test_parse_scale_reads_zyx_not_tzy(division_clip_fixture):
    """``_parse_scale`` must take the *last* three of the 4-element OME scale.

    The stored transform is ``(t, z, y, x) = (1.0, 1.625, 0.40625, 0.40625)``.
    Slicing the first three instead yields ``(1.0, 1.625, 0.40625)`` — z off by
    1.625/1.0 and y off by 4x — and every one of those numbers looks plausible,
    so nothing would raise.
    """
    group = zarr.open_group(str(FIXTURE_DIR / f"{FIXTURE_NAME}.zarr"), mode="r")
    stored = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert len(stored) == 4, "expected a (t, z, y, x) scale; the slicing assumption changed"
    assert _parse_scale(dict(group.attrs)) == ORIGINAL_SCALE


def test_open_dataset_scale_times_downsample_is_the_training_grid(division_clip_fixture):
    """The voxel size train/predict compute must be the cubic training grid.

    ``train_unet_transformer`` and ``predict_unet_transformer`` both derive
    ``voxel_size = ds.scale * downsample`` and feed it to the Gaussian target
    width and the suppression kernel. If ``ds.scale`` ever came back already
    resampled, that product would be 4x wrong in y/x with nothing to show for it.
    """
    ds = open_dataset(FIXTURE_DIR / FIXTURE_NAME, load_image=False, downsample=DOWNSAMPLE)
    assert ds.scale == ORIGINAL_SCALE
    voxel_size = tuple(s * d for s, d in zip(ds.scale, DOWNSAMPLE, strict=True))
    assert voxel_size == (ISO_UM,) * 3
    # Strided read of (64, 256, 256) by (1, 4, 4) is exactly 64 cubed.
    assert ds.image_shape[1:] == (64, 64, 64)


def test_open_dataset_downsample_applies_to_the_loaded_image(division_clip_fixture):
    """``image_shape`` must describe the array actually returned.

    This is the one place where the two isotropic conventions can be mixed
    without any arithmetic mistake at all: ask for a downsampled dataset, get a
    full-resolution image back, then index it with grid coordinates. The value
    check is the other half — a shape assertion alone would also pass for a
    mean-pool, which biases every coordinate by ``(d - 1) / 2`` voxels.
    """
    meta_only = open_dataset(FIXTURE_DIR / FIXTURE_NAME, load_image=False, downsample=DOWNSAMPLE)
    loaded = open_dataset(
        FIXTURE_DIR / FIXTURE_NAME, load_image=True, normalize=False, downsample=DOWNSAMPLE
    )
    assert loaded.image_shape == meta_only.image_shape == (5, 64, 64, 64)
    assert tuple(loaded.image.shape) == loaded.image_shape

    dz, dy, dx = DOWNSAMPLE
    group = zarr.open_group(str(FIXTURE_DIR / f"{FIXTURE_NAME}.zarr"), mode="r")
    np.testing.assert_array_equal(
        np.asarray(loaded.image), np.asarray(group["0"][:, ::dz, ::dy, ::dx])
    )


def test_open_dataset_refuses_downsample_with_target_scale(division_clip_fixture):
    """The two ways to reach an isotropic grid must not be combinable.

    ``downsample`` strides the read while ``target_scale`` interpolates, and the
    zoom for the second is derived from ``ds.scale``, which keeps describing the
    *original* grid. Applying both would scale y/x twice — 4x wrong, nothing
    raised — and no caller has ever wanted that, so a raise is the whole fix.
    """
    with pytest.raises(ValueError, match="cannot be combined"):
        open_dataset(
            FIXTURE_DIR / FIXTURE_NAME, downsample=DOWNSAMPLE, target_scale=(ISO_UM,) * 3
        )


@pytest.mark.parametrize("bad", [(1, 4), (1, 4, 4, 4), (1, 4, 0), (1, 1.5, 1.5)])
def test_open_dataset_rejects_a_malformed_downsample(division_clip_fixture, bad):
    """A stride tuple that is not three positive integers must raise, not truncate.

    ``zip`` used to drop the surplus axis silently, so ``(1, 4)`` produced a
    3-dimensional ``image_shape`` for a 4-dimensional array, and a float stride is
    floored by the slice. Both are one-axis scale errors of the kind that costs
    the whole score without an exception.
    """
    with pytest.raises(ValueError, match="downsample"):
        open_dataset(FIXTURE_DIR / FIXTURE_NAME, load_image=False, downsample=bad)


def test_loader_decimates_rather_than_pooling(division_clip_fixture):
    """Downsampled index ``j`` must hold original voxel ``j * d``, not a bin average.

    This is the image-side half of the mapping and the only half nobody tests.
    ``coord_original = coord_downsampled * downsample`` (in ``build_graph``,
    ``predict_video`` and ``get_window_data``) is correct *because* the loaders
    decimate — ``arr[::dz, ::dy, ::dx]`` samples voxel ``j * d`` itself. Swap in a
    mean-pool or ``scipy.ndimage.zoom`` and every coordinate acquires a
    ``(d - 1) / 2`` voxel bias with nothing raised; at ``1,4,4`` that is 1.5
    voxels (0.61 um) in y and x.

    So this asserts against ``scoring.load_volumes`` — the live loader — rather
    than restating the slice in the test, which would only be testing numpy. The
    comparison is whole-volume because a single voxel is a weak discriminator:
    at a smooth location the mean of a 4x4 block sits within 0.005 of its corner
    sample, well inside any tolerance a spot check would use.
    """
    from volt_biohub.scoring import load_volumes

    group = zarr.open_group(str(FIXTURE_DIR / f"{FIXTURE_NAME}.zarr"), mode="r")
    q = group.attrs["image_statistics"]["quantiles"]
    lo, hi = float(q["0.001"]), float(q["0.999"])
    dz, dy, dx = DOWNSAMPLE

    loaded = load_volumes(FIXTURE_DIR, FIXTURE_NAME)
    raw = np.asarray(group["0"][:, ::dz, ::dy, ::dx]).astype(np.float32)
    np.testing.assert_allclose(loaded, np.clip((raw - lo) / (hi - lo + 1e-6), 0, None))

    # And the identity that makes `coord * downsample` sound, stated as a
    # property of the loaded array rather than of the slice expression.
    idx = (7, 11, 23)
    assert loaded[0][idx] == pytest.approx(
        np.clip((float(group["0"][0, idx[0] * dz, idx[1] * dy, idx[2] * dx]) - lo)
                / (hi - lo + 1e-6), 0, None)
    )


# ---------------------------------------------------------------------------
# 2. The live path: GT -> downsampled grid -> back -> scorer
# ---------------------------------------------------------------------------

def test_gt_survives_the_downsample_round_trip(gt_on_downsampled_grid):
    """GT pushed onto the ``1,4,4`` grid and back must score a perfect 1.0.

    This is the whole crossing in one assertion: original voxels -> divide by
    ``DOWNSAMPLE`` -> round to an integer detector coordinate -> ``build_graph``
    multiplies by ``DOWNSAMPLE`` -> ``evaluate`` scales by ``ORIGINAL_SCALE``.
    Anything wrong by 4x anywhere along it puts every node outside the 7 um
    tolerance. The companion test below proves the assertion has teeth.
    """
    gt, coords_per_t, links_per_t = gt_on_downsampled_grid
    pred = build_graph(coords_per_t, links_per_t, upsample=DOWNSAMPLE)
    er = evaluate(pred, gt, scale=ORIGINAL_SCALE, max_distance=7.0)
    assert er.edge_tp == gt.num_edges()
    assert er.edge_fp == 0
    assert er.edge_fn == 0
    assert er.division_tp == 1 and er.division_fn == 0  # the fixture's known division


@pytest.mark.parametrize("bad_upsample", [(1, 1, 1), (4, 4, 4), (4, 1, 1), (1, 4, 1)])
def test_wrong_upsample_collapses_the_score(gt_on_downsampled_grid, bad_upsample):
    """Any wrong ``upsample`` must score zero, not merely worse.

    The point of the exercise: the failure mode CLAUDE.md warns about is silent,
    so the test above is only worth having if a mistake is catastrophic rather
    than subtle. ``(1, 1, 1)`` is the forgot-to-scale bug, ``(4, 4, 4)`` is the
    wrong-convention bug (z upsampled instead of y/x downsampled), and the mixed
    pair are the transposed-tuple bugs.
    """
    gt, coords_per_t, links_per_t = gt_on_downsampled_grid
    pred = build_graph(coords_per_t, links_per_t, upsample=bad_upsample)
    assert _edge_jaccard(pred, gt, scale=ORIGINAL_SCALE, max_distance=7.0) == 0.0


def test_training_coords_and_inference_upsample_are_exact_inverses(division_clip_fixture):
    """``get_window_data``'s divide and ``predict_video``'s multiply must agree.

    Training divides raw GT coordinates by ``downsample`` as floats; inference
    multiplies detected coordinates by the same tuple. The two live in different
    files with no shared constant, so this pins them against each other on real
    GT. It also bounds the cost of the rounding a detector unavoidably applies:
    half a downsampled voxel is 0.81 um in y/x, an eighth of the 7 um tolerance,
    which is why sub-voxel localisation is worthless here — and why a *4x*
    error is not: it displaces a node at depth ``z`` by ``3z`` voxels, i.e.
    4.88 um for every z voxel of depth, so it clears the tolerance by z = 2.
    """
    from train_unet_transformer import get_window_data

    gt = _load_geff(FIXTURE_DIR / f"{FIXTURE_NAME}.geff")
    nodes = gt.node_attrs(attr_keys=["t", "z", "y", "x"])
    window = get_window_data(gt, (5, 64, 64, 64), 0, window_size=2, downsample=DOWNSAMPLE)
    assert window is not None

    ds = np.asarray(DOWNSAMPLE, dtype=np.float64)
    for frame, t in enumerate((0, 1)):
        expected = nodes.filter(pl.col("t") == t).select("z", "y", "x").to_numpy().astype(np.float64)
        train_coords = window.coords[frame].numpy().astype(np.float64)
        np.testing.assert_allclose(train_coords * ds, expected, atol=1e-4)
        # What a detector can actually emit, and what it costs in micrometres.
        err_um = np.abs(np.round(train_coords) * ds - expected) * np.asarray(ORIGINAL_SCALE)
        assert np.linalg.norm(err_um, axis=-1).max() < 7.0


def test_scorer_needs_original_voxel_space_and_original_scale(division_clip_fixture):
    """A 10-voxel x offset is 4.06 um in original space and 16.25 um isotropic.

    Two mistakes are indistinguishable by inspection and this separates them:
    handing the scorer isotropic-grid coordinates, or handing it original-space
    coordinates with the isotropic scale. Both push a node that is comfortably
    inside the 7 um tolerance comfortably outside it, and the node then matches
    nothing, so every edge through it becomes an FN.

    The offset is in x rather than z on purpose. z has the same 1.625 um pitch on
    both grids, so a z-only probe cannot tell the two apart — the whole 4x lives
    in y and x. The three rejected scales are, in order: the training grid's
    cubic scale, ``scale=None`` (the *default* of :func:`metrics.evaluate` and a
    live landmine — it measures raw voxels, so 10 > 7), and a ``(z, y, x)``
    tuple built in the wrong order.
    """
    gt = _point_graph([(0, 30, 100, 100), (1, 30, 100, 100)], [(0, 1)])
    offset = [(0, 30, 100, 110), (1, 30, 100, 110)]

    assert _edge_jaccard(_point_graph(offset, [(0, 1)]), gt,
                         scale=ORIGINAL_SCALE, max_distance=7.0) == 1.0
    for wrong in ((ISO_UM,) * 3, None, (0.40625, 0.40625, 1.625)):
        assert _edge_jaccard(_point_graph(offset, [(0, 1)]), gt,
                             scale=wrong, max_distance=7.0) == 0.0, wrong


# ---------------------------------------------------------------------------
# 3. Micrometres, never voxels: the linker and the suppression kernel
# ---------------------------------------------------------------------------

def test_link_gate_is_microns_not_voxels():
    """The same 6-voxel step must pass the gate in y and fail it in z.

    6 voxels is 9.75 um in z but 2.44 um in y on the original grid, so a gate
    applied in voxels rather than um is wrong by 4x on exactly one axis — and
    ``|dz|`` has median 0.00 um while ``|dxy|`` has median 1.22 um, so the
    error hides in the axis that moves least.
    """
    a = np.array([[30.0, 100.0, 100.0]])
    step_z = np.array([[36.0, 100.0, 100.0]])
    step_y = np.array([[30.0, 106.0, 100.0]])

    assert link_frames(a, step_y, ORIGINAL_SCALE, gate_um=8.0) == [(0, 0, pytest.approx(2.4375))]
    assert link_frames(a, step_z, ORIGINAL_SCALE, gate_um=8.0) == []
    # On the cubic training grid 6 voxels is 9.75 um on every axis, so both fail.
    assert link_frames(a, step_y, (ISO_UM,) * 3, gate_um=8.0) == []
    assert link_frames(a, step_z, (ISO_UM,) * 3, gate_um=8.0) == []


def test_pool_kernel_follows_the_grid_it_is_given():
    """A suppression radius in um must become a different kernel on each grid.

    ``predict_video`` passes ``ds.scale * downsample``. Passing ``ds.scale``
    alone — the obvious slip, since that is what ``open_dataset`` returns —
    turns a 3 um radius into a 7-voxel window in y/x instead of 3, suppressing
    over 11.4 um and merging division sisters (min separation 2.87 um). Pinning
    both values keeps the two distinguishable.
    """
    from predict_unet_transformer import pool_kernel_from_um

    assert pool_kernel_from_um(3.0, (ISO_UM,) * 3) == (3, 3, 3)
    assert pool_kernel_from_um(3.0, ORIGINAL_SCALE) == (3, 7, 7)


# ---------------------------------------------------------------------------
# 4. Submission CSV space
# ---------------------------------------------------------------------------

def test_csv_round_trip_preserves_original_voxel_coordinates(gt_on_downsampled_grid, tmp_path):
    """``geff -> submission.csv -> geff`` must not move a single coordinate.

    Also asserts the coordinates are in the 256-wide original space rather than
    the 64-wide training grid. That check is not redundant: isotropic-space
    coordinates are all <= 63, which is a *legal* value in original space, so no
    range validation can catch the mistake — only the magnitude of real data can.
    """
    from csv_to_geffs import csv_to_geffs
    from geffs_to_csv import geffs_to_csv

    gt, coords_per_t, links_per_t = gt_on_downsampled_grid
    graph = build_graph(coords_per_t, links_per_t, upsample=DOWNSAMPLE)
    before = graph.node_attrs().select("t", "z", "y", "x").to_numpy()

    save_graph(graph, tmp_path / "geffs" / f"{FIXTURE_NAME}.geff")
    csv_path = geffs_to_csv(tmp_path / "geffs", tmp_path / "submission.csv")
    csv_to_geffs(csv_path, tmp_path / "back")

    after = _load_geff(tmp_path / "back" / f"{FIXTURE_NAME}.geff")
    np.testing.assert_array_equal(
        np.sort(before, axis=0),
        np.sort(after.node_attrs().select("t", "z", "y", "x").to_numpy(), axis=0),
    )
    assert before[:, 2].max() > 63, "coordinates look like the 64-cubed grid, not original space"
    # The scorer must be indifferent to the detour through CSV.
    assert _edge_jaccard(after, gt, scale=ORIGINAL_SCALE, max_distance=7.0) == 1.0


# ---------------------------------------------------------------------------
# 5. io.rescale_graph_to_original — the site CLAUDE.md names, and its trap
# ---------------------------------------------------------------------------

def test_rescale_coords_to_isotropic_upsamples_z_fourfold():
    """Pin the *direction and magnitude*, which the existing round-trip test cannot.

    ``rescale_coords_to_isotropic`` normalises to ``min(scale)``, i.e. it
    interpolates z **up** by 1.625/0.40625 = 4 and leaves y/x alone. That is the
    opposite of the ``--downsample 1,4,4`` convention the rest of the project
    uses, and confusing the two is the 4x z bug. The pre-existing
    ``test_rescale_coords_roundtrip`` asserts only ``f_inv(f(x)) == x`` at a
    synthetic ``(4, 1, 1)`` scale, which passes even if both halves are inverted.
    """
    coords = np.array([[0, 10.0, 40.0, 60.0], [1, 63.0, 255.0, 0.0]])
    iso = rescale_coords_to_isotropic(coords, ORIGINAL_SCALE)

    np.testing.assert_allclose(iso[:, 0], coords[:, 0])           # t untouched
    np.testing.assert_allclose(iso[:, 1], coords[:, 1] * 4.0)     # z upsampled
    np.testing.assert_allclose(iso[:, 2:], coords[:, 2:])         # y, x untouched


def test_rescale_graph_to_original_inverts_the_min_scale_convention():
    """``rescale_graph_to_original`` inverts ``img_proc``'s ``min(scale)`` upsample.

    Cross-module round trip, because the forward transform lives in
    ``img_proc`` and the inverse in ``io`` and nothing links them. The isotropic
    scale has to be stated — ``min(ORIGINAL_SCALE)`` is what that convention
    normalises to — since the test below drives the same function with the other
    convention and the two answers differ by 4x in y/x. ``z = 63`` and ``y = 255``
    are included on purpose: the far edge of the volume is where a rounding or
    clamping bug hides (see the ``_stamp`` regression that lost 225 real GT nodes).
    """
    coords = np.array([[0, 10.0, 40.0, 60.0], [1, 63.0, 255.0, 0.0]])
    iso = rescale_coords_to_isotropic(coords, ORIGINAL_SCALE)
    graph = _point_graph([(int(t), z, y, x) for t, z, y, x in iso], [])

    back = rescale_graph_to_original(graph, ORIGINAL_SCALE, min(ORIGINAL_SCALE))
    np.testing.assert_allclose(
        back.node_attrs().select("t", "z", "y", "x").to_numpy(), coords
    )


def test_rescale_graph_to_original_inverts_an_explicit_target_scale():
    """The same function must also invert ``target_scale=(1.625, 1.625, 1.625)``.

    ``_process_on_gpu`` computes ``zoom = scale / target_scale`` elementwise, so
    an explicit cubic target *downsamples y and x* rather than upsampling z —
    the same grid the strided ``1,4,4`` read produces. This is the pairing the
    signature invites (``ds.original_scale`` with ``ds.scale``) and the one that
    used to return garbage, because the inverse recomputed
    ``zoom = scale / min(scale)`` instead of using the target it was handed.
    """
    target = (ISO_UM, ISO_UM, ISO_UM)
    zoom = np.asarray(ORIGINAL_SCALE) / np.asarray(target)  # (1.0, 0.25, 0.25)
    coords = np.array([[0, 10.0, 40.0, 60.0], [1, 63.0, 252.0, 0.0]])
    iso = coords.copy()
    iso[:, 1:] *= zoom

    graph = _point_graph([(int(t), z, y, x) for t, z, y, x in iso], [])
    back = rescale_graph_to_original(graph, ORIGINAL_SCALE, target)
    np.testing.assert_allclose(
        back.node_attrs().select("t", "z", "y", "x").to_numpy(), coords
    )


def test_rescale_graph_to_original_needs_the_grid_named():
    """Omitting the isotropic scale must be a TypeError, not a guess.

    The two conventions above differ by 4x in y and x on identical input, so any
    default is wrong half the time and wrong silently.
    """
    graph = _point_graph([(0, 10.0, 10.0, 10.0)], [])
    with pytest.raises(TypeError):
        rescale_graph_to_original(graph, ORIGINAL_SCALE)


def test_resampled_dataset_pairs_with_rescale_graph_to_original(division_clip_fixture):
    """The live pairing, on real data: ``ds.original_scale`` with ``ds.scale``.

    ``open_dataset(target_scale=(1.625,) * 3)`` resamples the image *and* the GT
    node coordinates onto the cubic grid, and this is the only mode
    ``rescale_graph_to_original`` exists to invert. Recovery is not exact — the
    forward transform rounds the resampled coordinates to integers, which is 2
    original voxels in y/x — so the assertion is in micrometres against the 7 um
    match tolerance, the only accuracy that is actually measured.
    """
    gt = _load_geff(FIXTURE_DIR / f"{FIXTURE_NAME}.geff")
    expected = gt.node_attrs(attr_keys=["t", "z", "y", "x"]).sort("t", "z", "y", "x")

    ds = open_dataset(
        FIXTURE_DIR / FIXTURE_NAME, target_scale=(ISO_UM,) * 3,
        normalize=False, require_tracks=True, device="cpu",
    )
    assert ds.scale == (ISO_UM,) * 3 and ds.original_scale == ORIGINAL_SCALE
    assert tuple(ds.image.shape) == (5, 64, 64, 64)

    back = rescale_graph_to_original(ds.tracks, ds.original_scale, ds.scale)
    got = back.node_attrs(attr_keys=["t", "z", "y", "x"]).sort("t", "z", "y", "x")

    np.testing.assert_array_equal(got["t"].to_numpy(), expected["t"].to_numpy())
    err_um = np.abs(
        got.select("z", "y", "x").to_numpy() - expected.select("z", "y", "x").to_numpy()
    ) * np.asarray(ORIGINAL_SCALE)
    assert np.linalg.norm(err_um, axis=-1).max() < 7.0
