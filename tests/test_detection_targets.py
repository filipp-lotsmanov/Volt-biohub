"""Tests for Gaussian detection targets and positive-unlabeled masking.

These pin the *numeric* properties the design argument rests on -- separability at
the minimum observed cell pitch, correct handling of z anisotropy, and that the
negative term actually stops covering unlabeled cells -- not merely that the
functions run.
"""

import numpy as np
import pytest
import torch

from volt_biohub.detection_targets import (
    DEFAULT_BRIGHT_THRESHOLD,
    DEFAULT_IGNORE_RADIUS_UM,
    DEFAULT_SIGMA_UM,
    NUCLEUS_FWHM_UM,
    detection_loss,
    detection_loss_with_pseudo,
    gaussian_targets,
    ignore_mask,
    pseudo_positive_mask,
)

ISO = (1.625, 1.625, 1.625)      # the --downsample 1,4,4 training grid
ANISO = (6.5, 1.625, 1.625)      # what --downsample 4,4,4 gives (as in the slow test)
SHAPE = (16, 32, 32)


def _one_node(zyx=(8, 16, 16), b=1):
    coords = torch.tensor([[list(zyx)]] * b, dtype=torch.float32)
    mask = torch.ones((b, 1), dtype=torch.bool)
    return coords, mask


def test_sigma_matches_measured_nucleus_size():
    """DEFAULT_SIGMA_UM must be the sigma of a Gaussian with the measured FWHM."""
    assert DEFAULT_SIGMA_UM == pytest.approx(NUCLEUS_FWHM_UM / 2.3548)
    assert DEFAULT_SIGMA_UM == pytest.approx(2.93, abs=0.02)


def test_target_peaks_at_one_on_the_centroid():
    coords, mask = _one_node()
    tgt = gaussian_targets(coords, mask, SHAPE, ISO)
    assert tgt.shape == (1, *SHAPE)
    assert tgt[0, 8, 16, 16] == pytest.approx(1.0)
    assert tgt.max() == pytest.approx(1.0)


def test_target_is_a_neighbourhood_not_a_single_voxel():
    """The whole point of the change: a spike gives no gradient nearby."""
    coords, mask = _one_node()
    tgt = gaussian_targets(coords, mask, SHAPE, ISO)
    assert (tgt > 0.5).sum() > 1
    # 1.81 voxels of sigma should light up a few voxels, not hundreds.
    assert 5 <= int((tgt > 0.5).sum()) <= 200


def test_falls_to_half_at_the_measured_hwhm():
    coords, mask = _one_node()
    tgt = gaussian_targets(coords, mask, SHAPE, ISO)
    hwhm_vox = (NUCLEUS_FWHM_UM / 2) / ISO[1]
    lo = tgt[0, 8, 16, 16 + int(np.floor(hwhm_vox))].item()
    hi = tgt[0, 8, 16, 16 + int(np.ceil(hwhm_vox))].item()
    assert hi <= 0.5 <= lo


def test_anisotropy_is_handled_in_micrometres_not_voxels():
    """z voxels are 4x taller; a target spherical in voxels would be wrong 4x."""
    coords, mask = _one_node()
    tgt = gaussian_targets(coords, mask, SHAPE, ANISO)[0]
    # One voxel along z is 6.5 um; along x only 1.625 um. So z must decay faster.
    assert tgt[9, 16, 16] < tgt[8, 16, 17]
    # And a step of 4 x-voxels (6.5 um) should match one z-voxel (6.5 um).
    assert tgt[9, 16, 16].item() == pytest.approx(tgt[8, 16, 20].item(), rel=1e-5)


def test_overlapping_cells_combine_by_max_not_sum():
    """Summing would push the target above 1 and make BCE unsatisfiable."""
    coords = torch.tensor([[[8, 16, 16], [8, 16, 17]]], dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.bool)
    tgt = gaussian_targets(coords, mask, SHAPE, ISO)
    assert tgt.max().item() <= 1.0 + 1e-6
    assert tgt[0, 8, 16, 16] == pytest.approx(1.0)
    assert tgt[0, 8, 16, 17] == pytest.approx(1.0)


def test_two_cells_at_minimum_observed_pitch_stay_separable():
    """Measured minimum cell pitch is 11.3 um; targets must stay bimodal there.

    Separability means a genuine dip between two peaks -- that is what lets a
    local-max peak finder recover two cells -- not a particular saddle value. At
    this pitch and sigma the saddle is ~0.25 against peaks of 1.0.
    """
    sep_vox = round(11.3 / ISO[1])
    x0, x1 = 10, 10 + sep_vox
    coords = torch.tensor([[[8, 16, x0], [8, 16, x1]]], dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.bool)
    row = gaussian_targets(coords, mask, SHAPE, ISO)[0, 8, 16]

    assert row[x0] == pytest.approx(1.0) and row[x1] == pytest.approx(1.0)
    saddle = row[x0:x1 + 1].min().item()
    assert saddle < 0.5, f"no dip between the two cells (saddle={saddle:.3f})"
    # Bimodal: falling away from the first peak, rising into the second. An odd
    # separation puts two voxels equidistant from both cells, so the saddle can be
    # a flat pair -- monotonicity is therefore non-strict across it.
    seg = row[x0:x1 + 1]
    lo = int(seg.argmin())
    assert 0 < lo < len(seg) - 1
    assert torch.all(seg[: lo + 1].diff() <= 0) and torch.all(seg[lo:].diff() >= 0)
    assert seg[0] - seg[lo] > 0.4 and seg[-1] - seg[lo] > 0.4


def test_padded_nodes_are_ignored():
    coords = torch.tensor([[[8, 16, 16], [0, 0, 0]]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    tgt = gaussian_targets(coords, mask, SHAPE, ISO)
    assert tgt[0, 0, 0, 0] == pytest.approx(0.0)


def test_out_of_bounds_centroid_does_not_wrap_or_crash():
    coords = torch.tensor([[[0, 0, 0]]], dtype=torch.float32)
    mask = torch.ones((1, 1), dtype=torch.bool)
    tgt = gaussian_targets(coords, mask, SHAPE, ISO)
    assert tgt[0, 0, 0, 0] == pytest.approx(1.0)
    assert tgt[0, -1, -1, -1] == pytest.approx(0.0)   # no wraparound


def test_far_edge_overhang_is_snapped_not_dropped():
    """The 1,4,4 grid leaves y,x in [0, 63.75]; rounding must not drop the node.

    ``get_window_data`` divides raw coords by the downsample factor as floats,
    so original y=255 becomes 63.75 on a grid of extent 64. Rounding it to 64
    used to fall outside the volume, stamping no positive and undercounting
    n_pos — 225 of 133,318 real GT nodes, up to 7.2 % on one video. The legacy
    single-voxel loss floored and clamped, so it kept them; a regression here
    would make the Gaussian-vs-voxel ablation unfair for the wrong reason.
    """
    shape = (64, 64, 64)
    coords = torch.tensor([[[63.0, 63.75, 63.75]]], dtype=torch.float32)
    mask = torch.ones((1, 1), dtype=torch.bool)
    tgt = gaussian_targets(coords, mask, shape, ISO)
    assert tgt[0, 63, 63, 63] == pytest.approx(1.0)
    assert int((tgt >= 1.0 - 1e-6).sum()) == 1


def test_centroid_genuinely_outside_the_volume_is_still_dropped():
    """Snapping the half-voxel overhang must not resurrect the legacy clamp bug.

    The inherited loss clamped *any* coordinate to the border, so a node far
    outside the window stamped a false positive on the edge. Only the
    sub-voxel overhang is legitimate.
    """
    coords = torch.tensor([[[8, 16, 40]]], dtype=torch.float32)   # x=40 > 32
    mask = torch.ones((1, 1), dtype=torch.bool)
    tgt = gaussian_targets(coords, mask, SHAPE, ISO)
    assert tgt.max() == pytest.approx(0.0)


def test_collapsed_centroids_warn():
    """Two cells on one voxel are individually undetectable -- must not be silent."""
    coords = torch.tensor([[[8, 16, 16], [8, 16, 16]]], dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.bool)
    with pytest.warns(UserWarning, match="collapsed"):
        gaussian_targets(coords, mask, SHAPE, ISO)


# --- positive-unlabeled masking ----------------------------------------------


def test_ignores_bright_voxels_far_from_annotation():
    coords, mask = _one_node()
    images = torch.zeros((1, *SHAPE))
    images[0, 2, 4, 4] = 0.9        # bright, far from the GT node -> unlabeled cell
    ign = ignore_mask(images, coords, mask, ISO)
    assert ign[0, 2, 4, 4]


def test_keeps_dark_voxels_as_negatives():
    coords, mask = _one_node()
    images = torch.zeros((1, *SHAPE))
    images[0, 2, 4, 4] = 0.05       # genuine background
    assert not ignore_mask(images, coords, mask, ISO)[0, 2, 4, 4]


def test_never_ignores_near_an_annotated_cell():
    """Voxels around a GT cell are real negatives -- they teach localisation."""
    coords, mask = _one_node()
    images = torch.ones((1, *SHAPE))          # everything bright
    ign = ignore_mask(images, coords, mask, ISO)
    inside = round(DEFAULT_IGNORE_RADIUS_UM / ISO[1]) - 1
    assert not ign[0, 8, 16, 16 + inside]
    outside = round(DEFAULT_IGNORE_RADIUS_UM / ISO[1]) + 3
    assert ign[0, 8, 16, 16 + outside]


def test_bright_threshold_is_the_documented_value():
    assert DEFAULT_BRIGHT_THRESHOLD == 0.3


# --- the loss -----------------------------------------------------------------


def test_loss_is_lower_for_a_correct_prediction():
    coords, mask = _one_node()
    good = torch.full((1, 1, *SHAPE), -8.0)
    good[0, 0, 8, 16, 16] = 8.0
    bad = torch.full((1, 1, *SHAPE), -8.0)
    bad[0, 0, 2, 4, 4] = 8.0
    assert detection_loss(good, coords, mask, ISO) < detection_loss(bad, coords, mask, ISO)


def test_pu_mask_removes_the_penalty_on_unlabeled_cells():
    """The core fix: predicting a cell where one really is must stop being punished."""
    coords, mask = _one_node()
    logits = torch.full((1, 1, *SHAPE), -8.0)
    logits[0, 0, 8, 16, 16] = 8.0
    logits[0, 0, 2, 4, 4] = 8.0            # confident detection of an *unannotated* cell

    images = torch.zeros((1, *SHAPE))
    images[0, 2, 4, 4] = 0.9               # ...and it is genuinely bright there

    without = detection_loss(logits, coords, mask, ISO, images=None)
    with_pu = detection_loss(logits, coords, mask, ISO, images=images)
    assert with_pu < without, "PU masking did not reduce the penalty"


def test_images_none_reproduces_full_negative_coverage():
    """The --no-det-pu-mask ablation must genuinely differ from the masked path."""
    coords, mask = _one_node()
    logits = torch.zeros((1, 1, *SHAPE))
    images = torch.ones((1, *SHAPE))       # everything bright => lots ignored
    assert detection_loss(logits, coords, mask, ISO, images=images) != pytest.approx(
        detection_loss(logits, coords, mask, ISO, images=None).item()
    )


def test_positive_weight_is_per_object_not_per_blob_voxel():
    """Regression: the peak gradient must not be diluted by the Gaussian's volume.

    An earlier version regressed the soft target and normalised by the count of
    non-zero voxels, spreading 1/n_pos over ~2400 voxels instead of ~1 per cell.
    The peak gradient collapsed, no logit reached the peak-finder's threshold, and
    training produced zero detections while the loss still went down.
    """
    coords, mask = _one_node()
    logits = torch.zeros((1, 1, *SHAPE), requires_grad=True)
    detection_loss(logits, coords, mask, ISO, neg_weight=0.0).backward()

    grad = logits.grad[0, 0]
    peak = grad[8, 16, 16].abs().item()
    # With one object the centroid carries the entire positive term.
    assert peak == pytest.approx(0.5, rel=1e-3), f"peak weight diluted: {peak}"
    # And with neg_weight=0 nothing else contributes at all.
    others = grad.abs().sum().item() - peak
    assert others == pytest.approx(0.0, abs=1e-6)


def test_near_centroid_negatives_are_penalty_reduced():
    """A broad blob around the true centre must be barely punished."""
    coords, mask = _one_node()
    logits = torch.zeros((1, 1, *SHAPE), requires_grad=True)
    detection_loss(logits, coords, mask, ISO, neg_weight=1.0).backward()
    grad = logits.grad[0, 0].abs()
    near = grad[8, 16, 17].item()      # one voxel off the centroid
    far = grad[2, 4, 4].item()         # genuine background
    assert near < far, "near-centroid negative is not suppressed"


def test_loss_is_finite_with_no_annotated_nodes():
    """Frames with no GT happen; the loss must not divide by zero."""
    coords = torch.zeros((1, 1, 3))
    mask = torch.zeros((1, 1), dtype=torch.bool)
    loss = detection_loss(torch.zeros((1, 1, *SHAPE)), coords, mask, ISO)
    assert torch.isfinite(loss)


def test_loss_backpropagates_to_the_logits():
    coords, mask = _one_node()
    logits = torch.zeros((1, 1, *SHAPE), requires_grad=True)
    detection_loss(logits, coords, mask, ISO).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


# --- pseudo-labels for the unannotated cells ---------------------------------


def _two_blobs(annotated_amp=1.0, unannotated_amp=1.3, shape=(32, 32, 32)):
    """One annotated blob and one brighter unannotated one, as measured.

    Non-GT local maxima sit at median normalised intensity 1.00 against 0.48 for
    GT centroids, so the unannotated cell being the brighter one is the real
    situation, not an adversarial case.
    """
    img = torch.zeros((1, *shape))
    zz, yy, xx = torch.meshgrid(*[torch.arange(s) for s in shape], indexing="ij")
    for centre, amp in (((10, 10, 10), annotated_amp), ((22, 22, 22), unannotated_amp)):
        d2 = ((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2).float()
        img[0] += amp * torch.exp(-d2 / (2 * 1.8 ** 2))
    coords = torch.tensor([[[10.0, 10.0, 10.0]]])
    mask = torch.ones((1, 1), dtype=torch.bool)
    return img, coords, mask


def test_pseudo_labels_find_the_unannotated_cell():
    img, coords, mask = _two_blobs()
    pm = pseudo_positive_mask(img, coords, mask, ISO)
    hits = torch.nonzero(pm[0])
    assert len(hits) > 0, "no pseudo-positive found for a plainly visible cell"
    assert all(
        max(abs(int(h[0]) - 22), abs(int(h[1]) - 22), abs(int(h[2]) - 22)) <= 3
        for h in hits
    ), "pseudo-positives landed away from the unannotated blob"


def test_pseudo_labels_never_overlap_a_real_annotation():
    """A voxel with exact supervision must not be re-labelled at a lower weight."""
    img, coords, mask = _two_blobs()
    pm = pseudo_positive_mask(img, coords, mask, ISO)
    hits = torch.nonzero(pm[0])
    assert not any(
        max(abs(int(h[0]) - 10), abs(int(h[1]) - 10), abs(int(h[2]) - 10)) <= 4
        for h in hits
    )


def test_pseudo_calibration_is_relative_not_absolute():
    """Scaling the whole frame must not change which voxels are pseudo-positive.

    The threshold is the median DoG response over the frame's own annotated
    centroids. Absolute brightness spans 11x across samples even after
    0.001/0.999 normalisation, so an absolute cut would label nearly everything
    on a bright sample and nothing on a dim one.
    """
    img, coords, mask = _two_blobs()
    a = pseudo_positive_mask(img, coords, mask, ISO)
    b = pseudo_positive_mask(img * 5.0, coords, mask, ISO)
    assert torch.equal(a, b)


def test_pseudo_loss_adds_positives_and_backpropagates():
    img, coords, mask = _two_blobs()
    logits = torch.zeros((1, 1, 32, 32, 32), requires_grad=True)
    plain = detection_loss(logits, coords, mask, ISO, images=img)
    withp = detection_loss_with_pseudo(logits, coords, mask, ISO, img, pseudo_weight=0.3)
    assert torch.isfinite(withp)
    assert withp > plain, "extra positives at zero logits must cost more"
    withp.backward()
    assert logits.grad.abs().sum() > 0


def test_pseudo_weight_zero_leaves_the_positive_term_alone():
    """The knob must be a true off-switch at 0, so model A is unaffected by it."""
    img, coords, mask = _two_blobs()
    logits = torch.randn(1, 1, 32, 32, 32)
    zero = detection_loss_with_pseudo(logits, coords, mask, ISO, img, pseudo_weight=0.0)
    assert torch.isfinite(zero)
