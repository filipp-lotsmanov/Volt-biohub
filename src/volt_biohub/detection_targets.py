"""Detection targets and loss for sparsely annotated 3D cell volumes.

The inherited loss marked a **single voxel** per GT cell positive and penalised
every other voxel at ``neg_weight``. Two things are wrong with that, and both are
measured rather than assumed (numbers from the full training set, 2026-09-02):

1. **Ground truth is ~3.5 % annotated at the median**, so ~96 % of the real cells
   in every frame were trained as background. Worse, local maxima that carry *no*
   GT annotation are **brighter** than the annotated centroids (median normalised
   intensity 1.00 vs 0.48) — the loss was actively teaching the network that the
   brightest blobs in the image are not cells. That is why inference needed
   ``--det-threshold 0.99``: the detector was miscalibrated by construction.

   The right framing is **positive-unlabeled learning** — an unannotated cell is
   *unlabeled*, not negative. ``ignore_mask`` drops the negative term on voxels
   that are bright *and* far from any GT centroid.

2. **A single hot voxel in 262,144 gives the network no gradient in a
   neighbourhood**, and every voxel just off-centre is punished for being right.
   ``gaussian_targets`` builds a Gaussian per cell whose width comes from the
   measured nucleus size: radial intensity profiles around 208 GT centroids give
   **FWHM 6.9 µm**, hence sigma ~2.93 µm. At the *minimum* observed cell pitch
   (11.3 µm) two such targets stay clearly bimodal, so peaks remain separable;
   the profile is flat beyond ~7 µm, which is also where a voxel stops plausibly
   belonging to the annotated nucleus.

   The Gaussian is applied as a **negative-suppression weight** rather than a
   regression target — see :func:`detection_loss`. Regressing it directly and
   normalising by the number of non-zero voxels dilutes the peak gradient a few
   hundred-fold and the detector then emits nothing at all; that was measured on
   the fixture, not theorised.

Everything works in **µm**, never voxels: the raw grid is 4× anisotropic in z, so
a radius in voxels would be wrong by 4× in one axis. ``voxel_size_um`` is passed
in rather than assumed, because the training grid is only isotropic under the
default ``--downsample 1,4,4``.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

# --- Measured constants (see module docstring; scripts/inspect_data.py) --------
NUCLEUS_FWHM_UM = 6.9
"""Nucleus diameter at half maximum, from radial profiles around GT centroids."""

DEFAULT_SIGMA_UM = NUCLEUS_FWHM_UM / 2.3548
"""Sigma of a Gaussian with the measured FWHM (2.3548 = 2*sqrt(2*ln2)); ~2.93 µm."""

DEFAULT_IGNORE_RADIUS_UM = 7.0
"""Beyond this, a bright voxel is treated as an unlabeled cell rather than
background. Chosen where the measured radial profile flattens."""

DEFAULT_BRIGHT_THRESHOLD = 0.3
"""Normalised intensity above which a voxel might be a cell. Sits below 90.9 % of
GT centroids and above ~95 % of non-GT local maxima."""

_TRUNCATE_SIGMA = 3.0
"""Gaussians are stamped out to this many sigma; beyond it the target is <1.2 %."""


def _offset_grid(
    half: tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """All integer voxel offsets within *half* of the origin, shape (K, 3)."""
    axes = [torch.arange(-h, h + 1, device=device) for h in half]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def _stamp(
    coords: torch.Tensor,
    mask: torch.Tensor,
    shape: tuple[int, int, int],
    voxel_size_um: tuple[float, float, float],
    radius_um: float,
    values_fn,
) -> torch.Tensor:
    """Scatter a per-node kernel into a (B, *shape) volume, combining by max.

    Max rather than sum: overlapping cells must not push the target above 1,
    which would make BCE unsatisfiable and bias the logits high everywhere two
    cells are close. A shared helper because the Gaussian target and the
    ignore-ball differ only in *values_fn*.
    """
    B = coords.shape[0]
    device = coords.device
    vox = torch.as_tensor(voxel_size_um, dtype=torch.float32, device=device)

    half = tuple(int(max(1, torch.ceil(torch.tensor(radius_um) / v).item())) for v in vox)
    offs = _offset_grid(half, device)                          # (K, 3)
    dist_um = torch.linalg.vector_norm(offs.float() * vox, dim=-1)   # (K,)
    vals = values_fn(dist_um)                                  # (K,)

    # (B, N, K, 3) absolute voxel positions of every stamp element.
    pos = coords.round().long().unsqueeze(2) + offs.view(1, 1, -1, 3)
    inside = torch.ones(pos.shape[:3], dtype=torch.bool, device=device)
    for ax, extent in enumerate(shape):
        inside &= (pos[..., ax] >= 0) & (pos[..., ax] < extent)
    keep = inside & mask.unsqueeze(-1) & (vals.view(1, 1, -1) > 0)

    out = torch.zeros((B, *shape), dtype=torch.float32, device=device)
    if not keep.any():
        return out

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(inside)[keep]
    z, y, x = (pos[..., ax][keep] for ax in range(3))
    flat = ((b_idx * shape[0] + z) * shape[1] + y) * shape[2] + x
    src = vals.view(1, 1, -1).expand_as(inside)[keep]

    out.view(-1).scatter_reduce_(0, flat, src, reduce="amax")
    return out


def gaussian_targets(
    coords: torch.Tensor,
    mask: torch.Tensor,
    shape: tuple[int, int, int],
    voxel_size_um: tuple[float, float, float],
    sigma_um: float = DEFAULT_SIGMA_UM,
    warn_on_collapse: bool = True,
) -> torch.Tensor:
    """Soft detection target: a Gaussian of width *sigma_um* at each GT centroid.

    Parameters
    ----------
    coords : torch.Tensor
        (B, N, 3) GT centroids in *this grid's* voxel coordinates.
    mask : torch.Tensor
        (B, N) bool, True for real (non-padded) nodes.
    shape : tuple of int
        (Z, Y, X) of the target volume.
    voxel_size_um : tuple of float
        Physical size of one voxel on this grid, (z, y, x).
    sigma_um : float
        Gaussian σ in µm. Defaults to the measured nucleus size.
    warn_on_collapse : bool
        Warn when two GT centroids land on the same voxel — after downsampling
        those cells are individually undetectable, and silently training a single
        blob for two cells hides a real recall ceiling.

    Returns
    -------
    torch.Tensor
        (B, *shape) float32 in [0, 1], peaking at 1.0 on each centroid.
    """
    if warn_on_collapse:
        _warn_collapsed(coords, mask)

    two_sig_sq = 2.0 * sigma_um * sigma_um
    return _stamp(
        coords, mask, shape, voxel_size_um,
        radius_um=_TRUNCATE_SIGMA * sigma_um,
        values_fn=lambda d: torch.exp(-(d * d) / two_sig_sq),
    )


def ignore_mask(
    images: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    voxel_size_um: tuple[float, float, float],
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
    ignore_radius_um: float = DEFAULT_IGNORE_RADIUS_UM,
) -> torch.Tensor:
    """Voxels to exclude from the negative term: bright, and far from any GT node.

    "Far" is required as well as "bright" because the voxels *around* an annotated
    cell are genuine negatives — they are what teaches the detector to localise and
    to separate touching cells. Only bright material with no annotation nearby is
    ambiguous, and that is precisely an unlabeled cell.

    Parameters
    ----------
    images : torch.Tensor
        (B, *shape) normalised intensities, nominally in [0, 1].
    coords, mask : torch.Tensor
        As in :func:`gaussian_targets`.
    voxel_size_um : tuple of float
        Physical voxel size, (z, y, x).
    bright_threshold : float
        Normalised intensity above which a voxel might be a cell.
    ignore_radius_um : float
        Voxels within this distance of a GT centroid are never ignored.

    Returns
    -------
    torch.Tensor
        (B, *shape) bool, True where the voxel must not contribute a negative.
    """
    shape = tuple(images.shape[1:])
    near_gt = _stamp(
        coords, mask, shape, voxel_size_um,
        radius_um=ignore_radius_um,
        values_fn=lambda d: (d <= ignore_radius_um).float(),
    ) > 0
    return (images > bright_threshold) & ~near_gt


def _warn_collapsed(coords: torch.Tensor, mask: torch.Tensor) -> None:
    for b in range(coords.shape[0]):
        n = int(mask[b].sum().item())
        if n <= 1:
            continue
        c = coords[b, :n].round().long()
        n_unique = torch.unique(c, dim=0).shape[0]
        if n_unique < n:
            warnings.warn(
                f"Sample {b}: {n - n_unique}/{n} GT nodes collapsed to duplicate "
                f"voxels at this downsampling — those cells are undetectable.",
                stacklevel=3,
            )


PENALTY_REDUCTION_BETA = 4.0
"""Exponent suppressing the negative term near a centroid, as in CenterNet."""


def detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    voxel_size_um: tuple[float, float, float],
    images: torch.Tensor | None = None,
    neg_weight: float = 0.1,
    sigma_um: float = DEFAULT_SIGMA_UM,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
    ignore_radius_um: float = DEFAULT_IGNORE_RADIUS_UM,
    beta: float = PENALTY_REDUCTION_BETA,
) -> torch.Tensor:
    """Penalty-reduced BCE: hard positive at each centroid, softened around it.

    The Gaussian enters as a **negative-suppression weight**, not a regression
    target, which is the CenterNet formulation and matters for a concrete reason.
    Regressing the soft target directly and normalising by the count of non-zero
    voxels spreads ``1/n_pos`` over the whole blob (~2400 voxels here instead of
    ~6), which weakens the peak gradient a few hundred-fold; the detector then
    never pushes any logit past the peak-finder's threshold and produces no
    detections at all. Measured, not hypothesised.

    So the three regimes are:

    - **The centroid voxel** is a hard positive at ``1/n_objects`` — exactly the
      weight the single-voxel loss gave it, which is what keeps *neg_weight*
      comparable across the two and makes the ablation a fair test.
    - **Voxels near a centroid** stay negatives but are scaled by
      ``(1 - target) ** beta``, so a broad, smooth blob around the true centre is
      barely penalised. That is what buys the gradient in a neighbourhood that a
      lone hot voxel could not provide.
    - **Bright voxels far from any annotation** are dropped entirely when
      *images* is given, because they are unlabeled cells rather than background.

    Parameters
    ----------
    det_logits : torch.Tensor
        (B, 1, Z, Y, X) raw logits from the detection head.
    coords : torch.Tensor
        (B, N, 3) GT centroids in this grid's voxel coordinates.
    mask : torch.Tensor
        (B, N) bool, True for real nodes.
    voxel_size_um : tuple of float
        Physical voxel size on this grid, (z, y, x).
    images : torch.Tensor, optional
        (B, Z, Y, X) normalised intensities enabling positive-unlabeled masking.
        None reproduces full negative coverage, which is the
        ``--no-det-pu-mask`` ablation.
    neg_weight : float
        Relative weight of the negative term.
    sigma_um, bright_threshold, ignore_radius_um : float
        See :func:`gaussian_targets` and :func:`ignore_mask`.
    beta : float
        Penalty-reduction exponent for near-centroid negatives.

    Returns
    -------
    torch.Tensor
        Scalar loss, mean over the batch.
    """
    B = det_logits.shape[0]
    shape = tuple(det_logits.shape[2:])
    logits = det_logits[:, 0]

    target = gaussian_targets(coords, mask, shape, voxel_size_um, sigma_um)
    is_pos = target >= 1.0 - 1e-6          # the centroid voxels themselves

    keep_neg = ~is_pos
    if images is not None:
        keep_neg = keep_neg & ~ignore_mask(
            images, coords, mask, voxel_size_um, bright_threshold, ignore_radius_um
        )

    flat = (B, -1)
    n_pos = is_pos.reshape(flat).sum(1).clamp(min=1)
    n_neg = keep_neg.reshape(flat).sum(1).clamp(min=1)
    per_sample = (B,) + (1,) * len(shape)

    w_pos = (1.0 / n_pos).reshape(per_sample).expand_as(target)
    # (1 - target)**beta leaves far negatives at full weight and all but silences
    # the immediate neighbourhood of a centroid.
    w_neg = (neg_weight / n_neg).reshape(per_sample) * (1.0 - target).pow(beta)

    weight = torch.where(
        is_pos, w_pos, torch.where(keep_neg, w_neg, torch.zeros((), device=target.device))
    )
    hard_target = is_pos.to(target.dtype)
    return F.binary_cross_entropy_with_logits(
        logits, hard_target, weight=weight, reduction="sum"
    ) / B
