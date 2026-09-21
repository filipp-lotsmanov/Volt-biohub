"""The reference inference path, verbatim, with tracksdata removed.

**Why a copy instead of the rewrite we already had.** ``kaggle/submit.py``
reimplements this and loses ~0.04 of score, which is now measured twice. On
2026-09-14 the gap was 0.129 and ``CLAUDE.md`` concluded "stop maintaining the
rewrite; ship the path that already works"; we rebuilt the rewrite instead. On
2026-09-21, on the four movies the public notebooks validate on and with our
own scorer, ``scripts/predict_unet_transformer.py`` reaches adjusted edge
Jaccard **0.9299** where our reimplementation of the same three stages reaches
**0.8600**, and four separate attempts to close it -- an ILP over the resolved
graph, real consecutive frame pairs for the trunk, the reference's peak window,
and all three together -- moved it by at most 0.009. The bodies below are
copied unchanged from ``scripts/train_unet_transformer.py`` and
``scripts/predict_unet_transformer.py``, which are themselves byte-identical to
the upstream baseline apart from docstrings, so there is nothing left to get
subtly wrong.

Three things are replaced, all of them plumbing:

* ``io.open_dataset`` becomes :func:`_open_meta`, four attributes read with
  plain zarr, because ``volt_biohub.io`` imports tracksdata at module scope and
  the inference path must run without it.
* ``tqdm`` is dropped.
* The ILP is not here. It belongs to the caller, and
  :func:`volt_biohub.ilp_link.prune_dp` solves the same problem exactly without
  a solver -- so the whole path, ILP included, stays free of tracksdata.

The contract with the caller is deliberately narrow: :func:`predict_video`
returns ``(coords, edges)`` exactly as the reference does -- coords ``(N, 4)``
as ``[t, z, y, x]`` in **original** voxel space, edges as
``(src_index, tgt_index, prob, distance)`` -- and
:func:`to_per_frame` converts that to the ``(coords_per_t, links_per_t)`` pair
on the downsampled grid that ``volt_biohub.postprocess`` and the submission
writer consume.

``tests/test_reference_predict.py`` pins the copy against its source: the two
must produce identical detections and edges on the shared fixture, and if
someone edits one of the three originals, that test is what notices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch import nn

from volt_biohub.models import TemporalUNet3D
from volt_biohub.models.simple_node_transformer import SimpleNodeTransformer

_POS_EMBED_DIM = 8   # per axis; total = 4 axes x _POS_EMBED_DIM = 32

_DEFAULT_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
}


@dataclass
class _Meta:
    quantiles: dict
    zarr_path: Path
    image_shape: tuple[int, ...]
    scale: tuple[float, float, float]


def _open_meta(ds_path: Path, downsample: tuple[int, ...]) -> _Meta:
    """``open_dataset(load_image=False)``'s four fields, without tracksdata.

    ``image_shape`` is the **downsampled** shape, matching ``io.open_dataset``
    -- the reference normalises positional embeddings by it, so getting this
    wrong silently changes what the edge head is asked.
    """
    g = zarr.open_group(str(ds_path), mode="r")
    attrs = dict(g.attrs)
    scale = (1.0, 1.0, 1.0)
    for ms in attrs.get("multiscales", []) or []:
        for d in ms.get("datasets", []) or []:
            for tf in d.get("coordinateTransformations", []) or []:
                if tf.get("type") == "scale":
                    scale = tuple(float(v) for v in tf["scale"][1:])
    raw = tuple(g["0"].shape)
    shape = raw[:1] + tuple(-(-s // d) for s, d in zip(raw[1:], downsample, strict=True))
    return _Meta(quantiles=attrs.get("image_statistics", {}).get("quantiles", {}),
                 zarr_path=ds_path, image_shape=shape, scale=scale)


def to_per_frame(coords: np.ndarray, edges, downsample: tuple[int, int, int],
                 n_frames: int, iso_um: float):
    """``(coords, edges)`` -> the ``(coords_per_t, links_per_t, probs)`` triple.

    Coordinates come back on the **downsampled** grid, which is the space
    ``volt_biohub.postprocess`` measures distances in (multiplying by
    *iso_um*), and the submission writer multiplies back out by *downsample*.
    The probabilities ride alongside the links because
    :func:`volt_biohub.ilp_link.prune_dp` prices an edge by its probability.
    """
    down = np.asarray(downsample, dtype=np.float64)
    coords_per_t: list[np.ndarray] = []
    local: dict[int, int] = {}
    for t in range(n_frames):
        sel = np.nonzero(coords[:, 0].astype(int) == t)[0] if len(coords) else np.empty(0, int)
        for k, gi in enumerate(sel):
            local[int(gi)] = k
        coords_per_t.append(coords[sel, 1:].astype(np.float64) / down
                            if len(sel) else np.empty((0, 3)))
    links: list[list[tuple[int, int, float]]] = [[] for _ in range(max(0, n_frames - 1))]
    probs: list[dict[tuple[int, int], float]] = [{} for _ in range(max(0, n_frames - 1))]
    for gi, gj, prob, _dist in edges:
        t = int(coords[gi, 0])
        if int(coords[gj, 0]) != t + 1 or t >= len(links):
            continue
        a, b = local[int(gi)], local[int(gj)]
        d = float(np.linalg.norm((coords_per_t[t][a] - coords_per_t[t + 1][b]) * iso_um))
        links[t].append((a, b, d))
        probs[t][(a, b)] = float(prob)
    return coords_per_t, links, probs


def probs_as_matrices(coords_per_t, probs):
    """Sparse per-pair probabilities as the dense matrices ``prune_dp`` expects."""
    out = []
    for t, d in enumerate(probs):
        m = np.zeros((len(coords_per_t[t]), len(coords_per_t[t + 1])), dtype=np.float32)
        for (a, b), p in d.items():
            m[a, b] = p
        out.append(m)
    return out


def extract_pos_features(
    coords: np.ndarray,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = _POS_EMBED_DIM,
) -> np.ndarray:
    """Sinusoidal positional embeddings for node coordinates (no intensity term).

    Parameters
    ----------
    coords : np.ndarray
        (N, 4) with columns [t, z, y, x].
    image_shape : tuple
        Full image shape (T, Z, Y, X) used for normalisation.
    pos_embed_dim : int
        Embedding width per axis: ``pos_embed_dim // 2`` frequencies, each
        contributing a sin and a cos, so the output is ``4 * pos_embed_dim``.

    Returns
    -------
    np.ndarray
        Shape (N, 4 * pos_embed_dim), float32.
    """
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    # strict: a 3-length image_shape would silently drop x and normalise the
    # wrong axes -- a coordinate-space error that raises nothing and scores zero.
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape, strict=True)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(pos_embed_dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


class UNetNodeTransformer(nn.Module):
    """TemporalUNet3D encoder + SimpleNodeTransformer edge predictor.

    Forward pass:
      1. Stack frames t and t+1 → (B, 2, 1, *spatial) → UNet → (B, 2, C_feat, *spatial)
      2. Integer-index feature maps at node coords (round + clamp; differentiable)
      3. Concatenate with sinusoidal positional embeddings
      4. Cross-attention transformer → (B, max_nodes, max_nodes) edge logits
    """

    def __init__(
        self,
        unet: nn.Module,
        unet_out_channels: int,
        pos_feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.unet = unet
        self.unet_out_channels = unet_out_channels

        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)

        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def _index_features(
        self,
        feat_maps: torch.Tensor,  # (B, C, *spatial)
        coords: torch.Tensor,     # (B, max_nodes, 3)
        mask: torch.Tensor,       # (B, max_nodes) bool
    ) -> torch.Tensor:
        """Integer-index feat_maps at node positions; padded slots → zeros.

        Gradients flow through the *feature map values* but NOT through the
        coordinates (integer indexing is non-differentiable w.r.t. position).
        """
        B, C = feat_maps.shape[:2]
        spatial = feat_maps.shape[2:]
        max_nodes = coords.shape[1]

        out = torch.zeros(B, max_nodes, C, device=feat_maps.device, dtype=feat_maps.dtype)
        for b in range(B):
            nt = int(mask[b].sum().item())
            if nt == 0:
                continue
            z = coords[b, :nt, 0].long().clamp(0, spatial[0] - 1)
            y = coords[b, :nt, 1].long().clamp(0, spatial[1] - 1)
            x = coords[b, :nt, 2].long().clamp(0, spatial[2] - 1)
            out[b, :nt] = feat_maps[b, :, z, y, x].T
        return out

    def detect(
        self,
        frame: torch.Tensor,  # (*spatial) — single pre-downsampled frame
    ) -> torch.Tensor:
        """Run UNet + detection head on a single frame.

        Returns
        -------
        torch.Tensor
            (*spatial) detection logits at the input (already downsampled) resolution.
        """
        # Duplicate the frame into a fake pair so the temporal UNet can run.
        pair = torch.stack([frame, frame], dim=0).unsqueeze(0).unsqueeze(2)  # (1, 2, 1, *spatial)
        unet_out = self.unet(pair)          # (1, 2, C_feat, *spatial)
        det = self.detect_head(unet_out[0, 0:1])  # (1, 1, *spatial)
        return det[0, 0]  # (*spatial)

    def encode(
        self,
        imgs: torch.Tensor,  # (B, W, *spatial) — already downsampled
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run UNet encoder on W pre-downsampled frames.

        Returns ``(unet_out, det_logits)`` where *unet_out* is
        ``(B, W, C_feat, *spatial)`` and *det_logits* is a list of W
        tensors each ``(B, 1, *spatial)``.
        """
        window = imgs.unsqueeze(2)  # (B, W, 1, *spatial)
        unet_out = self.unet(window)  # (B, W, C_feat, *spatial)
        W = unet_out.shape[1]
        det_logits = [self.detect_head(unet_out[:, i]) for i in range(W)]
        return unet_out, det_logits

    def predict_edges(
        self,
        unet_feat_src: torch.Tensor,  # (B, N_src, C_feat) pre-indexed
        unet_feat_tgt: torch.Tensor,  # (B, N_tgt, C_feat) pre-indexed
        coords_src: torch.Tensor,     # (B, N_src, 3)
        coords_tgt: torch.Tensor,     # (B, N_tgt, 3)
        pos_feat_src: torch.Tensor,   # (B, N_src, pos_feat_dim)
        pos_feat_tgt: torch.Tensor,   # (B, N_tgt, pos_feat_dim)
        mask_src: torch.Tensor,       # (B, N_src) bool
        mask_tgt: torch.Tensor,       # (B, N_tgt) bool
    ) -> torch.Tensor:
        """Run transformer edge predictor on pre-indexed UNet features."""
        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        return self.transformer(feat_src, feat_tgt, coords_src, coords_tgt, mask_src, mask_tgt)


# =============================================================================
# Detection loss
# =============================================================================


@dataclass
class PredictConfig:
    """All hyperparameters that can affect prediction quality / score.

    Detection
    ---------
    det_threshold : float
        Minimum sigmoid probability for a local-max peak to be kept.

    Edge filtering
    --------------
    edge_activation : str
        Activation applied to raw edge logits: ``"sigmoid"`` (independent
        per-edge scores) or ``"softmax"`` (row-normalised over t+1 nodes).
    threshold : float
        Minimum edge probability to consider a link at all.
    max_parents_per_node : int
        Maximum number of incoming edges per node (typically 1).
    max_children_per_node : int
        Maximum number of outgoing edges per node (1 = no divisions, 2 = divisions allowed).
    """
    # Detection
    det_threshold: float = 0.5
    det_tta: bool = True  # flip-xy TTA for detection logits
    pool_kernel_um: float = 3.0  # max-pool kernel size in µm for detection peak extraction
    # Edge filtering
    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5

    # ILP post-processing
    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None

    def __post_init__(self) -> None:
        # When ILP is enabled it handles parent/children constraints itself,
        # so greedy limits are left unconstrained (None).  When ILP is
        # disabled, default to 1/1 to avoid unconstrained edge assignment.
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2



def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    """Reconstruct UNetNodeTransformer from saved config + weights.

    Reads ``config.json`` from the same directory as the weights file.
    Falls back to ``_DEFAULT_CONFIG`` if the file is missing.

    Returns ``(model, window_size, downsample)``.
    """
    config_path = weights_path.parent / "config.json"
    if config_path.exists():
        config = {**_DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    else:
        print(f"Warning: config.json not found at {config_path}, using defaults.", flush=True)
        config = _DEFAULT_CONFIG

    # Support legacy configs that used "downsample_factor" (scalar).
    if "downsample_factor" in config and "downsample" not in config:
        df = config["downsample_factor"]
        config["downsample"] = [df, df, df]

    downsample = tuple(config["downsample"])

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config["window_size"], downsample


# =============================================================================
# Per-frame loading
# =============================================================================


def _load_frame(
    zarr_arr,
    t: int,
    target_shape: list[int],
    downsample: tuple[int, ...] = (1, 1, 1),
) -> torch.Tensor:
    """Load one frame from zarr with strided spatial downsample (no normalisation)."""
    dz, dy, dx = downsample
    raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None], size=target_shape,
            mode="trilinear", align_corners=False,
        )[0, 0]
    return frame


# =============================================================================
# Inference
# =============================================================================


def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel.

    Each axis gets ``round(um / voxel_size_axis)`` voxels, forced to odd
    (for symmetric padding) and at least 1.

    Parameters
    ----------
    um : float
        Desired suppression distance in microns.
    voxel_size : tuple[float, ...]
        Per-axis voxel sizes in microns, e.g. ``(1.625, 0.40625, 0.40625)``.
    """
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def _detect_cells_pooled(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
) -> np.ndarray:
    """Extract cell coordinates via max-pool local-max (same as training).

    Coordinates are returned in the downsampled grid.  The caller is
    responsible for scaling back to original resolution if needed.

    Parameters
    ----------
    det_logits : torch.Tensor
        (1, Z, Y, X) raw logits.
    t : int
        Time index to prepend as the first column.
    det_threshold : float
        Minimum sigmoid probability for a peak to be considered (default 0.5).
    pool_kernel : tuple[int, ...]
        Per-axis kernel size for local-max pooling,
        e.g. ``(3, 11, 11)`` for anisotropic data.

    Returns
    -------
    np.ndarray
        (N, 4) int16 array with columns [t, z, y, x] in downsampled space.
    """
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


@torch.no_grad()


def predict_video(
    model: UNetNodeTransformer,
    ds_path: Path,
    device: torch.device,
    cfg: PredictConfig,
    window_size: int = 2,
    max_frames: int | None = None,
    unet_batch_size: int = 4,
    downsample: tuple[int, ...] = (1, 4, 4),
    blend=None,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Run inference on a single video using sliding windows of W frames.

    Windows slide with stride ``W - 1`` so every consecutive pair is covered
    exactly once.  UNet features from each window are reused for edge
    prediction on all ``W - 1`` consecutive pairs within the window.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples
    """
    ds = _open_meta(ds_path, downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    image_shape = (T, *ds.image_shape[1:])
    target_shape = list(image_shape[1:])

    ds_arr = np.array(downsample, dtype=np.float32)  # for coord rescaling at the end
    ds_arr_t = torch.from_numpy(ds_arr).to(device)   # for predict_edges (original-space coords)
    W = window_size
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample, strict=True))
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)

    # Running node registry — each entry records the frame-t detections.
    # coord_offset[t] = (start, end) half-open range into the stacked array.
    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count: int = 0
    all_edges: list[tuple[int, int, float, float]] = []

    # Sliding windows with stride W-1 cover every consecutive pair exactly once.
    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    # Ensure the very last pair (T-2 → T-1) is covered.
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for ws in window_starts:
        frame_indices = list(range(ws, ws + W))

        # --- UNet encode (single window, batch_size=1) ---
        imgs = torch.stack([
            _load_frame(zarr_arr, t, target_shape, downsample)
            for t in frame_indices
        ])  # (W, *spatial)
        # Quantile normalisation (0.1%–99.9%) to match training pipeline.
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        imgs = imgs.unsqueeze(0).to(device)   # (1, W, *spatial)

        unet_out, det_logits = model.encode(imgs)
        # unet_out: (1, W, C, *spatial_down), det_logits: list of W × (1, 1, *spatial_down)

        # Detection TTA: original + flip-x + flip-y + flip-xy, average logits.
        # TTA: flip along Y (-2) and X (-1) only.  Z is excluded because
        # the data is highly anisotropic (Z resolution ~4x coarser than XY),
        # so Z-flips would produce out-of-distribution inputs.
        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4

        # The one addition to the copied body, and it is a no-op unless a
        # second checkpoint is passed. Two independently trained detectors have
        # no reason to share a logit scale, so the secondary is moment-matched
        # to the primary before averaging, with the scale ratio clamped the way
        # the reference clamps it and a per-frame retention guard that reverts
        # to the primary if the blend loses more than 10% of its candidates --
        # losing node recall is the one failure this pipeline cannot absorb.
        # Measured through our own detect-and-link at weight 0.3: fold 0
        # 0.8953 -> 0.8986, fold 1 0.8760 -> 0.8852.
        if blend is not None:
            for f in range(W):
                det_logits[f] = blend(det_logits[f], imgs, f, cfg.det_threshold)

        del imgs

        # --- Detect cells in each frame (dedup across windows) ---
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                arr = _detect_cells_pooled(
                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,
                )
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
                global_node_count += len(arr)
                coord_lists.append(arr)
                seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        )

        # --- Edge prediction for each consecutive pair in the window ---
        for f_idx in range(W - 1):
            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            if t_src not in coord_offset or t_tgt not in coord_offset:
                continue
            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]
            if e_src == s_src or e_tgt == s_tgt:
                continue

            c_src = coords_so_far[s_src:e_src]
            c_tgt = coords_so_far[s_tgt:e_tgt]
            n_src, n_tgt = len(c_src), len(c_tgt)
            idx_src = np.arange(s_src, e_src, dtype=np.int64)
            idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

            # Build tensors (batch_size=1).
            p_coords_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            # Use window-relative time (f_idx, f_idx+1) normalised by W, not absolute frame index.
            window_shape = (W, *image_shape[1:])
            c_src_rel = c_src.copy()
            c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt.copy()
            c_tgt_rel[:, 0] = f_idx + 1
            p_pos_src = torch.from_numpy(extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)
            p_pos_tgt = torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)
            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

            unet_feat_src = model._index_features(
                unet_out[:, f_idx], p_coords_src, p_mask_src,
            )
            unet_feat_tgt = model._index_features(
                unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
            )
            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)

            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )

            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}

            for prob, i, j in candidates:
                n_ch = children_count.get(i, 0)
                n_pa = parents_count.get(j, 0)
                if cfg.max_children_per_node is not None and n_ch >= cfg.max_children_per_node:
                    continue
                if cfg.max_parents_per_node is not None and n_pa >= cfg.max_parents_per_node:
                    continue

                gi, gj = int(idx_src[i]), int(idx_tgt[j])
                dist = float(np.linalg.norm(
                    coords_so_far[gi, 1:].astype(np.float32)
                    - coords_so_far[gj, 1:].astype(np.float32)
                ))
                all_edges.append((gi, gj, float(prob), dist))
                children_count[i] = n_ch + 1
                parents_count[j] = n_pa + 1

        del unet_out

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    # Scale spatial coords back to original resolution.
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16)
    return coords, all_edges


# =============================================================================
# Prediction loop
# =============================================================================


def make_blend(secondary_path: Path, device, weight: float,
               retention: float = 0.90):
    """A ``blend`` callable for :func:`predict_video`, or ``None`` at weight 0.

    Kept here rather than in ``kaggle/submit.py`` so the reference path owns
    its own optional second detector and the two cannot drift: ``submit.py``'s
    version blends a single duplicated frame, this one blends whichever frame
    of the window the caller names, which is what a temporal trunk needs.

    The guard is what makes it safe rather than merely plausible. A blend that
    loses detections costs node recall, and node recall is the term this
    pipeline can least afford -- it is what a misconfigured ILP cost us
    (0.9419 against 0.9830). If the blended map yields fewer than *retention*
    of the primary's candidates on a frame, that frame keeps the primary.
    """
    import math

    if weight <= 0.0:
        return None
    cfg = json.loads((secondary_path.parent / "config.json").read_text())
    unet2 = TemporalUNet3D(in_channels=1, out_channels=cfg["unet_out_channels"],
                           layers=cfg["unet_layers"])
    head2 = nn.Conv3d(cfg["unet_out_channels"], 1, kernel_size=1)
    state = torch.load(secondary_path, map_location="cpu", weights_only=True)
    unet2.load_state_dict({k[len("unet."):]: v for k, v in state.items()
                           if k.startswith("unet.")})
    head2.load_state_dict({k[len("detect_head."):]: v for k, v in state.items()
                           if k.startswith("detect_head.")})
    unet2.to(device).eval()
    head2.to(device).eval()

    def _blend(primary, imgs, f, det_threshold):
        with torch.no_grad():
            feats = unet2(imgs.unsqueeze(2))
            out2 = head2(feats[:, f])
        p_mean, p_std = primary.mean(), primary.std()
        s_mean, s_std = out2.mean(), out2.std()
        ratio = torch.clamp(p_std / (s_std + 1e-6), 0.5, 2.0)
        aligned = (out2 - s_mean) * ratio + p_mean
        blended = (1.0 - weight) * primary + weight * aligned
        cut = math.log(det_threshold / (1.0 - det_threshold))
        n_p = int((primary > cut).sum())
        n_b = int((blended > cut).sum())
        if n_p > 0 and n_b < retention * n_p:
            return primary
        return blended

    return _blend
