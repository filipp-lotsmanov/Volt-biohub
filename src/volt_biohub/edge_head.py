"""The trained edge head, on an inference path that cannot import tracksdata.

**This exists because we were discarding 28 % of the checkpoint.** The public
baseline checkpoint carries three groups of weights — `unet.*` (1,497,610
params), `detect_head.*` (33) and `transformer.*` (**580,353**) — and
`kaggle/submit.py`'s `load_detector` reads only the first two. The learned edge
scorer was trained, shipped, and thrown away at load time, while a geometric
Hungarian linker stood in for it.

The substitution was justified in `submit.py` and `detectors.py` by the claim
that the transformer "currently emits zero edges". That was measured on our own
12-epoch checkpoint and generalised to the architecture; with these weights it
emits 25k–68k edges per video and reaches edge Jaccard 1.000 on a video where
the Hungarian linker gets 0.636. Holding detections fixed, swapping our linker
for this one is worth **+0.073** — the largest single measured item left.

What the learned head has that assignment cannot: an **abstain option**. The
activation is `softmax` over the *source* axis, so at most one source per target
column can exceed the threshold, and when the model is unsure none do — the
target simply gets no parent. `scipy.optimize.linear_sum_assignment` has no such
move: every detection inside the gate is matched to something, and each wrong
match lands in the metric's denominator.

Ported rather than imported: the reference lives in
`scripts/train_unet_transformer.py`, which pulls in tracksdata through `io` and
`metrics` at module scope and therefore cannot load in the notebook. Only three
small pieces are needed, all pure torch/numpy, and `SimpleNodeTransformer`
itself is already tracksdata-free.
"""

from __future__ import annotations

import numpy as np

POS_EMBED_DIM = 8
"""Per axis; total width is 4 axes x 8 = 32, matching the trained checkpoint.
Changing it silently mismatches the transformer's input projection."""


def extract_pos_features(
    coords: np.ndarray,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = POS_EMBED_DIM,
) -> np.ndarray:
    """Sinusoidal positional embedding for ``(t, z, y, x)`` rows.

    `strict=True` on the zip is load-bearing: a 3-length *image_shape* would
    silently drop x and normalise the wrong axes, a coordinate-space error that
    raises nothing and scores zero.
    """
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape, strict=True)]

    def embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(pos_embed_dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([embed(v) for v in norms], axis=1).astype(np.float32)


def index_features(feat_map, coords):
    """Integer-index a ``(C, Z, Y, X)`` feature map at ``(N, 3)`` coordinates."""
    import torch

    if len(coords) == 0:
        return torch.zeros((0, feat_map.shape[0]), device=feat_map.device,
                           dtype=feat_map.dtype)
    spatial = feat_map.shape[1:]
    idx = [torch.as_tensor(coords[:, a], device=feat_map.device).long()
           .clamp(0, spatial[a] - 1) for a in range(3)]
    return feat_map[:, idx[0], idx[1], idx[2]].T


def load_edge_head(weights, config, device):
    """Rebuild the trained ``SimpleNodeTransformer``. Returns None if absent.

    Returns None rather than raising so a weights file without `transformer.*`
    degrades to the geometric linker instead of killing the run — the same
    contract `load_detector` has for a missing config.
    """
    import torch

    from volt_biohub.models import SimpleNodeTransformer

    state = torch.load(weights, map_location="cpu", weights_only=True)
    sub = {k[len("transformer."):]: v for k, v in state.items()
           if k.startswith("transformer.")}
    if not sub:
        return None

    feat_dim = config["unet_out_channels"] + 4 * POS_EMBED_DIM
    model = SimpleNodeTransformer(feat_dim=feat_dim)
    model.load_state_dict(sub)
    model.to(device).eval()
    return model


def pair_probs(
    edge_head,
    feat_src,
    feat_tgt,
    coords_src: np.ndarray,
    coords_tgt: np.ndarray,
    t_src: int,
    window_shape: tuple[int, ...],
    downsample: tuple[int, int, int],
    bidirectional_weight: float = 0.0,
) -> np.ndarray:
    """The raw ``n_src x n_tgt`` probability matrix, ungated and unresolved.

    Split out of :func:`link_pair` because *how* these probabilities are spent
    decides whether the head helps or hurts, and we measured both. Resolved
    greedily into edges (``link_pair``) the head scores **0.7632** on the full
    fold against the geometric linker's 0.7830: softmax over sources is an
    abstain mechanism, so when the model is unsure no source clears the
    threshold, the target keeps no parent, and pruning then deletes it — node
    recall falls 0.9808 -> 0.9390. Handed to a global solver instead, the same
    numbers are worth +0.037 in the reference's own ablation, because the solver
    decides node retention and topology *together* and an abstention becomes a
    disappearance cost rather than a deleted node.

    Time is expressed **window-relative** (``t_src``, ``t_src + 1``) and
    normalised by the window length, not by the absolute frame index, because
    that is how the positional embedding was fed during training.

    **Bidirectional fusion** (*bidirectional_weight* > 0) scores each pair a
    second time with the two frames swapped and combines the two by a weighted
    harmonic mean. The forward pass softmaxes over sources, so it answers
    "which parent for this child"; the swapped pass answers "which child for
    this parent". A harmonic mean is near zero unless *both* agree, which is the
    point — it is an agreement test, not an average. The reference measures it
    at **+0.0053** and reached it by *removing* an earlier hard guard that both
    directions must pick the same winner, in favour of this soft form.

    Why it is worth retrying here specifically: our detection-logit TTA lost
    because averaging over the D4 group blurs two daughters into one peak. This
    touches no detection and no geometry — it only sharpens the probabilities
    that the relink spends as a cost bonus, which is the one interface where
    this head has been measured to help.
    """
    import torch

    if len(coords_src) == 0 or len(coords_tgt) == 0:
        return np.zeros((len(coords_src), len(coords_tgt)), dtype=np.float32)

    dev = feat_src.device
    up = np.asarray(downsample, dtype=np.float32)

    def prep(coords, t):
        rel = np.concatenate(
            [np.full((len(coords), 1), t, dtype=np.float32),
             np.asarray(coords, dtype=np.float32)], axis=1)
        pos = extract_pos_features(rel, window_shape)
        return (torch.from_numpy(pos).unsqueeze(0).to(dev),
                torch.from_numpy(np.asarray(coords, dtype=np.float32) * up)
                .unsqueeze(0).to(dev))

    pos_s, xyz_s = prep(coords_src, t_src)
    pos_t, xyz_t = prep(coords_tgt, t_src + 1)
    f_s = index_features(feat_src, coords_src).unsqueeze(0)
    f_t = index_features(feat_tgt, coords_tgt).unsqueeze(0)
    m_s = torch.ones(1, len(coords_src), dtype=torch.bool, device=dev)
    m_t = torch.ones(1, len(coords_tgt), dtype=torch.bool, device=dev)

    with torch.no_grad():
        logits = edge_head(torch.cat([f_s, pos_s], dim=-1),
                           torch.cat([f_t, pos_t], dim=-1),
                           xyz_s, xyz_t, m_s, m_t)[0]
        # Softmax over sources (dim 0) is the abstain mechanism -- see module
        # docstring. A sigmoid here would score every pair independently and
        # give back exactly the over-linking the Hungarian linker suffers from.
        probs = torch.softmax(logits, dim=0).float().cpu().numpy()

        if bidirectional_weight > 0.0:
            # Swap the roles: the later frame is fed as the "source". Its
            # softmax over dim 0 then normalises over the t+1 nodes for each t
            # node, i.e. P(child | parent), and transposing puts it back in
            # (src, tgt) orientation to pair with P(parent | child).
            rev = edge_head(torch.cat([f_t, pos_t], dim=-1),
                            torch.cat([f_s, pos_s], dim=-1),
                            xyz_t, xyz_s, m_t, m_s)[0]
            p_rev = torch.softmax(rev, dim=0).float().cpu().numpy().T
            w = bidirectional_weight
            # Weighted harmonic mean, guarded: either probability at exactly 0
            # makes the reciprocal infinite, and the fused value should be 0
            # there anyway (no agreement).
            both = (probs > 0) & (p_rev > 0)
            fused = np.zeros_like(probs)
            fused[both] = 1.0 / ((1.0 - w) / probs[both] + w / p_rev[both])
            probs = fused
    return probs


def link_pair(
    edge_head,
    feat_src,
    feat_tgt,
    coords_src: np.ndarray,
    coords_tgt: np.ndarray,
    t_src: int,
    window_shape: tuple[int, ...],
    downsample: tuple[int, int, int],
    threshold: float = 0.5,
    max_parents: int = 1,
    max_children: int = 2,
) -> list[tuple[int, int, float]]:
    """Learned links between two frames, as ``(i, j, distance)`` triples.

    The reference's edge stage: keep pairs above *threshold*, then take
    candidates strongest-first subject to at most one parent per child and two
    children per parent. Those caps are not stylistic — the scorer silently
    truncates out-degree above 2 to the two lowest edge ids, so enforcing it
    here is what keeps the choice ours.

    Measured worse than the geometric linker (0.7632 against 0.7830); see
    :func:`pair_probs` for why, and prefer feeding the probabilities to the ILP.
    """
    if len(coords_src) == 0 or len(coords_tgt) == 0:
        return []
    probs = pair_probs(edge_head, feat_src, feat_tgt, coords_src, coords_tgt,
                       t_src, window_shape, downsample)

    ii, jj = np.nonzero(probs > threshold)
    cand = sorted(((float(probs[i, j]), int(i), int(j)) for i, j in zip(ii, jj, strict=True)),
                  reverse=True)

    out: list[tuple[int, int, float]] = []
    kids: dict[int, int] = {}
    parents: dict[int, int] = {}
    scale = np.asarray(downsample, dtype=np.float64)
    for _p, i, j in cand:
        if kids.get(i, 0) >= max_children or parents.get(j, 0) >= max_parents:
            continue
        d = float(np.linalg.norm(
            (np.asarray(coords_src[i], dtype=np.float64)
             - np.asarray(coords_tgt[j], dtype=np.float64)) * scale))
        out.append((i, j, d))
        kids[i] = kids.get(i, 0) + 1
        parents[j] = parents.get(j, 0) + 1
    return out
