"""The copy must equal its original, or the copy is just another rewrite.

``src/volt_biohub/reference_predict.py`` exists because two successive
reimplementations of this inference path each lost ~0.04 of score. A verbatim
copy only buys anything while it stays verbatim, so this compares it against
``scripts/predict_unet_transformer.py`` on the shared fixture with a randomly
initialised model -- no trained weights needed, because the claim is that the
two produce the *same* detections and edges from the same parameters, not that
either is good.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from volt_biohub import reference_predict as rp

pytest.importorskip("tracksdata")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """A randomly initialised checkpoint in the layout ``load_model`` expects."""
    d = tmp_path_factory.mktemp("ckpt")
    cfg = {"unet_out_channels": 8, "unet_layers": [8, 16],
           "downsample": [1, 4, 4], "window_size": 2}
    (d / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    unet = rp.TemporalUNet3D(in_channels=1, out_channels=cfg["unet_out_channels"],
                             layers=cfg["unet_layers"])
    model = rp.UNetNodeTransformer(
        unet=unet, unet_out_channels=cfg["unet_out_channels"],
        pos_feat_dim=4 * rp._POS_EMBED_DIM)
    torch.save(model.state_dict(), d / "edge_predictor_best.pth")
    return d / "edge_predictor_best.pth"


@pytest.fixture(scope="module")
def clip(division_clip_fixture) -> Path:
    return Path(__file__).parent / "data" / "division_clip" / "division_clip.zarr"


def test_port_reproduces_the_reference_exactly(tiny_checkpoint, clip):
    import predict_unet_transformer as ref

    dev = torch.device("cpu")
    # A low threshold so the random model actually emits nodes and edges;
    # an all-empty comparison would pass without testing anything.
    ours_cfg = rp.PredictConfig(det_threshold=0.05, det_tta=True)
    theirs_cfg = ref.PredictConfig(det_threshold=0.05, det_tta=True)

    m1, w1, d1 = rp.load_model(tiny_checkpoint, dev)
    m2, w2, d2 = ref.load_model(tiny_checkpoint, dev)
    assert (w1, tuple(d1)) == (w2, tuple(d2))

    c1, e1 = rp.predict_video(m1, clip, dev, cfg=ours_cfg,
                              window_size=w1, downsample=d1)
    c2, e2 = ref.predict_video(m2, clip, dev, cfg=theirs_cfg,
                               window_size=w2, downsample=d2)

    assert len(c1) > 0, "fixture produced no detections; the test proves nothing"
    np.testing.assert_array_equal(c1, c2)
    assert [(a, b) for a, b, _p, _d in e1] == [(a, b) for a, b, _p, _d in e2]
    np.testing.assert_allclose([p for _a, _b, p, _d in e1],
                               [p for _a, _b, p, _d in e2], rtol=0, atol=0)


def test_open_meta_matches_open_dataset(clip):
    """The one field that was not copied: shape, scale and quantiles by zarr."""
    from volt_biohub.io import open_dataset

    down = (1, 4, 4)
    ours = rp._open_meta(clip, down)
    theirs = open_dataset(clip, normalize=False, load_image=False,
                          downsample=down)
    assert ours.image_shape == tuple(theirs.image_shape)
    assert ours.quantiles == theirs.quantiles
    np.testing.assert_allclose(ours.scale, theirs.scale)


def test_bidirectional_fusion_uses_the_reverse_pass_transposed(
    tiny_checkpoint, clip, monkeypatch,
):
    """The transpose is the whole bug class, and it is silent when n_src == n_tgt.

    A missing transpose fuses ``P(parent|child)`` with ``P(child|parent)`` read
    off the wrong axis -- a plausible-looking matrix that scrambles which pair
    got which score. So rather than compare against a second implementation of
    the same arithmetic, this pins ``predict_edges`` to a deterministic
    asymmetric function of its own shapes, which makes every fused probability
    predictable in closed form.
    """
    dev = torch.device("cpu")
    model, W, down = rp.load_model(tiny_checkpoint, dev)
    w = 0.35

    def fake_predict_edges(f_a, f_b, *_args, **_kw):
        n_a, n_b = f_a.shape[1], f_b.shape[1]
        i = torch.arange(n_a, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(n_b, dtype=torch.float32).unsqueeze(0)
        return (0.37 * i - 0.11 * j).unsqueeze(0)

    monkeypatch.setattr(model, "predict_edges", fake_predict_edges)
    cfg = rp.PredictConfig(det_threshold=0.05, det_tta=False, threshold=0.0)
    coords, edges = rp.predict_video(model, clip, dev, cfg=cfg,
                                     window_size=W, downsample=down,
                                     bidir=w)
    assert edges, "fixture emitted no edges; the test proves nothing"

    # Recover each edge's (pair, local i, local j) from the frame-ordered coords.
    starts: dict[int, int] = {}
    for k, t in enumerate(coords[:, 0].tolist()):
        starts.setdefault(int(t), k)

    seen_asymmetric = False
    for gi, gj, prob, _dist in edges:
        t_src = int(coords[gi, 0])
        i = gi - starts[t_src]
        j = gj - starts[t_src + 1]
        n_src = sum(1 for t in coords[:, 0] if int(t) == t_src)
        n_tgt = sum(1 for t in coords[:, 0] if int(t) == t_src + 1)
        seen_asymmetric |= n_src != n_tgt

        fwd = fake_predict_edges(torch.zeros(1, n_src, 1),
                                 torch.zeros(1, n_tgt, 1))[0]
        rev = fake_predict_edges(torch.zeros(1, n_tgt, 1),
                                 torch.zeros(1, n_src, 1))[0]
        p_f = torch.softmax(fwd, dim=0)[i, j].item()
        p_r = torch.softmax(rev, dim=0).transpose(0, 1)[i, j].item()
        expected = 1.0 / ((1.0 - w) / p_f + w / p_r)
        assert prob == pytest.approx(expected, rel=1e-6), (gi, gj)

    assert seen_asymmetric, "every pair was square; the transpose went untested"


def test_secondary_ensembled_with_itself_is_a_no_op(tiny_checkpoint, clip):
    """A model averaged with itself must change nothing, in either half.

    This is the test that catches a mis-indexed feature lookup. ``_pair_probs``
    has to index the secondary trunk at ``f_idx`` for sources and ``f_idx + 1``
    for targets; off by one, or with the two swapped, it still returns a
    plausible matrix of the right shape and the run completes. Passing the same
    checkpoint as both primary and secondary makes the correct answer exactly
    the unensembled one, because averaging a probability matrix with itself is
    the identity -- so any indexing error shows up as a difference.

    ``det_weight=0`` is deliberate, and the reason is worth recording: the
    detection blend is **not** exactly identity against itself. Its scale ratio
    is ``p_std / (s_std + 1e-6)``, which for identical maps is marginally below
    1, so the blended map is shifted by ~1e-6 relative -- enough to flip a
    local-max tie and move an integer coordinate. That is a property of the
    moment matching rather than a bug, but it makes a whole-pipeline identity
    assertion untestable, so this pins the half that does have an exact answer.
    """
    dev = torch.device("cpu")
    # threshold=0.0 so the random model actually emits edges; at the default
    # 0.5 a source-softmax over many candidates clears nothing and the
    # comparison would be between two empty lists.
    cfg = rp.PredictConfig(det_threshold=0.05, det_tta=False, threshold=0.0)
    model, W, down = rp.load_model(tiny_checkpoint, dev)

    plain_c, plain_e = rp.predict_video(model, clip, dev, cfg=cfg,
                                        window_size=W, downsample=down)
    sec = rp.make_secondary(tiny_checkpoint, dev, det_weight=0.0,
                            edge_weight=0.5, window_size=W, downsample=down)
    same_c, same_e = rp.predict_video(model, clip, dev, cfg=cfg,
                                      window_size=W, downsample=down,
                                      secondary=sec)

    assert len(plain_e) > 0, "fixture emitted no edges; the test proves nothing"
    np.testing.assert_array_equal(plain_c, same_c)
    assert [(a, b) for a, b, _p, _d in plain_e] == \
           [(a, b) for a, b, _p, _d in same_e]
    np.testing.assert_allclose([p for _a, _b, p, _d in plain_e],
                               [p for _a, _b, p, _d in same_e],
                               rtol=1e-6, atol=1e-7)


def test_secondary_rejects_a_mismatched_coordinate_space(tiny_checkpoint):
    """Wrong downsample must raise, not read features from the wrong voxels."""
    dev = torch.device("cpu")
    with pytest.raises(ValueError, match="downsample"):
        rp.make_secondary(tiny_checkpoint, dev, det_weight=0.5,
                          downsample=(1, 2, 2))
    with pytest.raises(ValueError, match="window"):
        rp.make_secondary(tiny_checkpoint, dev, det_weight=0.5,
                          window_size=5)
