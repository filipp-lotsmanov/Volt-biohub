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
