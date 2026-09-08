"""Directory-mode inference: the submission path, where no ground truth exists.

The hidden test set ships images alone and its dataset names change at notebook
rerun, so ``predict`` has to enumerate a directory rather than read a fold out
of ``dataset_splits.json``. These tests run that path against a synthetic zarr
built in ``tmp_path`` — the real competition ``test/`` folder is four
placeholder copies of train samples, so it proves nothing the synthetic one
does not, and the fixture must not need the 80 GB dataset present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import tracksdata as td
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from predict_unet_transformer import (
    PredictConfig,
    enumerate_test_datasets,
    run_predictions,
)

# Small enough to run on CPU in seconds; (1, 4, 4) downsampling turns
# (8, 32, 32) into the 8**3 isotropic grid the real pipeline uses at 64**3.
_SHAPE = (3, 8, 32, 32)
_DOWNSAMPLE = [1, 4, 4]
_SCALE = (1.0, 1.625, 0.40625, 0.40625)
_MODEL_CONFIG = {
    "unet_out_channels": 8,
    "unet_layers": [8, 16],
    "downsample": _DOWNSAMPLE,
    "window_size": 2,
}


def _write_zarr(path: Path, *, with_quantiles: bool = True) -> None:
    """A minimal but structurally faithful competition sample.

    Faithful in the two ways inference depends on: the group-level attrs carry
    ``multiscales`` (for the voxel scale) and ``image_statistics.quantiles``
    (for normalisation), and there is no ``.geff`` beside it.
    """
    rng = np.random.default_rng(0)
    vol = rng.integers(20, 60, size=_SHAPE, dtype=np.uint16)
    # A few bright blobs so the detector has local maxima to find.
    for t in range(_SHAPE[0]):
        for z, y, x in [(2, 8, 8), (4, 20, 20), (6, 12, 26)]:
            vol[t, z, y - 1 + t: y + 2 + t, x - 1: x + 2] = 2000

    group = zarr.open_group(str(path), mode="w")
    group.create_array("0", data=vol, chunks=(1, *_SHAPE[1:]))
    attrs: dict = {
        "multiscales": [{
            "datasets": [{"coordinateTransformations": [
                {"type": "scale", "scale": list(_SCALE)},
            ]}],
        }],
    }
    if with_quantiles:
        attrs["image_statistics"] = {"quantiles": {"0.001": 20.0, "0.999": 2000.0}}
    group.attrs.update(attrs)


def _write_weights(weights_dir: Path) -> Path:
    """An untrained checkpoint. This tests plumbing, not prediction quality."""
    from train_unet_transformer import _POS_EMBED_DIM, UNetNodeTransformer
    from volt_biohub.models import TemporalUNet3D

    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "config.json").write_text(json.dumps(_MODEL_CONFIG))
    model = UNetNodeTransformer(
        unet=TemporalUNet3D(
            in_channels=1,
            out_channels=_MODEL_CONFIG["unet_out_channels"],
            layers=_MODEL_CONFIG["unet_layers"],
        ),
        unet_out_channels=_MODEL_CONFIG["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )
    path = weights_dir / "edge_predictor_best.pth"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plumbing test — it must not queue on whatever GPU is free.

    ``run_predictions`` resolves its own device, so the only way to pin it to
    CPU from outside is to make CUDA look absent. That also keeps the test
    honest about the rule that nothing may assume a GPU exists.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def test_dir(tmp_path: Path) -> Path:
    """Two image-only datasets, as the competition test folder ships them."""
    d = tmp_path / "test"
    d.mkdir()
    for name in ("emb_bbbb", "emb_aaaa"):  # written out of order on purpose
        _write_zarr(d / f"{name}.zarr")
    return d


# =============================================================================
# Enumeration
# =============================================================================

def test_enumerates_zarr_without_any_geff(test_dir: Path) -> None:
    assert not list(test_dir.glob("*.geff"))
    assert enumerate_test_datasets(test_dir) == ["emb_aaaa", "emb_bbbb"]


def test_enumeration_ignores_non_dataset_entries(test_dir: Path) -> None:
    """A stray .geff or CSV must not become a dataset name.

    The dataset list must never come from ``sample_submission.csv`` (a top-level
    file, whose four names are placeholders swapped out at rerun), and a stray
    file dropped into the mount must not become a dataset name either.
    """
    (test_dir / "sample_submission.csv").write_text("id,dataset\n")
    (test_dir / "orphan.geff").mkdir()
    assert enumerate_test_datasets(test_dir) == ["emb_aaaa", "emb_bbbb"]


def test_enumeration_raises_on_empty_and_missing_dir(tmp_path: Path) -> None:
    """Loud failure beats an empty submission: every test dataset must appear."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        enumerate_test_datasets(empty)
    with pytest.raises(FileNotFoundError):
        enumerate_test_datasets(tmp_path / "does_not_exist")


# =============================================================================
# Directory-mode prediction
# =============================================================================

@pytest.fixture
def cfg() -> PredictConfig:
    # det_threshold 0.0 keeps every local max: the checkpoint is untrained, so
    # any calibrated threshold would make the node count a coin flip.
    # TTA off because it quadruples the forward passes and tests no new code.
    return PredictConfig(det_threshold=0.0, det_tta=False)


def test_run_predictions_writes_one_geff_per_dataset(
    tmp_path: Path, test_dir: Path, cfg: PredictConfig,
) -> None:
    names = enumerate_test_datasets(test_dir)
    out_dir = tmp_path / "pred"
    out_dir.mkdir()
    # A leftover from an earlier run must be cleared: the CSV writer globs the
    # output directory rather than taking a dataset list, so it would ship.
    (out_dir / "from_a_previous_run.geff").mkdir()

    out = run_predictions(
        data_dir=test_dir,
        names=names,
        output_dir=out_dir,
        weights_path=_write_weights(tmp_path / "weights"),
        cfg=cfg,
    )

    assert sorted(p.stem for p in out.glob("*.geff")) == names

    graph, _ = td.graph.IndexedRXGraph.from_geff(out / f"{names[0]}.geff")
    attrs = graph.node_attrs()
    assert graph.num_nodes() > 0
    # Coordinates must come back in ORIGINAL anisotropic voxel space, which is
    # what the scorer and the submission format require. Wrong by 4x in z here
    # raises nothing and scores near zero, so it is asserted rather than eyeballed.
    assert attrs["t"].max() <= _SHAPE[0] - 1
    for axis, extent, stride in zip("zyx", _SHAPE[1:], _DOWNSAMPLE, strict=True):
        col = attrs[axis].to_numpy()
        assert col.min() >= 0 and col.max() < extent
        assert np.all(col % stride == 0)


def test_skip_failures_still_emits_a_geff(
    tmp_path: Path, test_dir: Path, cfg: PredictConfig,
) -> None:
    """One broken dataset must not cost the other 200.

    A submission missing a dataset is invalid, and there is no second 12 h pass
    in which to retry — so directory mode writes an empty graph and carries on.
    In CV the same failure must raise.
    """
    _write_zarr(test_dir / "emb_broken.zarr", with_quantiles=False)
    names = enumerate_test_datasets(test_dir)
    weights = _write_weights(tmp_path / "weights")

    with pytest.raises(ValueError, match="quantiles"):
        run_predictions(
            data_dir=test_dir, names=names, output_dir=tmp_path / "strict",
            weights_path=weights, cfg=cfg, skip_failures=False,
        )

    out = run_predictions(
        data_dir=test_dir, names=names, output_dir=tmp_path / "tolerant",
        weights_path=weights, cfg=cfg, skip_failures=True,
    )
    assert sorted(p.stem for p in out.glob("*.geff")) == names
    broken, _ = td.graph.IndexedRXGraph.from_geff(out / "emb_broken.geff")
    assert broken.num_nodes() == 0


# =============================================================================
# CLI wiring
# =============================================================================

def test_cli_directory_mode(
    tmp_path: Path, test_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import predict_unet_transformer as pred

    weights = _write_weights(tmp_path / "weights")
    out_dir = tmp_path / "cli_pred"
    monkeypatch.setattr(sys, "argv", [
        "predict", "--test-dir", str(test_dir), "--weights", str(weights),
        "--out-dir", str(out_dir), "--det-threshold", "0.0", "--slice", ":1",
    ])
    pred.main()

    # --slice applies in directory mode too: timing one video against the 12 h
    # cap is a step-4 requirement, and it must not need a splits file.
    assert [p.stem for p in out_dir.glob("*.geff")] == ["emb_aaaa"]


def test_cli_directory_mode_requires_weights(
    test_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fold index exists to derive weights from, and guessing one is worse."""
    import predict_unet_transformer as pred

    monkeypatch.setattr(sys, "argv", ["predict", "--test-dir", str(test_dir)])
    with pytest.raises(SystemExit):
        pred.main()
