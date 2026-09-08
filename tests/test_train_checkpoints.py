"""Per-epoch checkpoints and the run record written beside them.

The point of ``--save-every`` is to make the loss/metric divergence
measurable: model A peaked at CV epoch 0 and decayed while its training loss
kept improving, and with only the best checkpoint on disk there was nothing
left to score in between. So what these tests pin is not "a file appeared" but
"the file can be handed to score_pipeline.py unchanged", i.e. it sits beside
config.json and loads through predict_unet_transformer.load_model.

The same argument applies to the numbers: the per-epoch competition score is
computed every epoch regardless of ``--save-every``, so ``epoch_metrics.csv``
and a config.json carrying the *full* run config are what make a checkpoint
placeable in an ablation afterwards rather than an anonymous .pth.
"""

import csv
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import train_unet_transformer as tut
from predict_unet_transformer import load_model

_FIXTURE_DIR = Path(__file__).parent / "data" / "division_clip"
DS_PATH = _FIXTURE_DIR / "division_clip"

# Deliberately tiny: these tests are about the checkpoint plumbing, not about
# whether the model learns anything, so one iteration per epoch is enough.
_ARCH = {"unet_out_channels": 8, "unet_layers": [8, 16], "downsample": (4, 4, 4)}


def _train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save_every: int,
    n_epochs: int,
    **overrides: object,
) -> Path:
    """Run a throwaway fold on the 5-frame fixture, return its output directory.

    *overrides* go straight to ``train()`` so a test can pin a distinctive
    hyperparameter value and then look for it in the saved config.
    """
    # Force CPU. train() resolves its own device from torch.cuda.is_available()
    # and then DataParallels the UNet across *every* visible card, so on the
    # shared 8-GPU box this would train on other tenants' GPUs and hit the
    # scatter OOM CLAUDE.md documents for the slow test. The checkpoint plumbing
    # under test is device-independent and CPU runs it in ~10 s.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # train() derives its output dir from the module-level WEIGHTS_PATH; redirect
    # it so a test run cannot overwrite a real run's checkpoints.
    monkeypatch.setattr(tut, "WEIGHTS_PATH", tmp_path)
    # conftest rebuilds the fixture geff with tracksdata's to_geff, which drops
    # the `extra` metadata holding estimated_number_of_nodes. Per-epoch checkpoint
    # selection needs that denominator, so stand in a plausible node count rather
    # than skip the selection path these checkpoints are written alongside.
    monkeypatch.setattr("volt_biohub.scoring.read_t_true", lambda data_dir, name: 500.0)
    tut.train(**_fixture_kwargs(save_every, n_epochs, **overrides))
    return tmp_path / "ckpt_test" / "split_0"


def _fixture_kwargs(save_every: int, n_epochs: int, **overrides: object) -> dict:
    """The shared train() call as a dict, so a test can assert against the exact
    values that were passed rather than restating them."""
    kwargs: dict[str, object] = {
        "data_dir": _FIXTURE_DIR,
        "fold": 0,
        # ignored: debug_video is set
        "splits_file": _FIXTURE_DIR / "dataset_splits.json",
        "method": "ckpt_test",
        "n_epochs": n_epochs,
        "lr": 1e-3,
        "batch_size": 2,
        "num_workers": 0,
        "max_iters": 1,
        "unet_out_channels": _ARCH["unet_out_channels"],
        "unet_layers": _ARCH["unet_layers"],
        "downsample": _ARCH["downsample"],
        "debug_video": DS_PATH,
        "seed": 0,
        "augmentations": None,
        "legacy_eval_batches": None,
        "select_videos": 1,
        "save_every": save_every,
    }
    kwargs.update(overrides)
    return kwargs


def test_save_every_writes_one_scoreable_checkpoint_per_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, division_clip_fixture: None
) -> None:
    # Stride 2 over 3 epochs: 0 by the stride, 1 skipped, 2 as the final epoch.
    out_dir = _train(tmp_path, monkeypatch, save_every=2, n_epochs=3)

    # Zero-padded names so a sorted glob is epoch order, which is what makes
    # scoring the sequence a one-liner.
    ckpts = sorted(out_dir.glob("edge_predictor_epoch*.pth"))
    assert [p.name for p in ckpts] == [
        "edge_predictor_epoch000.pth", "edge_predictor_epoch002.pth",
    ]
    assert (out_dir / "edge_predictor_best.pth").exists()
    # score_pipeline.py --weights <ckpt> reads the arch from config.json beside
    # the weights, so a per-epoch checkpoint is useless without it.
    assert (out_dir / "config.json").exists()

    state = torch.load(ckpts[0], map_location="cpu", weights_only=True)
    assert not any(k.startswith("unet.module.") for k in state), \
        "DataParallel prefix must be normalised away or the checkpoint will not load on Kaggle"

    model, window_size, downsample = load_model(ckpts[0], torch.device("cpu"))
    assert tuple(downsample) == _ARCH["downsample"]
    assert window_size == 2
    assert not model.training


def test_save_every_defaults_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, division_clip_fixture: None
) -> None:
    """0 must leave existing runs byte-for-byte as they were: best only."""
    out_dir = _train(tmp_path, monkeypatch, save_every=0, n_epochs=1)
    assert list(out_dir.glob("edge_predictor_epoch*.pth")) == []
    assert (out_dir / "edge_predictor_best.pth").exists()


@pytest.mark.parametrize(
    ("save_every", "expected"),
    [
        (0, []),                    # off: existing runs unchanged
        (1, [0, 1, 2, 3, 4]),
        (2, [0, 2, 4]),
        (3, [0, 3, 4]),             # 4 is not a multiple of 3 but is the last epoch
        (10, [0, 4]),               # both ends of the curve even when the stride never lands
    ],
)
def test_which_epochs_are_saved(save_every: int, expected: list[int]) -> None:
    got = [e for e in range(5) if tut._save_this_epoch(e, 5, save_every)]
    assert got == expected


def test_portable_state_strips_the_dataparallel_prefix() -> None:
    """Pinned without a GPU: DataParallel only renames keys, so a plain wrapper
    reproduces exactly the prefix that has to be undone."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = nn.Linear(2, 2)
            self.head = nn.Linear(2, 2)

    model = Tiny()
    plain = set(tut._portable_state(model))
    model.unet = nn.DataParallel(model.unet)
    assert any(k.startswith("unet.module.") for k in model.state_dict())
    assert set(tut._portable_state(model)) == plain


def test_epoch_metrics_csv_records_the_cv_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, division_clip_fixture: None
) -> None:
    """The curve is written whether or not per-epoch checkpoints are on.

    ``epoch_selection_score`` runs every epoch either way, and the divergence
    it measures was previously only recoverable by grepping a tqdm-mangled
    console log.
    """
    out_dir = _train(tmp_path, monkeypatch, save_every=0, n_epochs=2)
    curve = out_dir / "epoch_metrics.csv"

    with curve.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == tut._EPOCH_METRIC_FIELDS
    assert [int(r["epoch"]) for r in rows] == [0, 1]
    # Every column has to parse as a number or the file is not machine-readable;
    # NaN is allowed (the legacy accuracy is skipped, and a split with no
    # divisions at all gives the scorer an empty denominator).
    for row in rows:
        for field in tut._EPOCH_METRIC_FIELDS:
            float(row[field])
    # The flag has to agree with the score, since it is what says which epoch
    # edge_predictor_best.pth actually holds.
    best = max(rows, key=lambda r: float(r["cv_score"]))
    assert int(best["is_best"]) == 1

    # A second run of the same fold replaces config.json, so it must replace the
    # curve too: two runs spliced into one file is a silently wrong ablation.
    _train(tmp_path, monkeypatch, save_every=0, n_epochs=2)
    with curve.open(newline="") as fh:
        assert [int(r["epoch"]) for r in csv.DictReader(fh)] == [0, 1]


def test_config_json_records_the_full_run_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, division_clip_fixture: None
) -> None:
    """Architecture alone cannot place a checkpoint in an ablation.

    Distinctive values throughout, so a config that silently recorded the
    defaults instead of the run would fail rather than pass by coincidence.
    """
    out_dir = _train(
        tmp_path, monkeypatch, save_every=3, n_epochs=1,
        lr=3.5e-4, seed=7, det_loss_weight=2.5, det_neg_weight=0.25,
        batch_size=1, select_budget=1.5,
    )
    config = json.loads((out_dir / "config.json").read_text())

    assert config["lr"] == 3.5e-4
    assert config["seed"] == 7
    assert config["det_loss_weight"] == 2.5
    assert config["det_neg_weight"] == 0.25
    assert config["n_epochs"] == 1
    assert config["batch_size"] == 1
    assert config["select_budget"] == 1.5
    assert config["save_every"] == 3
    assert config["max_iters"] == 1
    assert config["augmentations"] == []   # this run passes augmentations=None

    # The keys predict_unet_transformer.load_model indexes must stay flat and
    # top-level: this file is read back at inference, so enriching it cannot
    # become a submission-time crash.
    model, window_size, downsample = load_model(
        out_dir / "edge_predictor_best.pth", torch.device("cpu"),
    )
    assert tuple(downsample) == _ARCH["downsample"]
    assert window_size == config["window_size"] == 2
    assert not model.training


def test_config_records_every_train_parameter_with_the_value_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, division_clip_fixture: None
) -> None:
    """Not "some keys are present": every train() parameter, at the run's value.

    Spot-checking a handful of keys cannot catch the failure that matters --
    a field wired to the wrong variable, or a new parameter added later and
    never recorded -- and a config that quietly holds the defaults instead of
    the run makes every later ablation row a guess. Compared against the exact
    kwargs the run was given, with non-default values for the loss weights and
    the detector knobs so a stale field cannot agree by coincidence.
    """
    import inspect

    overrides = {
        "lr": 3.5e-4, "seed": 7, "det_loss_weight": 2.5, "det_neg_weight": 0.25,
        "det_pseudo_weight": 0.75, "select_budget": 1.5, "max_frames": 4,
        "det_sigma_um": 2.5, "det_target": "voxel", "det_pu_mask": False,
        "pool_kernel_um": 4.0, "data_parallel": False,
    }
    out_dir = _train(tmp_path, monkeypatch, save_every=3, n_epochs=1, **overrides)
    config = json.loads((out_dir / "config.json").read_text())
    passed = _fixture_kwargs(save_every=3, n_epochs=1, **overrides)

    sig = inspect.signature(tut.train)
    missing = [name for name in sig.parameters if name not in config]
    assert missing == [], f"train() parameters absent from config.json: {missing}"

    def _as_json(value: object) -> object:
        """json.dumps flattens paths to str and tuples to lists; mirror that."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    mismatched = {}
    for name, param in sig.parameters.items():
        want = passed.get(name, param.default)
        if name == "splits_file":
            want = None   # debug_video is set, so the splits file is unused
        if name == "augmentations":
            want = []     # recorded as function names; this run passes None
        if _as_json(want) != config[name]:
            mismatched[name] = (_as_json(want), config[name])
    assert mismatched == {}, mismatched


def test_is_best_flag_identifies_the_checkpoint_that_survived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, division_clip_fixture: None
) -> None:
    """The flag has to name the epoch ``edge_predictor_best.pth`` actually holds.

    ``is_best`` is a running-max flag, so several rows can carry it and only the
    last one is the checkpoint on disk. Asserting the *weights* match that epoch
    is what makes the curve usable for choosing an epoch to re-score; a row that
    merely correlates with the maximum would not.
    """
    out_dir = _train(tmp_path, monkeypatch, save_every=1, n_epochs=3)

    with (out_dir / "epoch_metrics.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    flagged = [int(r["epoch"]) for r in rows if int(r["is_best"])]
    assert flagged, "no epoch was ever flagged best"
    scores = [float(r["cv_score"]) for r in rows]
    assert scores[flagged[-1]] == max(scores)

    best = torch.load(out_dir / "edge_predictor_best.pth", map_location="cpu",
                      weights_only=True)
    epoch_ck = torch.load(out_dir / f"edge_predictor_epoch{flagged[-1]:03d}.pth",
                          map_location="cpu", weights_only=True)
    assert set(best) == set(epoch_ck)
    assert all(torch.equal(best[k], epoch_ck[k]) for k in best), (
        "edge_predictor_best.pth is not the epoch epoch_metrics.csv flags as best"
    )
