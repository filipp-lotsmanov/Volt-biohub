"""Kaggle mount discovery.

These functions exist because three hardcoded mount paths were each wrong in a
different way, and each cost a push-run-read cycle on Kaggle to discover. They
run only inside a Kaggle kernel, so this file is the only place they can be
checked at all — which is the whole reason they live in the repo rather than in
``notebook.py``.

Layouts below are the ones actually observed on the mount, not invented.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle"))

import kaggle_env


def build(root: Path, files: list[str]) -> Path:
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    return root


# What Kaggle actually produced: `-r zip` kept volt-code's two directory levels
# and flattened volt-weights' single one.
MEASURED = [
    "volt-code/kaggle/requirements-offline.txt",
    "volt-code/src/volt_biohub/metrics.py",
    "volt-wheels/zarr-3.0.10-py3-none-any.whl",
    "volt-weights/edge_predictor_best.pth",
    "volt-weights/config.json",
    "competitions/slug/test/a.zarr/zarr.json",
    "competitions/slug/sample_submission.csv",
]

# What kaggle/README.md originally assumed. Both must work — a re-upload with a
# different number of folders flips between them without warning.
NESTED = [
    "volt-code/volt-code/kaggle/requirements-offline.txt",
    "volt-code/volt-code/src/volt_biohub/metrics.py",
    "volt-wheels/wheels/zarr.whl",
    "volt-weights/split_0/edge_predictor_best.pth",
    "volt-weights/split_0/config.json",
    "competitions/slug/test/a.zarr/zarr.json",
]


@pytest.mark.parametrize(("label", "layout"), [("measured", MEASURED), ("nested", NESTED)])
def test_finds_every_input_in_both_observed_layouts(tmp_path, label, layout):
    root = build(tmp_path, layout)
    req = kaggle_env.find_in_mounts("requirements-offline.txt", root)
    pkg = kaggle_env.find_in_mounts("volt_biohub/metrics.py", root)
    whl = kaggle_env.find_in_mounts("*.whl", root)
    assert req is not None and req.name == "requirements-offline.txt", label
    assert pkg is not None and pkg.parent.parent.name == "src", label
    assert whl is not None and whl.suffix == ".whl", label
    assert kaggle_env.find_weights(root) is not None, label
    assert kaggle_env.find_test_dir(root).name == "test", label


def test_competition_mount_is_never_searched_for_datasets(tmp_path):
    """A stray .whl inside the competition must not be mistaken for the wheels.

    The competition mount holds ~90k zarr chunk files; searching it is both slow
    and wrong, since its contents are the organisers', not ours.
    """
    root = build(tmp_path, [*MEASURED, "competitions/slug/decoy-1.0-py3-none-any.whl"])
    whl = kaggle_env.find_in_mounts("*.whl", root)
    assert whl is not None
    assert "decoy" not in whl.name
    assert [m.name for m in kaggle_env.dataset_mounts(root)] == [
        "volt-code", "volt-weights", "volt-wheels",
    ]


def test_checkpoint_without_config_reads_as_no_weights(tmp_path):
    """A lone .pth must degrade to DoG, not crash load_detector mid-run."""
    root = build(tmp_path, ["volt-weights/edge_predictor_best.pth"])
    assert kaggle_env.find_weights(root) is None


def test_missing_inputs_return_none_rather_than_raising(tmp_path):
    (tmp_path / "volt-code").mkdir()
    assert kaggle_env.find_in_mounts("requirements-offline.txt", tmp_path) is None
    assert kaggle_env.find_weights(tmp_path) is None
    assert kaggle_env.dataset_mounts(tmp_path / "nonexistent") == []


def test_test_dir_must_contain_zarr_not_merely_be_named_test(tmp_path):
    """An empty test/ is the failure that would produce an empty submission.

    Taking the first directory named ``test`` would let a mount change sail
    through and be discovered only in the scored result. Fail in the opening
    seconds instead.
    """
    root = build(tmp_path, ["competitions/slug/test/README.md"])
    with pytest.raises(SystemExit, match="no test/ directory containing"):
        kaggle_env.find_test_dir(root)


def test_test_dir_prefers_the_one_with_data(tmp_path):
    root = build(tmp_path, [
        "other/test/notes.txt",
        "competitions/slug/test/a.zarr/zarr.json",
    ])
    assert kaggle_env.find_test_dir(root).parent.name == "slug"


def test_gpu_check_is_false_without_a_working_cuda_device(monkeypatch):
    """``is_available()`` True with dead kernels is the real failure mode.

    Kaggle's ``enable_gpu`` can hand out an sm_60 P100 against a torch built for
    sm_70+, where availability reports True and every conv raises. The check has
    to execute something.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert kaggle_env.gpu_is_usable() is False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(torch, "zeros", boom)
    assert kaggle_env.gpu_is_usable() is False
