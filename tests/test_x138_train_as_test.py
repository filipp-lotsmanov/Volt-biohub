"""--train-as-test: x138 runs on labelled train movies, and refuses to be graded.

x138's notebook is third-party and never committed, so the patch is applied to a
stand-in for its cell 2 that defines COMP_DIR above the same anchor line.
"""

import os
import shutil
import sys
from pathlib import Path

import polars as pl
import pytest
import tracksdata as td

sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle" / "forks"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import build_x138_variant as builder

import x138_divdiag_lists as lists

STAND_IN = 'COMP_DIR = Path(COMP)\nTEST_DIR = COMP_DIR / "test"\nWORKING_DIR = Path(".")\n'


def competition(root: Path, train: list[str], test: list[str]) -> Path:
    for split, stems in (("train", train), ("test", test)):
        for s in stems:
            (root / split / f"{s}.zarr").mkdir(parents=True)
    return root


def run(root: Path, stems: list[str], out: Path, monkeypatch) -> dict:
    monkeypatch.setenv("VOLT_TRAIN_AS_TEST_DIR", str(out))
    ns = {"Path": Path, "os": os, "shutil": shutil, "COMP": str(root)}
    exec(builder.patch_train_as_test(STAND_IN, stems), ns)
    return ns


def listed(test_dir: Path) -> list[str]:
    """x138's own `list_test_stems`."""
    return sorted(p.name[:-5] for p in test_dir.iterdir() if p.name.endswith(".zarr"))


TRAIN = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1", "6bba_0aaaaaaa"]


def test_test_dir_holds_links_to_exactly_the_chosen_train_movies(tmp_path, monkeypatch):
    root = competition(tmp_path / "comp", TRAIN, TRAIN[:4])      # the visible run's layout
    ns = run(root, ["6bba_0aaaaaaa", "44b6_0b24845f"], tmp_path / "t", monkeypatch)
    test_dir = ns["TEST_DIR"]
    assert test_dir == tmp_path / "t"
    assert listed(test_dir) == ["44b6_0b24845f", "6bba_0aaaaaaa"]
    link = test_dir / "6bba_0aaaaaaa.zarr"
    assert link.is_symlink() and link.resolve() == (root / "train" / "6bba_0aaaaaaa.zarr").resolve()
    assert ns["WORKING_DIR"] == Path(".")                        # the rest of the cell still runs


def test_rerun_replaces_the_previous_links(tmp_path, monkeypatch):
    root = competition(tmp_path / "comp", TRAIN, [])
    run(root, TRAIN, tmp_path / "t", monkeypatch)
    ns = run(root, TRAIN[:1], tmp_path / "t", monkeypatch)
    assert listed(ns["TEST_DIR"]) == TRAIN[:1]


def test_a_movie_missing_from_train_stops_the_run(tmp_path, monkeypatch):
    root = competition(tmp_path / "comp", TRAIN, [])
    with pytest.raises(FileNotFoundError, match="1 movie"):
        run(root, [TRAIN[0], "44b6_ffffffff"], tmp_path / "t", monkeypatch)


def test_refuses_a_graded_rerun(tmp_path, monkeypatch):
    # The hidden set is a third embryo: its prefix is not among train's.
    root = competition(tmp_path / "comp", TRAIN, ["9c1d_12345678"])
    with pytest.raises(RuntimeError, match="graded rerun"):
        run(root, TRAIN[:2], tmp_path / "t", monkeypatch)
    assert not (tmp_path / "t").exists()


def test_read_stems(tmp_path):
    f = tmp_path / "fold.txt"
    f.write_text("# fold 1\n44b6_0113de3b.zarr\n\n44b6_0b24845f  # has divisions\n")
    assert builder.read_stems(f) == ["44b6_0113de3b", "44b6_0b24845f"]
    for bad in ("44b6 0113de3b\n", "", "44b6_0113de3b\n44b6_0113de3b\n", "../etc\n"):
        f.write_text(bad)
        with pytest.raises(SystemExit):
            builder.read_stems(f)


def test_anchor_must_appear_exactly_once():
    with pytest.raises(SystemExit, match="found 0 times"):
        builder.patch_train_as_test("TEST_DIR = somewhere_else\n", TRAIN)


def test_kernel_goes_to_the_owner_given():
    meta = builder.kernel_metadata("volt-x138-divdiag-f1", "someone", [])
    assert meta["id"] == "someone/volt-x138-divdiag-f1"
    assert meta["code_file"] == "volt-x138-divdiag-f1.ipynb"
    assert meta["is_private"] is True
    assert "biohub-cell-tracking-during-development" in meta["competition_sources"]
    extra = builder.kernel_metadata("s", "o", [builder.SOUPS_DATASET])["dataset_sources"]
    assert extra[-1] == builder.SOUPS_DATASET
    assert builder.SOUPS_DATASET not in meta["dataset_sources"]


# ---- the run lists ----------------------------------------------------------------


def test_division_movies_come_first_and_every_movie_once():
    counts = {f"44b6_{i:08x}": (3 if i < 5 else 0) for i in range(12)}
    counts |= {f"6bba_{i:08x}": (1 if i < 2 else 0) for i in range(9)}
    out = lists.make_lists(counts, max_per_list=4)
    flat = [s for chunk in out for s in chunk]
    assert sorted(flat) == sorted(counts)
    assert all(len(chunk) <= 4 for chunk in out)
    n_div_movies = sum(v > 0 for v in counts.values())
    first = [s for chunk in out[:2] for s in chunk]           # 7 division movies -> 2 lists
    assert len(first) == n_div_movies and all(counts[s] > 0 for s in first)
    assert all(counts[s] == 0 for chunk in out[2:] for s in chunk)


def _geff(path: Path, divide: bool):
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, 0.0)
    a = g.add_node({"t": 0, "z": 1.0, "y": 1.0, "x": 1.0})
    b = g.add_node({"t": 1, "z": 1.0, "y": 2.0, "x": 1.0})
    g.add_edge(a, b, {})
    if divide:
        c = g.add_node({"t": 1, "z": 1.0, "y": 9.0, "x": 1.0})
        g.add_edge(a, c, {})
    g.to_geff(str(path))


def test_main_counts_gt_divisions_from_the_geffs(tmp_path, capsys):
    data = tmp_path / "train"
    data.mkdir()
    for stem, divide in (("44b6_00000001", True), ("44b6_00000002", False), ("6bba_00000003", False)):
        (data / f"{stem}.zarr").mkdir()
        _geff(data / f"{stem}.geff", divide)
    assert lists.gt_divisions(data / "44b6_00000001.geff") == 1
    assert lists.gt_divisions(data / "44b6_00000002.geff") == 0
    assert lists.main(["--data-dir", str(data), "--out-dir", str(tmp_path / "out"),
                       "--max-per-list", "2"]) == 0
    got = [builder.read_stems(p) for p in sorted((tmp_path / "out").glob("divdiag_*.txt"))]
    assert got == [["44b6_00000001"], ["44b6_00000002", "6bba_00000003"]]
    assert "1 GT divisions in 1 of 3 movies" in capsys.readouterr().out


# ---- the notebook argument and the ilpdis15 base -----------------------------------

def test_notebook_argument_accepts_the_pull_directory(tmp_path):
    (tmp_path / "x138").mkdir()
    nb = tmp_path / "x138" / "volt-x138-ilpdis15.ipynb"
    nb.write_text("{}")
    (tmp_path / "x138" / "kernel-metadata.json").write_text("{}")
    assert builder.resolve_notebook(tmp_path / "x138") == nb
    assert builder.resolve_notebook(nb) == nb
    with pytest.raises(SystemExit, match=r"0 \.ipynb"):
        builder.resolve_notebook(tmp_path)
    with pytest.raises(SystemExit, match="no such notebook"):
        builder.resolve_notebook(tmp_path / "missing.ipynb")


def test_ilpdis15_revert_changes_only_the_disappearance_value():
    for old, new in builder.ILPDIS15_REVERT:
        assert "BIOHUB_ILP_DISAPPEARANCE_WEIGHT" in old and "1.5" in old
        assert new.split("#")[0].replace("2.0", "1.5") == old.split("#")[0]
