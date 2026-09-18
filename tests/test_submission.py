"""Tests for the submission-format validator.

The negative cases carry the weight here: a validator that only recognises a
correct file is worth nothing, since the correct file is the one case we would
have noticed anyway. Every invariant in CLAUDE.md's "Submission format" section
gets a malformed CSV that must fail on a named error code.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
# kaggle/ holds the offline submission writer; like scripts/ it is not a
# package, so it is reached by path insertion rather than import.
sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle"))

from volt_biohub.submission import (
    COLUMNS,
    SubmissionError,
    assert_valid_submission,
    validate_submission,
)

# Small T so a t-out-of-bounds case is easy to write; Z/Y/X match the real data.
STUB_SHAPE = (10, 64, 256, 256)
DATASETS = ("aaaa_0001", "bbbb_0002")


# --------------------------------------------------------------------------- #
# fixtures and row helpers
# --------------------------------------------------------------------------- #

def _node(dataset: str, node_id: int, t: int, z: int, y: int, x: int) -> dict:
    return {"dataset": dataset, "row_type": "node", "node_id": node_id,
            "t": t, "z": z, "y": y, "x": x, "source_id": -1, "target_id": -1}


def _edge(dataset: str, source_id: int, target_id: int) -> dict:
    return {"dataset": dataset, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": source_id, "target_id": target_id}


@pytest.fixture(scope="session")
def test_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stand-in competition test directory: array metadata only, no chunks.

    The validator reads dataset names and volume shapes from here, both of which
    live in the zarr metadata, so no pixel data is needed and nothing is written
    into the real data directory. Session-scoped because every test reads it and
    none writes to it.
    """
    root = tmp_path_factory.mktemp("test_dir") / "test"
    root.mkdir()
    for name in DATASETS:
        group = zarr.open_group(str(root / f"{name}.zarr"), mode="w")
        group.create_array("0", shape=STUB_SHAPE, dtype="uint16",
                           chunks=(1, *STUB_SHAPE[1:]))
    return root


def _valid_rows() -> list[dict]:
    """A minimal but structurally complete submission.

    Deliberately includes a division (node 2 has two children) and restarts
    ``node_id`` at 1 in the second dataset, so a regression that demanded
    globally unique ids would fail here.
    """
    a, b = DATASETS
    rows = [
        _node(a, 1, 0, 10, 100, 100),
        _node(a, 2, 1, 10, 101, 100),
        _node(a, 3, 2, 10, 102, 100),
        _node(a, 4, 2, 10, 110, 100),
        _edge(a, 1, 2),
        _edge(a, 2, 3),
        _edge(a, 2, 4),
        # Second dataset reuses ids 1..2 and has an isolated node with no edges.
        _node(b, 1, 0, 20, 200, 200),
        _node(b, 2, 1, 20, 201, 200),
        _node(b, 3, 5, 30, 30, 30),
        _edge(b, 1, 2),
    ]
    return rows


def _write(path: Path, rows: list[dict], header: tuple[str, ...] = COLUMNS) -> Path:
    """Serialise rows, numbering the throwaway ``id`` index from 0."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for i, row in enumerate(rows):
            writer.writerow([i] + [row[c] for c in COLUMNS[1:]])
    return path


@pytest.fixture
def submission(tmp_path: Path) -> Path:
    return _write(tmp_path / "submission.csv", _valid_rows())


def _codes(report) -> set[str]:
    return {p.code for p in report.errors}


# --------------------------------------------------------------------------- #
# the happy path, and the mistaken-claim guard
# --------------------------------------------------------------------------- #

def test_valid_submission_passes(submission: Path, test_dir: Path) -> None:
    report = validate_submission(submission, test_dir)
    assert report.ok, str(report)
    assert not report.problems, str(report)
    assert report.stats["n_datasets"] == 2
    assert report.stats["n_nodes"] == 7
    assert report.stats["n_edges"] == 4


def test_node_ids_need_only_be_unique_within_a_dataset(
    submission: Path, test_dir: Path
) -> None:
    """Guards the earlier MISTAKEN claim that ids must be globally unique.

    sample_submission.csv restarts node_id at 1 in every dataset and
    csv_to_geffs groups by dataset before resolving edges, so cross-dataset
    reuse is legal and must never be reported.
    """
    report = validate_submission(submission, test_dir)
    assert report.ok
    rows = [r for r in _valid_rows() if r["row_type"] == "node"]
    assert {r["node_id"] for r in rows if r["dataset"] == DATASETS[0]} & {
        r["node_id"] for r in rows if r["dataset"] == DATASETS[1]
    }, "fixture no longer reuses ids across datasets, so it guards nothing"


def test_assert_valid_returns_report_on_success(submission: Path, test_dir: Path) -> None:
    assert assert_valid_submission(submission, test_dir).ok


def test_assert_valid_raises_on_failure(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows[0]["y"] = 999
    path = _write(tmp_path / "submission.csv", rows)
    with pytest.raises(SubmissionError) as excinfo:
        assert_valid_submission(path, test_dir)
    assert "y_out_of_bounds" in str(excinfo.value)
    assert not excinfo.value.report.ok


# --------------------------------------------------------------------------- #
# file-level structure
# --------------------------------------------------------------------------- #

def test_missing_file_is_an_error(tmp_path: Path, test_dir: Path) -> None:
    report = validate_submission(tmp_path / "submission.csv", test_dir)
    assert _codes(report) == {"missing_file"}


def test_wrong_header_is_fatal(tmp_path: Path, test_dir: Path) -> None:
    header = ("id", "dataset", "row_type", "node_id", "t", "x", "y", "z",
              "source_id", "target_id")  # z/y/x transposed
    report = validate_submission(_write(tmp_path / "submission.csv", _valid_rows(), header),
                                 test_dir)
    assert _codes(report) == {"header"}


def test_header_only_file_is_an_error(tmp_path: Path, test_dir: Path) -> None:
    report = validate_submission(_write(tmp_path / "submission.csv", []), test_dir)
    assert _codes(report) == {"empty"}


def test_wrong_filename_warns_but_passes(tmp_path: Path, test_dir: Path) -> None:
    """The name is enforced at the upload location, so a scratch CSV is legal."""
    report = validate_submission(_write(tmp_path / "preds.csv", _valid_rows()), test_dir)
    assert report.ok
    assert [p.code for p in report.warnings] == ["filename"]


def test_non_integer_coordinate_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows[0]["y"] = "100.5"
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "non_integer" in _codes(report)


def test_empty_field_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows[0]["z"] = ""
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "non_integer" in _codes(report)


def test_all_rows_unparseable_does_not_crash(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    for row in rows:
        row["z"] = "nan"
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "non_integer" in _codes(report)
    assert "missing_dataset" in _codes(report)


def test_duplicate_id_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    path = tmp_path / "submission.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([0] + [row[c] for c in COLUMNS[1:]])
    report = validate_submission(path, test_dir)
    assert "duplicate_id" in _codes(report)


# --------------------------------------------------------------------------- #
# row-type conventions
# --------------------------------------------------------------------------- #

def test_unknown_row_type_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows[0]["row_type"] = "nodes"
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "row_type" in _codes(report)


def test_node_row_must_carry_minus_one_endpoints(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows[0]["target_id"] = 2
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "node_sentinel" in _codes(report)


def test_edge_row_must_carry_minus_one_coordinates(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows[4]["t"] = 0
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "edge_sentinel" in _codes(report)


# --------------------------------------------------------------------------- #
# dataset coverage
# --------------------------------------------------------------------------- #

def test_missing_dataset_fails(tmp_path: Path, test_dir: Path) -> None:
    """A dataset predicting zero nodes still has to appear in the CSV."""
    rows = [r for r in _valid_rows() if r["dataset"] != DATASETS[1]]
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "missing_dataset" in _codes(report)


def test_dataset_name_with_zarr_suffix_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    for row in rows:
        if row["dataset"] == DATASETS[1]:
            row["dataset"] = f"{DATASETS[1]}.zarr"
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert {"missing_dataset", "unknown_dataset"} <= _codes(report)


def test_missing_test_dir_raises(tmp_path: Path, submission: Path) -> None:
    """A bad test_dir is a caller error, not a submission defect."""
    with pytest.raises(FileNotFoundError):
        validate_submission(submission, tmp_path / "nope")


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #

def test_duplicate_node_id_within_dataset_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows[1]["node_id"] = 1
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "duplicate_node_id" in _codes(report)


def test_negative_node_id_fails(tmp_path: Path, test_dir: Path) -> None:
    """-1 is the edge sentinel, so a node can never legally hold it."""
    rows = _valid_rows()
    rows[0]["node_id"] = -1
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "negative_node_id" in _codes(report)


def test_duplicate_node_position_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows[3].update(t=rows[2]["t"], z=rows[2]["z"], y=rows[2]["y"], x=rows[2]["x"])
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "duplicate_node" in _codes(report)


@pytest.mark.parametrize(
    ("axis", "value", "code"),
    [
        ("t", 10, "t_out_of_bounds"),
        ("t", -1, "t_out_of_bounds"),
        ("z", 64, "z_out_of_bounds"),
        ("y", 256, "y_out_of_bounds"),
        ("x", 300, "x_out_of_bounds"),
    ],
)
def test_out_of_bounds_coordinate_fails(
    tmp_path: Path, test_dir: Path, axis: str, value: int, code: str
) -> None:
    rows = _valid_rows()
    rows[0][axis] = value
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert code in _codes(report)


def _shrunk_nodes(dataset: str, count: int = 60) -> list[dict]:
    """Nodes confined to a quarter of y and x, as a 1,4,4 grid would leave them."""
    return [
        _node(dataset, i + 1, i % STUB_SHAPE[0], 30, i, (i * 3) % 60)
        for i in range(count)
    ]


def test_isotropic_coordinates_in_one_dataset_only_warn(
    tmp_path: Path, test_dir: Path
) -> None:
    """A single localised dataset is plausible biology, so it must not block."""
    rows = _shrunk_nodes(DATASETS[0])
    rows.append(_node(DATASETS[1], 1, 0, 20, 200, 200))
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert report.ok, str(report)
    assert "suspect_isotropic_coords" in {p.code for p in report.warnings}


def test_isotropic_coordinates_everywhere_is_fatal(tmp_path: Path, test_dir: Path) -> None:
    """Wrong by 4x, no exception, score near zero — and a global bug is global.

    Every dataset shrinking at once cannot be biology, so this is the one
    coordinate-space failure the validator refuses outright.
    """
    rows = _shrunk_nodes(DATASETS[0]) + _shrunk_nodes(DATASETS[1])
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "suspect_isotropic_coords" in _codes(report)


def test_correct_scale_does_not_warn_about_isotropic(tmp_path: Path, test_dir: Path) -> None:
    rows: list[dict] = []
    for i in range(60):
        rows.append(_node(DATASETS[0], i + 1, i % STUB_SHAPE[0], 30, i * 4, (i * 3) % 240))
    rows.append(_node(DATASETS[1], 1, 0, 20, 200, 200))
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert not report.problems, str(report)


# --------------------------------------------------------------------------- #
# edges -- each of these is silently repaired by the scorer, so it must be loud
# --------------------------------------------------------------------------- #

def test_unresolved_edge_endpoint_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows.append(_edge(DATASETS[0], 1, 99))
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "unresolved_edge" in _codes(report)


def test_edge_endpoints_resolve_within_their_own_dataset(
    tmp_path: Path, test_dir: Path
) -> None:
    """Node id 3 exists in both datasets; the edge must use its own dataset's."""
    rows = _valid_rows()
    # In dataset b, node 3 sits at t=5, so an edge 2->3 spans t=1 -> t=5.
    rows.append(_edge(DATASETS[1], 2, 3))
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "edge_dt" in _codes(report)
    assert "unresolved_edge" not in _codes(report)


def test_edge_spanning_two_frames_fails(tmp_path: Path, test_dir: Path) -> None:
    """The scorer deletes any edge with dt != 1 without saying so."""
    rows = _valid_rows()
    rows.append(_edge(DATASETS[0], 1, 3))  # t=0 -> t=2
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "edge_dt" in _codes(report)


def test_backwards_edge_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows.append(_edge(DATASETS[0], 3, 2))  # t=2 -> t=1
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "edge_dt" in _codes(report)


def test_duplicate_edge_fails(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    rows.append(_edge(DATASETS[0], 1, 2))
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "duplicate_edge" in _codes(report)


def test_out_degree_three_fails(tmp_path: Path, test_dir: Path) -> None:
    """Two children is a division; three is truncated by the scorer, silently."""
    rows = _valid_rows()
    rows.append(_node(DATASETS[0], 5, 2, 10, 120, 100))
    rows.append(_edge(DATASETS[0], 2, 5))
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "out_degree" in _codes(report)


def test_division_of_out_degree_two_is_allowed(submission: Path, test_dir: Path) -> None:
    report = validate_submission(submission, test_dir)
    assert "out_degree" not in _codes(report)


def test_merge_warns_but_passes(tmp_path: Path, test_dir: Path) -> None:
    """Two parents is impossible biology but legal format, so it warns."""
    rows = _valid_rows()
    rows.append(_node(DATASETS[0], 5, 1, 10, 120, 100))
    rows.append(_edge(DATASETS[0], 5, 3))
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert report.ok, str(report)
    assert "merge" in {p.code for p in report.warnings}


# --------------------------------------------------------------------------- #
# the report itself
# --------------------------------------------------------------------------- #

def test_all_problems_reported_at_once(tmp_path: Path, test_dir: Path) -> None:
    """The reason this returns a report instead of raising on the first defect.

    Regenerating the CSV is hours of inference, so one pass has to surface
    everything rather than the first thing.
    """
    rows = _valid_rows()
    rows[0]["y"] = 999
    rows[1]["node_id"] = 1
    rows.append(_edge(DATASETS[0], 1, 3))
    report = validate_submission(_write(tmp_path / "wrong.csv", rows), test_dir)
    assert {"y_out_of_bounds", "duplicate_node_id", "edge_dt"} <= _codes(report)
    assert "wrong.csv" in str(report) and "FAIL" in str(report)


def test_problem_counts_and_examples(tmp_path: Path, test_dir: Path) -> None:
    rows = _valid_rows()
    for row in rows:
        if row["row_type"] == "node":
            row["y"] = 999
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    bounds = [p for p in report.errors if p.code == "y_out_of_bounds"]
    assert len(bounds) == 1, "one problem per axis, not one per dataset"
    assert bounds[0].count == 7  # every node row, across both datasets
    bounds = bounds[0]
    assert bounds.examples and "y=999" in bounds.examples[0]


# --------------------------------------------------------------------------- #
# against the real data: the format oracle and the geff round trip
# --------------------------------------------------------------------------- #

def _data_root() -> Path:
    from dataspec import DATA_ROOT

    return DATA_ROOT


def test_sample_submission_oracle_passes() -> None:
    """sample_submission.csv is the format oracle, so it must validate.

    If this fails, the validator is wrong, not the file.
    """
    oracle = _data_root() / "sample_submission.csv"
    real_test_dir = _data_root() / "test"
    if not oracle.exists() or not real_test_dir.is_dir():
        pytest.skip("competition data not present")
    report = validate_submission(oracle, real_test_dir)
    assert report.ok, str(report)
    # Its only complaint may be the filename, which is a property of where the
    # file is uploaded rather than of its content.
    assert {p.code for p in report.warnings} <= {"filename"}, str(report)


def _graph_signature(geff_path: Path) -> tuple[list, list]:
    """Nodes as sorted (t,z,y,x), edges as sorted position pairs.

    Compared by position rather than id on purpose: csv_to_geffs assigns fresh
    ids, and the scorer matches on centroid distance, so ids carry no meaning
    across the round trip.
    """
    group = zarr.open_group(str(geff_path), mode="r")
    ids = [int(i) for i in group["nodes/ids"][:]]
    props = {k: group[f"nodes/props/{k}/values"][:] for k in ("t", "z", "y", "x")}
    position = {
        node_id: tuple(round(float(props[k][row])) for k in ("t", "z", "y", "x"))
        for row, node_id in enumerate(ids)
    }
    edges = sorted(
        (position[int(s)], position[int(t)]) for s, t in group["edges/ids"][:]
    )
    return sorted(position.values()), edges


@pytest.mark.parametrize("name", ["44b6_2a2eff9f", "6bba_c328f2fd"])
def test_geff_csv_round_trip_preserves_the_graph(tmp_path: Path, name: str) -> None:
    """geffs_to_csv -> validate -> csv_to_geffs on real GT must be lossless.

    Both samples contain a division, so this also pins that out-degree 2
    survives the round trip — the structure the division term is scored on.
    """
    from csv_to_geffs import csv_to_geffs
    from dataspec import DATASET_PATH
    from geffs_to_csv import geffs_to_csv

    source = DATASET_PATH / f"{name}.geff"
    if not source.exists():
        pytest.skip(f"{source} not present")

    geff_dir = tmp_path / "gt"
    geff_dir.mkdir()
    shutil.copytree(source, geff_dir / source.name)

    stub = tmp_path / "test"
    stub.mkdir()
    group = zarr.open_group(str(stub / f"{name}.zarr"), mode="w")
    group.create_array("0", shape=(100, 64, 256, 256), dtype="uint16",
                       chunks=(1, 64, 256, 256))

    csv_path = geffs_to_csv(geff_dir, tmp_path / "submission.csv")
    report = validate_submission(csv_path, stub)
    assert report.ok, str(report)
    assert not report.problems, str(report)

    out_dir = tmp_path / "round_trip"
    csv_to_geffs(csv_path, out_dir)
    assert _graph_signature(source) == _graph_signature(out_dir / f"{name}.geff")


# --------------------------------------------------------------------------- #
# the shell entry point
# --------------------------------------------------------------------------- #

def test_cli_exit_codes(
    tmp_path: Path, test_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Non-zero exit on failure, so it can gate a shell step."""
    from volt_biohub.submission import main

    good = _write(tmp_path / "submission.csv", _valid_rows())
    monkeypatch.setattr(sys, "argv", ["submission", str(good), str(test_dir)])
    assert main() == 0
    assert "PASS" in capsys.readouterr().out

    rows = _valid_rows()
    rows[0]["z"] = 64
    bad = _write(tmp_path / "bad.csv", rows)
    monkeypatch.setattr(sys, "argv", ["submission", str(bad), str(test_dir)])
    assert main() == 1
    assert "z_out_of_bounds" in capsys.readouterr().out


def test_non_utf8_file_is_reported_not_raised(tmp_path: Path, test_dir: Path) -> None:
    path = tmp_path / "submission.csv"
    path.write_bytes(b"id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
                     b"0,\xff\xfe,node,1,0,10,100,100,-1,-1\n")
    report = validate_submission(path, test_dir)
    assert _codes(report) == {"encoding"}


def test_non_utf8_past_the_header_is_reported_not_raised(
    tmp_path: Path, test_dir: Path
) -> None:
    """The same defect as above, but past the header line.

    A text-mode ``readline`` decodes a whole buffer chunk, so on a small file
    the header check happens to see a bad byte in row two. It does not on a
    286 MB submission, where the bad byte reaches the CSV reader instead — so
    this writes enough rows to clear the buffer and pins the reader's own
    failure being reported rather than propagated.
    """
    path = tmp_path / "submission.csv"
    with path.open("wb") as handle:
        handle.write((",".join(COLUMNS) + "\n").encode())
        for i in range(5000):
            handle.write(
                f"{i},{DATASETS[0]},node,{i + 1},{i % 10},10,{i % 200},100,-1,-1\n".encode()
            )
        handle.write(b"5000,\xff\xfe,node,5001,0,10,100,100,-1,-1\n")
    assert path.stat().st_size > 8192, "file must exceed one text-mode read buffer"
    report = validate_submission(path, test_dir)
    assert _codes(report) == {"encoding"}


def test_row_with_extra_fields_is_reported_not_raised(
    tmp_path: Path, test_dir: Path
) -> None:
    """A ragged row makes the CSV reader itself fail; that must land in the report."""
    path = tmp_path / "submission.csv"
    body = ",".join(str(v) for v in [0, DATASETS[0], "node", 1, 0, 10, 100, 100, -1, -1])
    path.write_text(f"{','.join(COLUMNS)}\n{body}\n{body},999\n", encoding="utf-8")
    report = validate_submission(path, test_dir)
    assert _codes(report) == {"unreadable"}


def test_non_consecutive_id_warns_and_a_parse_error_does_not_fake_one(
    tmp_path: Path, test_dir: Path
) -> None:
    """``id`` is throwaway, so a gap only warns — but it must not warn falsely.

    The consecutive check runs on the rows that parsed, so without a guard every
    ``non_integer`` error would drag a bogus ``id_not_consecutive`` along with
    it and teach us to ignore the warnings.
    """
    rows = _valid_rows()
    path = tmp_path / "submission.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for i, row in enumerate(rows):
            writer.writerow([i + 100] + [row[c] for c in COLUMNS[1:]])
    report = validate_submission(path, test_dir)
    assert report.ok, str(report)
    assert "id_not_consecutive" in {p.code for p in report.warnings}

    rows = _valid_rows()
    rows[0]["z"] = "nan"
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    assert "non_integer" in _codes(report)
    assert "id_not_consecutive" not in {p.code for p in report.warnings}, (
        "a dropped row must not be reported as a gap in the id index"
    )


def test_byte_order_mark_is_caught(tmp_path: Path, test_dir: Path) -> None:
    """A BOM corrupts the first column name; that must not pass silently."""
    path = tmp_path / "submission.csv"
    path.write_text("﻿" + ",".join(COLUMNS) + "\n", encoding="utf-8")
    report = validate_submission(path, test_dir)
    assert _codes(report) == {"header"}


# --------------------------------------------------------------------------- #
# expect_datasets -- coverage narrowed for a CV fold, guards left intact
# --------------------------------------------------------------------------- #

def test_expect_datasets_narrows_coverage(tmp_path: Path, test_dir: Path) -> None:
    """A CV run predicts a fold, not the whole directory.

    Without this the validator would report ~190 of the 199 training samples as
    missing on every CV run, and a validator that cries wolf on the cheap path
    is one nobody runs on the expensive one.
    """
    rows = [r for r in _valid_rows() if r["dataset"] == DATASETS[0]]
    path = _write(tmp_path / "submission.csv", rows)

    assert "missing_dataset" in _codes(validate_submission(path, test_dir))
    narrowed = validate_submission(path, test_dir, expect_datasets=[DATASETS[0]])
    assert narrowed.ok, str(narrowed)
    assert not narrowed.problems, str(narrowed)


def test_expect_datasets_reports_a_name_the_caller_asked_for(
    tmp_path: Path, test_dir: Path
) -> None:
    """Narrowing is not a licence to drop a dataset the fold does contain."""
    rows = [r for r in _valid_rows() if r["dataset"] == DATASETS[0]]
    report = validate_submission(
        _write(tmp_path / "submission.csv", rows), test_dir, expect_datasets=DATASETS
    )
    assert "missing_dataset" in _codes(report)


def test_expect_datasets_does_not_weaken_the_other_checks(
    tmp_path: Path, test_dir: Path
) -> None:
    """Coverage is the only thing it narrows.

    Shapes and the unknown-name check still come from test_dir, so the 4x
    coordinate guard cannot be switched off by passing a short dataset list.
    """
    rows = [r for r in _valid_rows() if r["dataset"] == DATASETS[0]]
    rows[0]["z"] = 64  # outside the stub volume
    rows.append(_node("cccc_0003", 1, 0, 10, 100, 100))  # no such test folder
    report = validate_submission(
        _write(tmp_path / "submission.csv", rows), test_dir,
        expect_datasets=[DATASETS[0]],
    )
    assert {"z_out_of_bounds", "unknown_dataset"} <= _codes(report)


def test_expect_datasets_rejects_a_bare_string(submission: Path, test_dir: Path) -> None:
    """``str`` is an ``Iterable[str]``, so the single-dataset mistake type-checks.

    Passing ``expect_datasets=name`` instead of ``[name]`` iterated the
    characters and reported one missing dataset per character -- loud, but the
    report named ``'0'``, ``'_'`` and friends rather than the mistake. Raising
    keeps it loud and says what to fix.
    """
    with pytest.raises(TypeError, match="not one string"):
        validate_submission(submission, test_dir, expect_datasets=DATASETS[0])
    with pytest.raises(TypeError, match="not one string"):
        assert_valid_submission(submission, test_dir, expect_datasets=DATASETS[0])
    # A one-element collection is the correct spelling and is accepted.
    narrowed = validate_submission(submission, test_dir, expect_datasets=(DATASETS[0],))
    assert narrowed.ok and not narrowed.problems, str(narrowed)


def test_assert_valid_forwards_expect_datasets(tmp_path: Path, test_dir: Path) -> None:
    rows = [r for r in _valid_rows() if r["dataset"] == DATASETS[0]]
    path = _write(tmp_path / "submission.csv", rows)
    with pytest.raises(SubmissionError):
        assert_valid_submission(path, test_dir)
    assert assert_valid_submission(path, test_dir, expect_datasets=[DATASETS[0]]).ok


# --------------------------------------------------------------------------- #
# the coordinate-space advice -- it is read by a human about to edit a pipeline
# --------------------------------------------------------------------------- #

def test_isotropic_advice_names_a_function_that_actually_inverts_the_downsample(
    tmp_path: Path, test_dir: Path
) -> None:
    """The message is the fix instructions, so a wrong name introduces the bug.

    io.rescale_graph_to_original inverts an interpolating *resample* by dividing
    out its zoom. The 1,4,4 path is a strided read with no zoom to invert --
    coord_original = coord_grid * downsample exactly -- so reaching for the
    scale-based function there is how the 4x error gets written. CLAUDE.md still
    recommends it for this, so the message may not simply omit the name: a
    reader who looks it up must find the disclaimer here rather than reinvent
    the bug. See tests/test_coordinate_roundtrip.py for which inverse pairs with
    which forward transform.
    """
    rows = _shrunk_nodes(DATASETS[0]) + _shrunk_nodes(DATASETS[1])
    report = validate_submission(_write(tmp_path / "submission.csv", rows), test_dir)
    advice = next(p.message for p in report.problems if p.code == "suspect_isotropic_coords")
    assert "upsample=DOWNSAMPLE" in advice and "build_graph" in advice
    assert "NOT io.rescale_graph_to_original" in advice, (
        "naming it as the fix is the defect; naming it as the trap is the point"
    )


def test_module_docstring_does_not_recommend_the_broken_rescaler() -> None:
    """The docstring is the other place a reader looks; it must not mislead either."""
    import volt_biohub.submission as mod

    doc = mod.__doc__ or ""
    assert "rescale_graph_to_original" in doc, "the trap is worth naming explicitly"
    where = doc.index("rescale_graph_to_original")
    assert "does **not** do this job" in doc[where:where + 120], (
        "the only mention of it must be the warning, not a recommendation"
    )
    assert "linking.build_graph" in doc


# --------------------------------------------------------------------------- #
# the id check, rewritten to work from a Series so the full table can be freed
# --------------------------------------------------------------------------- #

def test_duplicate_id_reports_every_offending_row(tmp_path: Path, test_dir: Path) -> None:
    """Count is rows, examples are the distinct offending ids.

    _check_id_column takes the id column rather than the frame, so that the
    validator can release the whole table before the expensive checks (see the
    module docstring's memory note). This pins that the report did not get
    weaker in the process.
    """
    rows = _valid_rows()
    path = tmp_path / "submission.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        # Two ids used twice each, so count is 4 rows over 2 distinct values.
        ids = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8]
        for i, row in enumerate(rows):
            writer.writerow([ids[i]] + [row[c] for c in COLUMNS[1:]])
    report = validate_submission(path, test_dir)
    dup = next(p for p in report.errors if p.code == "duplicate_id")
    assert dup.count == 4
    assert dup.examples == ("id=0", "id=1")


def test_isolated_nodes_are_pruned_and_ids_stay_consecutive():
    """Detections with no edge cost score and can never earn it.

    They are counted in `num_pred_nodes`, so they pay the node-count multiplier,
    and they compete in the per-timepoint bipartite assignment where one can take
    the match a correctly-linked detection needed -- costing two edges. Our first
    submission carried 3.1-6.9 % of them per dataset.
    """
    from submission import dataset_rows

    coords = [[(0, 0, 0), (1, 1, 1), (2, 2, 2)],   # index 1 is isolated
              [(0, 0, 1), (1, 1, 2), (2, 2, 3)]]   # index 1 is isolated
    links = [[(0, 0, 0.5), (2, 2, 0.5)]]

    rows = list(dataset_rows("emb_x", coords, links))
    nodes = [r for r in rows if r["row_type"] == "node"]
    edges = [r for r in rows if r["row_type"] == "edge"]

    assert len(nodes) == 4, "the two isolated detections should be gone"
    assert [n["node_id"] for n in nodes] == [1, 2, 3, 4], "ids must stay consecutive from 1"

    ids = {n["node_id"] for n in nodes}
    for e in edges:
        assert e["source_id"] in ids and e["target_id"] in ids, "edge lost its endpoint"
    # The surviving t=1 detections are the originals at index 0 and 2.
    assert sorted((n["t"], n["x"]) for n in nodes) == [(0, 0), (0, 2), (1, 1), (1, 3)]


def test_pruning_is_skipped_when_nothing_linked():
    """Every test dataset must appear in the submission, even a hopeless one.

    With no links at all, pruning isolated nodes would delete the dataset
    entirely -- turning a bad score into a rejected file.
    """
    from submission import dataset_rows

    rows = list(dataset_rows("emb_x", [[(0, 0, 0), (1, 1, 1)]], [[]]))
    assert len([r for r in rows if r["row_type"] == "node"]) == 2


def test_pruning_can_be_turned_off():
    from submission import dataset_rows

    coords = [[(0, 0, 0), (1, 1, 1)], [(0, 0, 1), (1, 1, 2)]]
    links = [[(0, 0, 0.5)]]
    rows = list(dataset_rows("emb_x", coords, links, drop_isolated=False))
    assert len([r for r in rows if r["row_type"] == "node"]) == 4
