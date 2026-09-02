"""Tests for the Kaggle download path.

The bulk-archive path is exercised against a synthetic zip built in-process, so
these need neither network access nor the competition data. That matters because
the alternative is discovering a selection bug after an 88 GB transfer.
"""

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import fetch_data as fd

# Mirrors the real competition layout: samples are .zarr/.geff *directories*,
# plus top-level files that belong to no sample.
_SAMPLES = {"train": ["6bba_c328f2fd", "6bba_aaaa1111", "44b6_0113de3b"], "test": ["9zzz_deadbeef"]}
_ROOT_FILE = "sample_submission.csv"


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    members = {_ROOT_FILE: b"id,dataset,row_type\n"}
    for split, samples in _SAMPLES.items():
        for s in samples:
            for tp in range(3):
                members[f"{split}/{s}.zarr/0/c/{tp}/0/0/0"] = b"chunk" * (10 + tp)
            members[f"{split}/{s}.geff/nodes/props/t/values/0"] = b"gt" * 7
    path = tmp_path / fd.ZIP_NAME
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _select_like_cli(entries, **kw):
    """Selection exactly as the `zip` command composes it, root files included."""
    kw.setdefault("samples", None)
    kw.setdefault("embryos", None)
    kw.setdefault("per_embryo", None)
    kw.setdefault("take_all", False)
    chosen = fd.select(entries, **kw)
    return chosen + [e for e in entries if e.sample is None]


def test_manifest_from_archive_reads_central_directory(archive):
    entries = fd.manifest_from_archive(archive)
    assert len(entries) == 17
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.split] = counts.get(e.split, 0) + 1
    assert counts == {"root": 1, "train": 12, "test": 4}
    assert {e.embryo for e in entries if e.embryo} == {"6bba", "44b6", "9zzz"}


def test_manifest_from_archive_rejects_unexpected_layout(tmp_path):
    """A wrapper directory would silently select nothing, so it must fail loudly."""
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("wrapper/train/x.zarr/0", b"x")
    with pytest.raises(SystemExit, match="layout has changed"):
        fd.manifest_from_archive(bad)


def test_extract_takes_only_selected_samples(archive, tmp_path):
    entries = fd.manifest_from_archive(archive)
    chosen = _select_like_cli(entries, split="train", samples=["6bba_c328f2fd"])
    fd.extract(archive, chosen, tmp_path, workers=4, batch=2)

    got = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()} - {archive.name}
    assert sum(1 for g in got if "6bba_c328f2fd" in g) == 4
    assert not any("44b6" in g or "9zzz" in g or "aaaa1111" in g for g in got)


def test_extract_always_keeps_root_files(archive, tmp_path):
    """sample_submission.csv is the submission-format oracle; select() drops it
    because it has no sample stem, so the `zip` command re-adds it by hand."""
    entries = fd.manifest_from_archive(archive)
    chosen = _select_like_cli(entries, split="train", samples=["6bba_c328f2fd"])
    assert any(e.name == _ROOT_FILE for e in chosen)
    fd.extract(archive, chosen, tmp_path, workers=2, batch=8)
    assert (tmp_path / _ROOT_FILE).exists()


def test_extract_is_idempotent_and_repairs_truncated_members(archive, tmp_path):
    entries = fd.manifest_from_archive(archive)
    chosen = _select_like_cli(entries, split="both", take_all=True)
    fd.extract(archive, chosen, tmp_path, workers=4, batch=4)
    assert [e for e in chosen if fd._needs_fetch(e, tmp_path)] == []

    victim = next(e for e in chosen if e.name.endswith(".zarr/0/c/2/0/0/0"))
    (tmp_path / victim.name).write_bytes(b"short")
    assert fd._needs_fetch(victim, tmp_path)
    fd.extract(archive, chosen, tmp_path, workers=2, batch=4)
    assert (tmp_path / victim.name).stat().st_size == victim.size


def test_select_all_spans_both_splits(archive):
    entries = fd.manifest_from_archive(archive)
    chosen = fd.select(entries, split="both", samples=None, embryos=None,
                       per_embryo=None, take_all=True)
    assert len({e.sample for e in chosen}) == 4


def test_select_per_embryo_caps_each_embryo(archive):
    entries = fd.manifest_from_archive(archive)
    chosen = fd.select(entries, split="train", samples=None, embryos=None,
                       per_embryo=1, take_all=False)
    by_embryo: dict[str, set[str]] = {}
    for e in chosen:
        by_embryo.setdefault(e.embryo, set()).add(e.sample)
    assert {k: len(v) for k, v in by_embryo.items()} == {"6bba": 1, "44b6": 1}


def test_zip64_archive_reports_real_sizes(tmp_path):
    """The real archive is 81 GiB, so it is necessarily ZIP64.

    ZIP64 members carry ``0xFFFFFFFF`` sentinels in their local headers and the true
    sizes in an extra field. If those sentinels ever reached ``Entry.size``, the
    idempotency check would compare every local file against 4 GiB and re-extract the
    whole dataset on every run. ``force_zip64`` reproduces that framing without
    writing 81 GiB.
    """
    path = tmp_path / fd.ZIP_NAME
    members = {
        _ROOT_FILE: b"id,dataset,row_type\n",
        "train/6bba_c328f2fd.zarr/0/c/0/0/0/0": b"chunk" * 11,
        "test/9zzz_deadbeef.zarr/0/c/0/0/0/0": b"x" * 33,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name, data in members.items():
            with zf.open(zipfile.ZipInfo(name), "w", force_zip64=True) as fh:
                fh.write(data)

    entries = fd.manifest_from_archive(path)
    assert {e.name: e.size for e in entries} == {k: len(v) for k, v in members.items()}

    out = tmp_path / "out"
    fd.extract(path, entries, out, workers=2, batch=2)
    assert [e for e in entries if fd._needs_fetch(e, out)] == []


def test_write_manifest_round_trips(archive, tmp_path):
    entries = fd.manifest_from_archive(archive)
    fd.write_manifest(entries, tmp_path)
    assert sorted(fd.load_manifest(tmp_path), key=lambda e: e.name) == sorted(
        entries, key=lambda e: e.name
    )
