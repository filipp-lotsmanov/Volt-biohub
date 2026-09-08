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


def _truncate(path: Path, fraction: float) -> int:
    """Cut *path* down to *fraction* of its length, as an interrupted transfer would.

    A zip's central directory sits at the tail, so this removes it — which is exactly
    why `exists()` cannot stand in for `complete`.
    """
    full = path.stat().st_size
    with path.open("r+b") as fh:
        fh.truncate(int(full * fraction))
    return full


def test_incomplete_archive_is_resumed_not_reused(archive, tmp_path, monkeypatch, capsys):
    """An interrupted transfer must resume; before this, `zip` raised BadZipFile.

    The old code branched on `zip_path.exists()`, so a partial archive was announced
    as "Reusing" and handed straight to `manifest_from_archive`. The only documented
    escape was `--force`, i.e. re-transferring 88 GB from byte 0.
    """
    pristine = tmp_path / "pristine.zip"
    pristine.write_bytes(archive.read_bytes())
    _truncate(archive, 0.45)
    assert not fd._archive_is_readable(archive)

    calls = []

    def fake_download(dest: Path, *, force: bool = False) -> Path:
        # Stands in for the client's Range request completing the transfer.
        calls.append(force)
        (dest / fd.ZIP_NAME).write_bytes(pristine.read_bytes())
        return dest / fd.ZIP_NAME

    monkeypatch.setattr(fd, "download_archive", fake_download)
    fd.main(["zip", "--dest", str(tmp_path), "--samples", "6bba_c328f2fd"])

    assert calls == [False], "must resume (force=False), not re-transfer from byte 0"
    assert "incomplete" in capsys.readouterr().out
    entries = fd.manifest_from_archive(tmp_path / fd.ZIP_NAME)
    wanted = [e for e in entries if e.sample == "6bba_c328f2fd"]
    assert wanted and not [e for e in wanted if fd._needs_fetch(e, tmp_path)]


def test_complete_archive_is_still_reused(archive, tmp_path, monkeypatch, capsys):
    """The reuse path must survive the fix: no transfer when the archive is readable."""
    calls = []
    monkeypatch.setattr(
        fd, "download_archive",
        lambda dest, *, force=False: calls.append(force) or (dest / fd.ZIP_NAME),
    )
    fd.main(["zip", "--dest", str(tmp_path), "--samples", "6bba_c328f2fd"])

    assert calls == [], "a readable archive must not trigger a transfer"
    assert "Reusing" in capsys.readouterr().out


def test_zero_byte_archive_counts_as_incomplete(tmp_path):
    """Disk full on the first chunk leaves an empty file, not a missing one."""
    (tmp_path / fd.ZIP_NAME).write_bytes(b"")
    assert not fd._archive_is_readable(tmp_path / fd.ZIP_NAME)


def test_unreadable_after_resume_exits_instead_of_looping(archive, tmp_path, monkeypatch):
    """One resume attempt, then a clear exit — a server rejecting Range must not spin."""
    _truncate(archive, 0.45)
    monkeypatch.setattr(fd, "download_archive", lambda dest, *, force=False: dest / fd.ZIP_NAME)
    with pytest.raises(SystemExit, match="still unreadable after resuming"):
        fd.main(["zip", "--dest", str(tmp_path), "--samples", "6bba_c328f2fd"])


def test_write_manifest_round_trips(archive, tmp_path):
    entries = fd.manifest_from_archive(archive)
    fd.write_manifest(entries, tmp_path)
    assert sorted(fd.load_manifest(tmp_path), key=lambda e: e.name) == sorted(
        entries, key=lambda e: e.name
    )


def _drop_member(monkeypatch, victim: str, *, raise_instead: bool):
    """Make ``ZipFile.extract`` fail on one member, the two ways it can fail.

    ``raise_instead`` reproduces a batch that throws (what the makedirs race did);
    otherwise the call returns as if it had worked and writes nothing, which is the
    shape of a silent short write. Both must be caught by the same check.
    """
    real = zipfile.ZipFile.extract

    def patched(self, member, path=None, pwd=None):
        if member == victim:
            if raise_instead:
                raise OSError("simulated batch failure")
            return None
        return real(self, member, path, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "extract", patched)


@pytest.mark.parametrize("raise_instead", [True, False])
def test_extract_refuses_to_report_a_partial_extraction(
    archive, tmp_path, monkeypatch, raise_instead
):
    """Returning normally here made a partial 24,886-file dataset look complete.

    A batch that threw was appended to a list, printed, and swallowed — so the only
    thing standing between a half-extracted dataset and a training run was someone
    reading the scrollback.
    """
    entries = fd.manifest_from_archive(archive)
    chosen = _select_like_cli(entries, split="both", take_all=True)
    victim = chosen[3].name
    _drop_member(monkeypatch, victim, raise_instead=raise_instead)

    with pytest.raises(fd.IncompleteTransfer, match="Re-run the same command"):
        fd.extract(archive, chosen, tmp_path, workers=2, batch=4)


def test_zip_command_exits_and_keeps_the_archive_when_extraction_is_partial(
    archive, tmp_path, monkeypatch
):
    """Non-zero exit, and `--rm-zip` must not delete the only local copy."""
    entries = fd.manifest_from_archive(archive)
    victim = next(e.name for e in entries if "6bba_c328f2fd" in e.name)
    _drop_member(monkeypatch, victim, raise_instead=True)
    monkeypatch.setattr(fd, "download_archive", lambda dest, *, force=False: dest / fd.ZIP_NAME)

    with pytest.raises(SystemExit, match="Extraction incomplete"):
        fd.main(["zip", "--dest", str(tmp_path), "--samples", "6bba_c328f2fd", "--rm-zip"])
    assert archive.exists(), "re-running is a local extraction; the archive must survive"


def test_extract_retries_only_the_missing_members(archive, tmp_path, monkeypatch, capsys):
    """The documented remedy — re-run the same command — has to actually work."""
    entries = fd.manifest_from_archive(archive)
    chosen = _select_like_cli(entries, split="both", take_all=True)
    victim = chosen[3].name
    _drop_member(monkeypatch, victim, raise_instead=True)
    with pytest.raises(fd.IncompleteTransfer):
        fd.extract(archive, chosen, tmp_path, workers=2, batch=1)

    monkeypatch.undo()
    capsys.readouterr()
    fd.extract(archive, chosen, tmp_path, workers=2, batch=1)
    assert "1 member(s) to extract" in capsys.readouterr().out
    assert [e for e in chosen if fd._needs_fetch(e, tmp_path)] == []


def test_extract_creates_parent_dirs_itself(archive, tmp_path, monkeypatch):
    """The flake: zipfile's own directory creation is not thread-safe.

    ``_extract_member`` does ``if not os.path.exists(upperdirs): os.makedirs(upperdirs)``
    with no ``exist_ok``, and ``os.makedirs`` re-raises FileExistsError for its leaf.
    Members sharing a parent therefore race — and ``dest`` is itself the parent of
    ``sample_submission.csv``, so extracting a root file alongside anything else into a
    dest that does not exist yet lost a whole batch on roughly 10 % of runs (measured
    at 34/300 with two workers). Extraction must never reach that code path.
    """
    entries = fd.manifest_from_archive(archive)
    chosen = _select_like_cli(entries, split="both", take_all=True)

    calls = []
    real_makedirs = zipfile.os.makedirs

    def watched(name, *a, **kw):
        calls.append(name)
        return real_makedirs(name, *a, **kw)

    monkeypatch.setattr(zipfile.os, "makedirs", watched)
    out = tmp_path / "fresh"  # deliberately absent: the first run on a new machine
    fd.extract(archive, chosen, out, workers=4, batch=1)

    assert calls == [], "zipfile's non-atomic makedirs is back on the extraction path"
    assert [e for e in chosen if fd._needs_fetch(e, out)] == []


@pytest.mark.parametrize("name", ["/etc/passwd", "train/../../x.zarr/0", "C:/x.zarr/0"])
def test_manifest_from_archive_rejects_escaping_member_names(tmp_path, name):
    """A name that escapes `dest` breaks resume, not just safety.

    `ZipFile.extract` rewrites such a name to stay inside its destination; our size
    check and parent-directory creation join it onto `dest` verbatim. The two would
    then disagree about where the file is, so the member would read as missing on
    every re-run — an infinite "re-run to retry".
    """
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("train/6bba_c328f2fd.zarr/0/c/0/0/0/0", b"ok")
        zf.writestr(name, b"x")
    with pytest.raises(SystemExit, match="escape the destination"):
        fd.manifest_from_archive(bad)
