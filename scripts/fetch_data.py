#!/usr/bin/env python
"""Download the Biohub competition data from Kaggle.

Each sample is a ``.zarr`` / ``.geff`` **directory** (Zarr v3, one chunk per
timepoint), so the 88 GB dataset is ~25k small files. Two ways to get them, and the
choice is not a preference:

**``zip`` — one bulk request, the default for any real download.** Kaggle serves the
whole competition as a single archive. One connection, no rate limiting, and the
archive's own central directory doubles as the manifest, so this needs no file
listing at all. The transfer is resumable across runs (HTTP Range, validated against
the remote ETag), and extraction is filtered by the same selection flags as ``get``.
Note the archive is all-or-nothing: ``--samples`` narrows what gets *extracted*, not
what gets *transferred*.

**``get`` — per-file, for surgical top-ups only.** Kaggle rate-limits the per-file
endpoint hard, and rejected requests still count against the bucket, so a few hundred
files is the practical ceiling; it is not a way to fetch the dataset. ``--all`` is
refused here and points at ``zip``.

Usage
-----
    # the whole dataset, one request (server: 88 GB archive + 88 GB extracted)
    python scripts/fetch_data.py zip --dest /data/biohub --split both --all

    # a subset: still transfers the whole archive, extracts only these samples
    python scripts/fetch_data.py zip --dest D:/data/biohub --samples 6bba_c328f2fd
    python scripts/fetch_data.py zip --dest /data/biohub --all --rm-zip

    # what is in there (offline once `zip` or `manifest` has cached manifest.csv)
    python scripts/fetch_data.py samples --dest D:/data/biohub

    # progress, from any terminal, for a run started by any shell
    python scripts/fetch_data.py status --dest D:/data/biohub
    python scripts/fetch_data.py status --dest D:/data/biohub --per-embryo 2

    # surgical per-file top-up of one sample (throttled endpoint; keep it small)
    python scripts/fetch_data.py get --dest D:/data/biohub --samples 6bba_c328f2fd

    # cache the file table over the API — only needed without a local archive
    python scripts/fetch_data.py manifest --dest D:/data/biohub

Both paths are resumable and idempotent: a file whose local size already matches is
skipped, so re-running after an interruption is cheap. Neither reports success on a
partial result — an incomplete download or extraction exits non-zero, because a
half-populated data directory is otherwise indistinguishable from the dataset.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

COMPETITION = "biohub-cell-tracking-during-development"
ZIP_NAME = f"{COMPETITION}.zip"  # what competition_download_files writes
MANIFEST_NAME = "manifest.csv"
PARTIAL_NAME = "manifest.partial.csv"
TOKEN_NAME = "manifest.token"

# tqdm redraws one line with \r, which is unreadable once redirected to a log file.
# Off a TTY, emit periodic plain lines instead so a backgrounded run can be tailed.
_TTY = sys.stderr.isatty()
_LOG_EVERY = 25
_EXTRACT_LOG_EVERY = 1000  # ~25k members, so 25 lines for a full extraction

# Kaggle rate-limits the per-file endpoint hard. Rejected requests still count against
# the bucket, so retrying through a 429 storm sustains the throttle instead of riding it
# out: throughput collapses to a crawl and looks like a slow network. Past this many
# consecutive 429s, stop and tell the user to wait — resume is byte-exact and free.
_MAX_CONSECUTIVE_THROTTLES = 12


class Throttled(RuntimeError):
    """Raised when the API is rejecting requests faster than they can succeed."""


class IncompleteTransfer(RuntimeError):
    """Raised when some requested files are not on disk at their recorded size.

    Raised rather than printed. A partial 24,886-file dataset is invisible in the
    output of a run that reports thousands of successes, and a caller that returns
    normally hands a half-populated data directory to `train` — which will happily
    fit on whatever subset arrived. Resume is free (see ``_needs_fetch``), so the
    honest response to a partial result is to fail and say "re-run".
    """

# The API throttles aggressively on both listing and download; treat 429 and the
# transient 5xx family as retryable rather than fatal.
_RETRY_STATUS = {429, 500, 502, 503, 504}

# Competition paths look like:
#   train/{embryo}_{fov}.zarr/0/c/{t}/0/0/0
#   train/{embryo}_{fov}.geff/nodes/props/t/values/0
#   sample_submission.csv
_SAMPLE_SUFFIXES = (".zarr", ".geff")


@dataclass(frozen=True)
class Entry:
    """One downloadable file in the competition."""

    name: str  # full competition-relative path, forward slashes
    size: int

    @property
    def split(self) -> str:
        """``train``, ``test``, or ``root`` for top-level files."""
        head = self.name.split("/", 1)[0]
        return head if head in ("train", "test") else "root"

    @property
    def sample(self) -> str | None:
        """Sample stem (``{embryo}_{fov}``), or None for top-level files."""
        parts = self.name.split("/")
        if len(parts) < 2:
            return None
        for suffix in _SAMPLE_SUFFIXES:
            if parts[1].endswith(suffix):
                return parts[1][: -len(suffix)]
        return None

    @property
    def embryo(self) -> str | None:
        """Embryo id — the first segment of the sample name.

        Train and test are embryo-disjoint upstream, so this is the only correct
        grouping key for cross-validation folds.
        """
        sample = self.sample
        return sample.split("_", 1)[0] if sample else None


def _api():
    """Authenticated Kaggle API client.

    Imported lazily so ``--help`` works without credentials.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


_thread_state = threading.local()


def _thread_api():
    """One Kaggle client per thread.

    ``kagglesdk`` wraps a ``requests`` session that is not thread-safe. Sharing a
    single client across the download pool corrupts its connection pool: sockets get
    abandoned rather than reclaimed (visible as a growing pile of CLOSE_WAIT), and
    after roughly a thousand files the client wedges permanently — alive, reconnecting,
    transferring nothing. Per-thread clients cost one token-file read each.
    """
    api = getattr(_thread_state, "api", None)
    if api is None:
        api = _api()
        _thread_state.api = api
    return api


def _status_of(exc: Exception) -> int | None:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) if resp is not None else None


def _retryable(exc: Exception) -> bool:
    return _status_of(exc) in _RETRY_STATUS


def _retry(fn, *, attempts: int = 7, base: float = 3.0, cap: float = 120.0):
    """Call *fn*, backing off exponentially on throttling and transient 5xx.

    Jittered so parallel workers don't retry in lockstep and re-trigger the limit.
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not _retryable(exc) or i == attempts - 1:
                raise
            delay = min(cap, base * 2**i) * (0.5 + random.random())
            retry_after = (exc.response.headers or {}).get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            time.sleep(delay)
    raise AssertionError("unreachable")


# =============================================================================
# Manifest
# =============================================================================


def build_manifest(
    dest: Path, page_size: int = 200, sleep: float = 1.0,
) -> list[Entry]:
    """Page through the competition file table and cache it as CSV.

    ~25k files means ~125 requests, which the API will throttle, so pages are
    appended to a partial file and the next page token is checkpointed beside it.
    Re-running resumes instead of starting over.
    """
    dest.mkdir(parents=True, exist_ok=True)
    partial, token_path = dest / PARTIAL_NAME, dest / TOKEN_NAME

    entries: list[Entry] = []
    token: str | None = None
    if partial.exists() and token_path.exists():
        with partial.open(newline="", encoding="utf-8") as fh:
            entries = [Entry(name=r["name"], size=int(r["size"])) for r in csv.DictReader(fh)]
        token = token_path.read_text(encoding="utf-8").strip() or None
        print(f"Resuming from {len(entries)} entries already listed.")

    api = _api()
    fresh = not entries
    with partial.open("a" if entries else "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if fresh:
            writer.writerow(["name", "size"])
        with tqdm(
            desc="listing files", unit="file", initial=len(entries), disable=not _TTY
        ) as pbar:
            while True:
                resp = _retry(
                    lambda t=token: api.competition_list_files(
                        COMPETITION, page_token=t, page_size=page_size
                    )
                )
                files = getattr(resp, "files", None) or []
                for f in files:
                    e = Entry(name=str(f.name).replace("\\", "/"), size=int(f.total_bytes))
                    entries.append(e)
                    writer.writerow([e.name, e.size])
                fh.flush()
                pbar.update(len(files))

                token = getattr(resp, "next_page_token", None)
                token_path.write_text(token or "", encoding="utf-8")
                if not token or not files:
                    break
                time.sleep(sleep)

    path = write_manifest(entries, dest)
    partial.unlink(missing_ok=True)
    token_path.unlink(missing_ok=True)

    print(f"Wrote {len(entries)} entries to {path}")
    return entries


def write_manifest(entries: list[Entry], dest: Path) -> Path:
    """Cache *entries* as ``manifest.csv`` so ``samples``/``status`` work offline."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / MANIFEST_NAME
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "size"])
        for e in sorted(entries, key=lambda e: e.name):
            writer.writerow([e.name, e.size])
    return path


def load_manifest(dest: Path) -> list[Entry]:
    """Read the cached manifest, or exit with instructions if absent."""
    path = dest / MANIFEST_NAME
    if not path.exists():
        sys.exit(
            f"No manifest at {path}.\n"
            f"Run: python scripts/fetch_data.py manifest --dest {dest}"
        )
    with path.open(newline="", encoding="utf-8") as fh:
        return [Entry(name=r["name"], size=int(r["size"])) for r in csv.DictReader(fh)]


# =============================================================================
# Selection
# =============================================================================


def summarise(entries: list[Entry]) -> None:
    """Print per-split sample and embryo counts, then a per-sample table."""
    by_sample: dict[tuple[str, str], list[Entry]] = {}
    for e in entries:
        if e.sample:
            by_sample.setdefault((e.split, e.sample), []).append(e)

    for split in ("train", "test"):
        samples = sorted(s for (sp, s) in by_sample if sp == split)
        embryos = sorted({s.split("_", 1)[0] for s in samples})
        total = sum(
            e.size for (sp, s), group in by_sample.items() if sp == split for e in group
        )
        print(
            f"{split}: {len(samples)} samples, {len(embryos)} embryos, "
            f"{total / 2**30:.1f} GiB"
        )
        print(f"  embryos: {', '.join(embryos)}")

    print(f"\n{'split':<6} {'sample':<28} {'files':>6} {'MiB':>9}")
    for (split, sample), group in sorted(by_sample.items()):
        size = sum(e.size for e in group)
        print(f"{split:<6} {sample:<28} {len(group):>6} {size / 2**20:>9.1f}")


def status(entries: list[Entry], dest: Path, expected: list[Entry] | None = None) -> None:
    """Compare the manifest against what is on disk.

    Reads only the filesystem, so it reports honestly on a download started by any
    shell — including one that has since gone away. *expected* restricts the report
    to a selection; without it, every sample with at least one local file is shown.
    """
    pool = expected if expected is not None else entries
    by_sample: dict[tuple[str, str], list[Entry]] = {}
    for e in pool:
        if e.sample:
            by_sample.setdefault((e.split, e.sample), []).append(e)

    rows, tot_have, tot_want, complete = [], 0, 0, 0
    for (split, sample), group in sorted(by_sample.items()):
        want = sum(e.size for e in group)
        have = sum(
            e.size for e in group
            if (dest / e.name).exists() and (dest / e.name).stat().st_size == e.size
        )
        n_have = sum(
            1 for e in group
            if (dest / e.name).exists() and (dest / e.name).stat().st_size == e.size
        )
        if expected is None and n_have == 0:
            continue  # not requested, not started
        tot_have += have
        tot_want += want
        done = n_have == len(group)
        complete += done
        rows.append((split, sample, n_have, len(group), have, want, done))

    if not rows:
        print(f"Nothing downloaded under {dest} yet.")
        return

    print(f"{'split':<6} {'sample':<28} {'files':>11} {'GiB':>13}  state")
    for split, sample, n_have, n_want, have, want, done in rows:
        print(
            f"{split:<6} {sample:<28} {n_have:>5}/{n_want:<5} "
            f"{have / 2**30:>6.2f}/{want / 2**30:<6.2f}  "
            f"{'complete' if done else 'PARTIAL'}"
        )
    pct = 100 * tot_have / tot_want if tot_want else 0.0
    print(
        f"\n{complete}/{len(rows)} samples complete — "
        f"{tot_have / 2**30:.2f} of {tot_want / 2**30:.2f} GiB ({pct:.1f}%)"
    )
    if complete < len(rows):
        # Reached from both `get` and `zip`, so name neither: the remedy is the same
        # command again either way.
        print("Re-run the same command to finish; completed files are skipped.")


def select(
    entries: list[Entry],
    split: str,
    samples: list[str] | None,
    embryos: list[str] | None,
    per_embryo: int | None,
    take_all: bool,
) -> list[Entry]:
    """Filter manifest entries down to the requested samples."""
    pool = [e for e in entries if e.sample and (split == "both" or e.split == split)]

    if take_all:
        return pool

    if samples:
        wanted = set(samples)
        chosen = {e.sample for e in pool if e.sample in wanted}
        missing = wanted - chosen
        if missing:
            sys.exit(f"Not in manifest for split={split}: {sorted(missing)}")
    elif embryos or per_embryo:
        by_embryo: dict[str, set[str]] = {}
        for e in pool:
            if embryos is None or e.embryo in embryos:
                by_embryo.setdefault(e.embryo, set()).add(e.sample)
        chosen = set()
        for embryo in sorted(by_embryo):
            picked = sorted(by_embryo[embryo])
            chosen.update(picked[:per_embryo] if per_embryo else picked)
    else:
        sys.exit("Choose one of --samples, --embryos, --per-embryo, or --all.")

    return [e for e in pool if e.sample in chosen]


# =============================================================================
# Download
# =============================================================================


def _needs_fetch(entry: Entry, dest: Path) -> bool:
    """True unless the file is already on disk at exactly the expected size.

    The size check is what makes both the per-file downloader and the archive
    extractor idempotent, so an interrupted run resumes for free.
    """
    local = dest / entry.name
    return not (local.exists() and local.stat().st_size == entry.size)


def download(entries: list[Entry], dest: Path, workers: int = 6) -> None:
    """Fetch *entries* into ``dest``, preserving competition-relative paths.

    ``competition_download_file`` writes ``<path>/<basename>``, and every timepoint
    chunk is named ``0``, so each file must be given its own parent directory.
    """
    todo = [e for e in entries if _needs_fetch(e, dest)]
    skipped = len(entries) - len(todo)
    total = sum(e.size for e in todo)
    print(
        f"{len(todo)} files to fetch ({total / 2**30:.2f} GiB); "
        f"{skipped} already present"
    )
    if not todo:
        return

    def _one(entry: Entry) -> None:
        target = dest / entry.name
        target.parent.mkdir(parents=True, exist_ok=True)
        api = _thread_api()
        _retry(
            lambda: api.competition_download_file(
                COMPETITION, entry.name, path=str(target.parent), force=True, quiet=True
            )
        )

    failures: list[tuple[str, str]] = []
    done_files = done_bytes = 0
    throttles = 0  # consecutive 429s; reset by any success
    with tqdm(
        total=total, unit="B", unit_scale=True, desc="downloading", disable=not _TTY
    ) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, e): e for e in todo}
            for fut in as_completed(futures):
                entry = futures[fut]
                try:
                    fut.result()
                    throttles = 0
                except Exception as exc:  # network hiccup, transient 5xx
                    if _status_of(exc) == 429:
                        throttles += 1
                    failures.append((entry.name, f"{type(exc).__name__}: {exc}"))
                    if throttles >= _MAX_CONSECUTIVE_THROTTLES:
                        for f in futures:
                            f.cancel()
                        raise Throttled(
                            f"Kaggle returned HTTP 429 for {throttles} consecutive files.\n"
                            f"The per-file endpoint is rate-limited and retrying keeps the "
                            f"bucket empty. Wait (limits are typically hourly), then re-run "
                            f"the same command — {done_files} file(s) fetched this run are "
                            f"already on disk and will be skipped.\n"
                            f"For anything larger than a handful of samples use the "
                            f"bulk archive instead — one connection, no throttling:\n"
                            f"    python scripts/fetch_data.py zip --dest {dest} --all"
                        ) from exc
                pbar.update(entry.size)
                done_files += 1
                done_bytes += entry.size
                if not _TTY and done_files % _LOG_EVERY == 0:
                    print(
                        f"  {done_files}/{len(todo)} files, "
                        f"{done_bytes / 2**30:.2f}/{total / 2**30:.2f} GiB",
                        flush=True,
                    )

    missing = [e for e in todo if _needs_fetch(e, dest)]
    if failures or missing:
        lines = [
            f"Download incomplete: {len(missing)} of {len(todo)} requested file(s) "
            f"missing or short on disk; {len(failures)} request(s) failed."
        ]
        lines += [f"  {name}: {err}" for name, err in failures[:10]]
        lines.append("Re-run the same command to retry; completed files are skipped.")
        raise IncompleteTransfer("\n".join(lines))
    print("All files fetched.")


# =============================================================================
# Bulk archive — the path that actually scales
# =============================================================================


def download_archive(dest: Path, *, force: bool = False) -> Path:
    """Fetch the whole competition as one archive and return its path.

    ``competition_download_files`` streams to ``<dest>/<competition>.zip`` in 1 MiB
    chunks and, with ``force=False``, resumes an interrupted transfer with an HTTP
    Range request — validated against the remote ETag via a ``.kaggle-partial``
    marker, so a stale partial is re-fetched rather than silently concatenated.
    That is the whole reason to prefer this over 25k per-file requests: one
    connection the API does not throttle, and interruption costs nothing.
    """
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / ZIP_NAME
    have = zip_path.stat().st_size / 2**30 if zip_path.exists() else 0.0
    print(
        f"Requesting the competition archive into {zip_path}"
        + (f" (resuming from {have:.2f} GiB)" if have else "")
    )
    # Not wrapped in _retry: download_file already retries chunk reads internally,
    # and a bulk request is not the endpoint that 429s.
    # Off a TTY the client's \r progress bar would write ~90k lines into a log for an
    # 88 GB transfer, so stay quiet there and let `status` report progress instead.
    _api().competition_download_files(
        COMPETITION, path=str(dest), force=force, quiet=not _TTY
    )

    if not zip_path.exists():
        sys.exit(
            f"Kaggle reported success but {zip_path} is missing.\n"
            f"If the competition rules have not been accepted in the browser the API "
            f"returns 403 here."
        )
    print(f"Archive on disk: {zip_path.stat().st_size / 2**30:.2f} GiB")
    return zip_path


def _archive_is_readable(zip_path: Path) -> bool:
    """Whether the archive's central directory can be read.

    This is the invariant the bulk path actually depends on: ``manifest_from_archive``
    and every extraction worker open the central directory, which lives at the *tail*
    of the file, so an interrupted transfer has none. Checking it costs a seek to the
    end plus a few MiB — not a scan of 88 GB.

    Presence on disk is a different property, and the client's ``.kaggle-partial``
    marker is not a reliable substitute either: it is skipped entirely when the server
    sends neither ETag nor Last-Modified, so an interrupted transfer can leave the file
    present and the marker absent. Reading the directory has no such hole, and catches
    a zero-byte file (disk full on the first chunk) for free.
    """
    try:
        with zipfile.ZipFile(zip_path):
            return True
    except zipfile.BadZipFile:
        # Deliberately narrow: a PermissionError or ENOSPC must not be read as
        # "incomplete" and trigger an 88 GB re-transfer.
        return False


def obtain_archive(dest: Path, *, force: bool = False) -> Path:
    """Return a path to a complete, readable archive, transferring only what is missing.

    Four states, not two. A local file that exists is not necessarily complete, and
    treating it as complete is what turns an interrupted 88 GB transfer into a
    ``BadZipFile`` traceback whose only documented remedy is re-transferring from
    byte 0 — the opposite of the resume the client already implements.
    """
    zip_path = dest / ZIP_NAME
    if force or not zip_path.exists():
        return download_archive(dest, force=force)

    if _archive_is_readable(zip_path):
        # Complete: re-extraction is local-only, so this is worth reusing.
        print(f"Reusing {zip_path} ({zip_path.stat().st_size / 2**30:.2f} GiB); "
              f"--force re-downloads.")
        return zip_path

    # Interrupted: the bytes on disk are a prefix of the archive, so the client can
    # resume onto them over HTTP Range. Exactly one attempt — a server that rejects
    # the Range outright must not spin.
    print(
        f"{zip_path} is incomplete ({zip_path.stat().st_size / 2**30:.2f} GiB on disk, "
        f"central directory unreadable) — resuming."
    )
    zip_path = download_archive(dest, force=False)
    if not _archive_is_readable(zip_path):
        sys.exit(
            f"{zip_path} is still unreadable after resuming "
            f"({zip_path.stat().st_size / 2**30:.2f} GiB on disk).\n"
            f"Delete it, or re-transfer from byte 0:\n"
            f"    python scripts/fetch_data.py zip --dest {dest} --force"
        )
    return zip_path


def manifest_from_archive(zip_path: Path) -> list[Entry]:
    """Read the archive's central directory as the file manifest.

    The archive already lists every member with its uncompressed size, so the bulk
    path needs no ``competition_list_files`` paging at all — which is the throttled
    call that made building a manifest slow in the first place.
    """
    with zipfile.ZipFile(zip_path) as zf:
        entries = [
            Entry(name=info.filename.replace("\\", "/"), size=info.file_size)
            for info in zf.infolist()
            if not info.is_dir()
        ]

    if not any(e.sample for e in entries):
        # Guard against Kaggle wrapping everything in a top-level folder: selection
        # keys off `train/{sample}.zarr/...`, and a silent layout change would
        # otherwise extract nothing and look like an empty download.
        tops = sorted({e.name.split("/", 1)[0] for e in entries})[:10]
        sys.exit(
            f"No `train/`- or `test/`-style samples found in {zip_path.name}.\n"
            f"Top-level entries: {tops}\n"
            f"The archive layout has changed; Entry.split/Entry.sample need updating."
        )

    # Every consumer joins these names onto `dest` — `_needs_fetch` for the size
    # check, `extract` to pre-create the parent directory. `ZipFile.extract` instead
    # *rewrites* a name that escapes its destination, so the two would disagree about
    # where the file went and the size check would never converge on "complete", with
    # "re-run to retry" looping forever. Cheaper to refuse the archive.
    escaping = [
        e.name for e in entries
        if e.name.startswith(("/", "\\")) or ".." in e.name.split("/")
        or ":" in e.name.split("/")[0]
    ]
    if escaping:
        sys.exit(
            f"{len(escaping)} member name(s) in {zip_path.name} escape the destination "
            f"directory: {escaping[:5]}"
        )
    return entries


def extract(
    zip_path: Path, entries: list[Entry], dest: Path, workers: int = 8, batch: int = 200
) -> None:
    """Extract *entries* from the archive into ``dest``, skipping what is present.

    ``ZipFile`` is not thread-safe, so each worker opens its own handle — the same
    reason the per-file downloader keeps one API client per thread. Members are zarr
    chunks that arrive already blosc/zstd-compressed, so the zip's own deflate pass
    is close to a passthrough and the cost is dominated by filesystem writes; a
    handle is amortised over a batch of members to keep open/close off the hot path.

    Raises ``IncompleteTransfer`` unless every requested member ends up on disk at
    its recorded size — checked against the filesystem, not against the futures.
    """
    todo = [e for e in entries if _needs_fetch(e, dest)]
    total = sum(e.size for e in todo)
    print(
        f"{len(todo)} member(s) to extract ({total / 2**30:.2f} GiB); "
        f"{len(entries) - len(todo)} already present"
    )
    if not todo:
        return

    batches = [todo[i : i + batch] for i in range(0, len(todo), batch)]

    def _one(group: list[Entry]) -> int:
        with zipfile.ZipFile(zip_path) as zf:
            for e in group:
                # Create the parent ourselves: `ZipFile.extract` uses an
                # `exists()`-then-`os.makedirs()` pair with no `exist_ok` (CPython
                # 3.11), so two workers whose members share a parent directory race
                # and the loser's FileExistsError kills the rest of its batch. `dest`
                # itself is such a parent — sample_submission.csv lands there — so a
                # run into a fresh dest lost a whole batch ~10 % of the time.
                # `Path.mkdir(exist_ok=True)` has no such window.
                (dest / e.name).parent.mkdir(parents=True, exist_ok=True)
                zf.extract(e.name, dest)
        return sum(e.size for e in group)

    failures: list[tuple[str, str]] = []
    done_files = done_bytes = 0
    next_log = _EXTRACT_LOG_EVERY
    with tqdm(
        total=total, unit="B", unit_scale=True, desc="extracting", disable=not _TTY
    ) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, g): g for g in batches}
            for fut in as_completed(futures):
                group = futures[fut]
                try:
                    pbar.update(fut.result())
                except Exception as exc:
                    failures.append((group[0].name, f"{type(exc).__name__}: {exc}"))
                done_files += len(group)
                done_bytes += sum(e.size for e in group)
                if not _TTY and done_files >= next_log:
                    print(
                        f"  {done_files}/{len(todo)} members, "
                        f"{done_bytes / 2**30:.2f}/{total / 2**30:.2f} GiB",
                        flush=True,
                    )
                    next_log = done_files + _EXTRACT_LOG_EVERY

    # Trust the filesystem over the futures: a batch that raised took every member
    # after the throwing one with it, and a short write raises nothing at all. This is
    # the same size check that makes a re-run cheap, so it costs one stat per member.
    missing = [e for e in todo if _needs_fetch(e, dest)]
    if failures or missing:
        lines = [
            f"Extraction incomplete: {len(missing)} of {len(todo)} requested member(s) "
            f"missing or short on disk; {len(failures)} batch(es) raised."
        ]
        lines += [f"  near {name}: {err}" for name, err in failures[:10]]
        lines += [f"  missing: {e.name}" for e in missing[:10]]
        if len(missing) > 10:
            lines.append(f"  ... and {len(missing) - 10} more missing")
        lines.append(
            "Re-run the same command to retry: members already on disk at the right "
            "size are skipped, so only the missing ones are extracted again."
        )
        raise IncompleteTransfer("\n".join(lines))
    print("Extraction complete.")


# =============================================================================
# CLI
# =============================================================================


def _default_dest() -> Path:
    sys.path.insert(0, str(Path(__file__).parent))
    from dataspec import DATA_ROOT

    return Path(DATA_ROOT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command", choices=["zip", "manifest", "samples", "get", "status"],
        help="zip: one bulk archive request, then extract (use this to download). "
             "manifest: cache the file table over the API. samples: summarise it. "
             "get: per-file top-up, rate-limited. "
             "status: compare disk against the manifest.",
    )
    parser.add_argument(
        "--dest", type=Path, default=None,
        help="Data root; train/ and test/ are created inside it. "
             "Default: dataspec.DATA_ROOT, which honours $VOLT_DATA_DIR.",
    )
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated sample stems, e.g. 6bba_c328f2fd.")
    parser.add_argument("--embryos", type=str, default=None,
                        help="Comma-separated embryo ids, e.g. 6bba,44b6.")
    parser.add_argument("--per-embryo", type=int, default=None,
                        help="Take at most N samples from each embryo.")
    parser.add_argument("--all", action="store_true", help="Take every sample in --split.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent per-file downloads for `get` (default: 4). The "
                             "API throttles; raising this trades speed for 429 backoff. "
                             "Also the `zip` extraction thread count, where it is a "
                             "local-CPU knob and 8-16 is reasonable.")
    parser.add_argument("--rm-zip", action="store_true",
                        help="`zip`: delete the archive once extraction is verified "
                             "complete. Halves peak disk (88 GB rather than 176 GB) at "
                             "the cost of a re-download if you later want other samples.")
    parser.add_argument("--force", action="store_true",
                        help="`zip`: re-transfer the archive from byte 0 instead of "
                             "resuming or reusing a complete local copy.")
    parser.add_argument("--page-sleep", type=float, default=1.0,
                        help="Seconds between manifest pages (default: 1.0).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be fetched, then stop.")
    args = parser.parse_args(argv)

    dest = args.dest or _default_dest()

    if args.command == "manifest":
        build_manifest(dest, sleep=args.page_sleep)
        return

    if args.command == "zip":
        zip_path = obtain_archive(dest, force=args.force)
        entries = manifest_from_archive(zip_path)
        write_manifest(entries, dest)

        chosen = select(
            entries,
            split=args.split,
            samples=args.samples.split(",") if args.samples else None,
            embryos=args.embryos.split(",") if args.embryos else None,
            per_embryo=args.per_embryo,
            take_all=args.all,
        )
        # select() keys off the sample stem, so top-level files fall out of it.
        # sample_submission.csv is the submission-format oracle (CLAUDE.md) and the
        # root files are tiny, so always take them.
        chosen = chosen + [e for e in entries if e.sample is None]

        names = sorted({e.sample for e in chosen if e.sample})
        print(f"{len(names)} sample(s) selected for extraction: {', '.join(names)}")
        if args.dry_run:
            print(f"{len(chosen)} members, {sum(e.size for e in chosen) / 2**30:.2f} GiB")
            return

        try:
            extract(zip_path, chosen, dest, workers=args.workers)
        except IncompleteTransfer as exc:
            failure = str(exc)
        else:
            failure = None
        status(entries, dest, chosen)
        if failure:
            # Non-zero exit, and the archive stays: re-running is a local extraction,
            # not another 88 GB transfer.
            sys.exit(failure)
        if args.rm_zip:
            zip_path.unlink()
            print(f"Removed {zip_path}")
        return

    entries = load_manifest(dest)

    if args.command == "samples":
        summarise(entries)
        return

    if args.command == "status":
        # With no selection flags, report on whatever is already on disk.
        any_selection = args.samples or args.embryos or args.per_embryo or args.all
        expected = (
            select(
                entries,
                split=args.split,
                samples=args.samples.split(",") if args.samples else None,
                embryos=args.embryos.split(",") if args.embryos else None,
                per_embryo=args.per_embryo,
                take_all=args.all,
            )
            if any_selection
            else None
        )
        status(entries, dest, expected)
        return

    if args.all:
        sys.exit(
            "`get --all` would issue ~25k requests against the per-file endpoint, which "
            "Kaggle rate-limits into the ground.\n"
            f"Use one bulk request instead:\n"
            f"    python scripts/fetch_data.py zip --dest {dest} --split {args.split} --all"
        )

    chosen = select(
        entries,
        split=args.split,
        samples=args.samples.split(",") if args.samples else None,
        embryos=args.embryos.split(",") if args.embryos else None,
        per_embryo=args.per_embryo,
        take_all=args.all,
    )
    names = sorted({e.sample for e in chosen})
    print(f"{len(names)} sample(s): {', '.join(names)}")
    print(f"{len(chosen)} files, {sum(e.size for e in chosen) / 2**30:.2f} GiB total")

    if args.dry_run:
        return
    try:
        download(chosen, dest, workers=args.workers)
    except IncompleteTransfer as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
