#!/usr/bin/env python
"""Download Biohub competition samples from Kaggle, file by file.

Each sample is a ``.zarr`` / ``.geff`` **directory** (Zarr v3, one chunk per
timepoint) while the Kaggle API serves individual files, so fetching a subset means
enumerating the competition file table and downloading each member separately.
Enumerating ~25k files takes a few minutes, so the manifest is cached on disk and
reused.

Usage
-----
    # once: cache the file table (needed by every other command)
    python scripts/fetch_data.py manifest --dest D:/data/biohub

    # what is in there
    python scripts/fetch_data.py samples --dest D:/data/biohub

    # dev box: a couple of samples, enough for the test fixture + inspection
    python scripts/fetch_data.py get --dest D:/data/biohub --samples 6bba_c328f2fd
    python scripts/fetch_data.py get --dest D:/data/biohub --per-embryo 2 --dry-run

    # progress, from any terminal, for a download started by any shell
    python scripts/fetch_data.py status --dest D:/data/biohub
    python scripts/fetch_data.py status --dest D:/data/biohub --per-embryo 2

    # server: everything — prefer ONE bulk request over 25k per-file ones, which the
    # API rate-limits into the ground:
    #     kaggle competitions download -c biohub-cell-tracking-during-development
    # then unzip into the data root. `get --all` exists but expect 429 throttling.
    python scripts/fetch_data.py get --dest /scratch/biohub --all

Downloads are resumable: a file whose local size already matches the manifest is
skipped, so re-running after an interruption costs only the manifest read.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

COMPETITION = "biohub-cell-tracking-during-development"
MANIFEST_NAME = "manifest.csv"
PARTIAL_NAME = "manifest.partial.csv"
TOKEN_NAME = "manifest.token"

# tqdm redraws one line with \r, which is unreadable once redirected to a log file.
# Off a TTY, emit periodic plain lines instead so a backgrounded run can be tailed.
_TTY = sys.stderr.isatty()
_LOG_EVERY = 25

# Kaggle rate-limits the per-file endpoint hard. Rejected requests still count against
# the bucket, so retrying through a 429 storm sustains the throttle instead of riding it
# out: throughput collapses to a crawl and looks like a slow network. Past this many
# consecutive 429s, stop and tell the user to wait — resume is byte-exact and free.
_MAX_CONSECUTIVE_THROTTLES = 12


class Throttled(RuntimeError):
    """Raised when the API is rejecting requests faster than they can succeed."""

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

    path = dest / MANIFEST_NAME
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "size"])
        for e in sorted(entries, key=lambda e: e.name):
            writer.writerow([e.name, e.size])
    partial.unlink(missing_ok=True)
    token_path.unlink(missing_ok=True)

    print(f"Wrote {len(entries)} entries to {path}")
    return entries


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
        print("Re-run the same `get` command to finish; completed files are skipped.")


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


def _needs_download(entry: Entry, dest: Path) -> bool:
    local = dest / entry.name
    return not (local.exists() and local.stat().st_size == entry.size)


def download(entries: list[Entry], dest: Path, workers: int = 6) -> None:
    """Fetch *entries* into ``dest``, preserving competition-relative paths.

    ``competition_download_file`` writes ``<path>/<basename>``, and every timepoint
    chunk is named ``0``, so each file must be given its own parent directory.
    """
    todo = [e for e in entries if _needs_download(e, dest)]
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
                            f"For the full dataset use one bulk request instead: "
                            f"`kaggle competitions download -c {COMPETITION}`."
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

    if failures:
        print(f"\n{len(failures)} file(s) failed; re-run to retry:")
        for name, err in failures[:10]:
            print(f"  {name}: {err}")
    else:
        print("All files fetched.")


# =============================================================================
# CLI
# =============================================================================


def _default_dest() -> Path:
    sys.path.insert(0, str(Path(__file__).parent))
    from dataspec import DATASET_PATH

    return Path(DATASET_PATH).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "command", choices=["manifest", "samples", "get", "status"],
        help="manifest: cache the file table. samples: summarise it. "
             "get: download. status: compare disk against the manifest.",
    )
    parser.add_argument(
        "--dest", type=Path, default=None,
        help="Data root; train/ and test/ are created inside it. "
             "Default: parent of dataspec.DATASET_PATH.",
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
                        help="Concurrent downloads (default: 4). The API throttles; "
                             "raising this trades speed for 429 backoff.")
    parser.add_argument("--page-sleep", type=float, default=1.0,
                        help="Seconds between manifest pages (default: 1.0).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be fetched, then stop.")
    args = parser.parse_args()

    dest = args.dest or _default_dest()

    if args.command == "manifest":
        build_manifest(dest, sleep=args.page_sleep)
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
    download(chosen, dest, workers=args.workers)


if __name__ == "__main__":
    main()
