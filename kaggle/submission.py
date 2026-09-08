"""Write a submission CSV from detections and links — stdlib only, on purpose.

``scripts/geffs_to_csv.py`` is the reference writer and cannot run offline: it
loads a ``.geff`` through ``tracksdata.graph.IndexedRXGraph.from_geff`` and
flattens it with polars. The ``.geff`` detour only ever existed to feed the local
scorer, so the offline path writes rows straight from the ``(coords, links)`` the
linker already produced and never materialises a graph at all.

Validation is **not** duplicated here. ``volt_biohub.submission`` is the
authoritative validator, it needs only polars and zarr (both bundled offline),
and it checks more than a writer-side check could — including per-dataset volume
bounds read from the test zarrs. This module's contribution is the ordering:
write to a temporary file, validate *that*, and only then move it into place, so
a rejected submission never leaves a plausible-looking ``submission.csv`` on disk
for the notebook to hand in.

One convention worth stating because it looks like a bug: ``node_id`` restarts at
1 in every dataset. ``sample_submission.csv`` does that, and ``csv_to_geffs.py``
groups by dataset before resolving edges, so ids need only be unique *within* a
dataset. Global uniqueness is not required and not produced.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path

COLUMNS: tuple[str, ...] = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
)

_UNUSED = -1
"""Filler for the fields a row type does not use, per sample_submission.csv."""


def dataset_rows(
    name: str,
    coords_per_t: Sequence[Sequence[Sequence[float]]],
    links_per_t: Sequence[Sequence[tuple[int, int, float]]],
    upsample: tuple[int, int, int] = (1, 1, 1),
) -> Iterator[dict]:
    """Flatten one video's detections and links into submission rows.

    Parameters
    ----------
    name
        Dataset name — the test folder name without ``.zarr``.
    coords_per_t
        One ``(N_t, 3)`` array of ``(z, y, x)`` per timepoint, on whatever
        downsampled grid the detector ran on.
    links_per_t
        ``links_per_t[t]`` holds ``(index_in_t, index_in_t+1, distance)`` triples,
        exactly what :func:`volt_biohub.linking.link_sequence` returns.
    upsample
        Multiplies coordinates back to **original anisotropic voxel space**, the
        only space the scorer and the submission format accept. This is the
        highest-risk line in the offline path: at the default ``1,4,4``
        downsample a missing upsample is wrong by 4x in y and x, raises nothing,
        and scores near zero. ``volt_biohub.submission`` guards both directions:
        bounds-checking against each test volume catches an *over*-scaled result,
        and its ``suspect_isotropic_coords`` check catches the *under*-scaled one
        (y and x spanning under a quarter of the volume while z is unscaled) at
        error level once the pattern is global.
    """
    uz, uy, ux = upsample
    # Node ids restart at 1 per dataset, matching sample_submission.csv. The
    # per-timepoint offsets convert the linker's within-frame indices to them.
    offsets: list[int] = []
    next_id = 1
    for t, coords in enumerate(coords_per_t):
        offsets.append(next_id)
        for i, c in enumerate(coords):
            yield {
                "dataset": name, "row_type": "node", "node_id": next_id + i,
                "t": t,
                "z": round(float(c[0]) * uz),
                "y": round(float(c[1]) * uy),
                "x": round(float(c[2]) * ux),
                "source_id": _UNUSED, "target_id": _UNUSED,
            }
        next_id += len(coords)

    for t, links in enumerate(links_per_t):
        for i, j, _dist in links:
            yield {
                "dataset": name, "row_type": "edge", "node_id": _UNUSED,
                "t": _UNUSED, "z": _UNUSED, "y": _UNUSED, "x": _UNUSED,
                "source_id": offsets[t] + i,
                "target_id": offsets[t + 1] + j,
            }


def _write_csv(rows: Sequence[dict], path: Path) -> None:
    """Write rows with a consecutive throwaway ``id`` index.

    ``lineterminator="\\n"`` is load-bearing, not style. ``csv.writer`` defaults
    to ``\\r\\n``, and with ``newline=""`` that CR reaches the file — so the
    default writer produces a CRLF submission while ``sample_submission.csv``,
    the format oracle, is LF. Most parsers tolerate the difference and Kaggle's
    probably does, but matching the oracle byte for byte is free and the cost of
    being wrong is a submission slot. This was caught by diffing the two headers,
    which is exactly why CLAUDE.md says to diff against the oracle rather than
    trust the spec.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"id": i, **r})


def write_validated(
    rows: Sequence[dict], path: Path | str, test_dir: Path | str, validate: bool = True
):
    """Write, validate, then move into place. Returns the validator's report.

    Staging is ``<parent>/.volt-staging/<same filename>``, not a ``.staging``
    suffix: the validator checks the filename against the required
    ``submission.csv`` and a suffixed name trips that warning on every run, which
    trains the reader to ignore it. A subdirectory of the destination also keeps
    the move a same-filesystem rename, so it cannot half-succeed.

    *validate* exists for one narrow case: a timing probe over a subset of the
    test datasets is legitimately incomplete, and the validator would correctly
    reject it for missing datasets. Never pass False for a real submission.
    """
    path = Path(path)
    if not validate:
        _write_csv(rows, path)
        return f"{path}: NOT VALIDATED (partial run)"

    # Imported here, not at module scope, so a rename in volt_biohub.submission
    # surfaces as a clear message at the one call site instead of breaking the
    # import of the whole offline path.
    try:
        from volt_biohub.submission import assert_valid_submission
    except ImportError as exc:
        raise ImportError(
            "volt_biohub.submission is the authoritative submission validator and "
            "could not be imported. It needs only polars and zarr, both of which "
            "kaggle/requirements-offline.txt bundles — check the wheels dataset "
            "installed. Refusing to write an unvalidated submission."
        ) from exc

    staging = path.parent / ".volt-staging" / path.name
    _write_csv(rows, staging)
    # A failure leaves the staging file in place on purpose: knowing *what* was
    # wrong matters more than a clean directory, and it is not where the grader
    # looks, so it cannot be submitted by accident.
    report = assert_valid_submission(staging, test_dir)
    staging.replace(path)
    # Tidy-up must never fail a run that already produced a valid submission.
    # The failure path above deliberately leaves files behind, so a *different*
    # --out name in the same directory makes this directory non-empty and a bare
    # rmdir() then raises OSError after submission.csv is already in place —
    # measured, not hypothetical. A leftover staging directory is harmless.
    try:
        staging.parent.rmdir()
    except OSError:
        pass
    return report
