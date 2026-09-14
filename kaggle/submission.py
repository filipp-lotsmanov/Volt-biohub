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
    drop_isolated: bool = True,
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
    drop_isolated
        Omit detections that carry no edge. They can never earn an edge TP, yet
        they pay the node-count multiplier and can steal a bipartite match from a
        detection that was correctly linked. Ignored when the video produced no
        links at all, since every test dataset must still appear in the file.
    """
    uz, uy, ux = upsample

    # Drop detections that ended up with no edge at all. An isolated node cannot
    # contribute a single edge TP, but it is counted in `num_pred_nodes` so it pays
    # the node-count multiplier, and it competes in the per-timepoint bipartite
    # assignment where it can take the match a correctly-linked detection needed --
    # costing two edges. Measured on our first submission: 3.1-6.9 % of nodes per
    # dataset were isolated (435/13,900, 3,480/50,561, 426/7,100, 1,299/35,200).
    #
    # Guarded, because "emit fewer nodes" has an obvious failure mode: if nothing
    # linked at all we would emit an empty dataset, and every test dataset must be
    # present in the submission. With no links we keep every node instead.
    keep: set[tuple[int, int]] = set()
    for t, links in enumerate(links_per_t):
        for i, j, _dist in links:
            keep.add((t, i))
            keep.add((t + 1, j))
    prune = bool(keep) and drop_isolated

    # Collapse detections that round onto the SAME original-space voxel. Two nodes
    # at one position can never both match a ground-truth node -- the assignment is
    # bipartite -- so the second is pure node-count penalty, and the validator
    # rejects the file outright.
    #
    # This is not hypothetical tidying: it is what made the peak-separation
    # parameter unreachable. Detection runs on the downsampled 1,4,4 grid and
    # coordinates are multiplied back by 4, so two peaks two voxels apart survive
    # suppression and then land on one integer coordinate once `refine` has pulled
    # their sub-voxel centroids together. At 5.86 um separation that never happens;
    # at the 3.0 um the reference implementation uses, it happens immediately and
    # the whole run fails validation.
    #
    # The survivor is the earlier (stronger) peak, since detectors return peaks
    # ordered by response. Edges are remapped onto it rather than dropped, because
    # a duplicate position genuinely is one node.
    rounded: dict[int, dict[int, tuple[int, int, int]]] = {}
    canon: dict[tuple[int, int], int] = {}
    for t, coords in enumerate(coords_per_t):
        seen: dict[tuple[int, int, int], int] = {}
        here: dict[int, tuple[int, int, int]] = {}
        for i, c in enumerate(coords):
            key = (round(float(c[0]) * uz), round(float(c[1]) * uy), round(float(c[2]) * ux))
            if key in seen:
                canon[(t, i)] = seen[key]
            else:
                seen[key] = i
                here[i] = key
        rounded[t] = here

    # Node ids restart at 1 per dataset, matching sample_submission.csv. They must
    # be assigned *after* pruning and stay consecutive, so the index map is built
    # per timepoint rather than from a flat offset.
    # An index that was collapsed onto an earlier duplicate must be kept if ITS
    # survivor carries an edge, so resolve every index through `canon` first.
    resolve = lambda t, i: canon.get((t, i), i)  # noqa: E731
    keep = {(t, resolve(t, i)) for (t, i) in keep}

    ids: list[dict[int, int]] = []
    next_id = 1
    for t in range(len(coords_per_t)):
        here: dict[int, int] = {}
        for i in rounded[t]:
            if prune and (t, i) not in keep:
                continue
            z, y, x = rounded[t][i]
            here[i] = next_id
            next_id += 1
            yield {
                "dataset": name, "row_type": "node", "node_id": here[i],
                "t": t, "z": z, "y": y, "x": x,
                "source_id": _UNUSED, "target_id": _UNUSED,
            }
        ids.append(here)

    # Collapsing nodes can collapse two distinct links onto one edge; the scorer
    # drops merge duplicates silently, so emit each pair once instead.
    emitted: set[tuple[int, int]] = set()
    for t, links in enumerate(links_per_t):
        for i, j, _dist in links:
            src, tgt = ids[t].get(resolve(t, i)), ids[t + 1].get(resolve(t + 1, j))
            if src is None or tgt is None or (src, tgt) in emitted:
                continue
            emitted.add((src, tgt))
            yield {
                "dataset": name, "row_type": "edge", "node_id": _UNUSED,
                "t": _UNUSED, "z": _UNUSED, "y": _UNUSED, "x": _UNUSED,
                "source_id": src, "target_id": tgt,
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
