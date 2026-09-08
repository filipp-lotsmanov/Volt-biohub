"""Validate a submission CSV before it is uploaded.

A silent format error costs a submission slot, and near the entry deadline it
costs the competition. Kaggle rejects or misreads the file without telling us
which invariant broke, and the scorer's own habits make several defects *quiet*
rather than fatal: edges with ``dt != 1`` are deleted, out-degree above two is
truncated to two arbitrary edges, and merge duplicates are dropped. Those cost
score with no error anywhere. This module makes all of it loud, locally, in
seconds.

**Why a report rather than an exception.** Producing the CSV is hours of
inference over a train-sized hidden test set, so the useful failure mode is
"here is everything wrong with it", not "here is the first thing wrong with
it" — fix-one-and-rerun is the expensive loop. A report also lets a check be
advisory: the isotropic-coordinate and filename checks are heuristics or
context-dependent, and promoting them to exceptions would train us to bypass
the validator. ``assert_valid_submission`` is the raising wrapper to wire into
save paths, so nothing has to remember to inspect the return value.

The invariants come from CLAUDE.md's "Submission format" section, checked
against ``sample_submission.csv`` (the format oracle). Note in particular that
node ids need only be unique **within a dataset** — the oracle restarts them at
1 for every dataset and ``csv_to_geffs`` groups by dataset before resolving
edges, so global uniqueness is *not* required and must not be enforced.

**What actually inverts the ``1,4,4`` downsample** — stated here because
``suspect_isotropic_coords`` is user-facing advice and pointing a teammate at
the wrong function would *introduce* the 4x coordinate bug this module exists to
catch. Three sites do it, all by the same multiply, all verified:
``volt_biohub.linking.build_graph(coords, links, upsample=DOWNSAMPLE)`` on the
scoring path, ``predict_unet_transformer.predict_video`` inline
(``coords[:, 1:] *= ds_arr``), and ``kaggle/submission.py``'s ``upsample``
argument on the offline submission path. ``scripts/geffs_to_csv.py`` scales
nothing — it rounds whatever the ``.geff`` already holds, so the graph must
reach it in original space.

``io.rescale_graph_to_original`` does **not** do this job, whatever CLAUDE.md's
"Submission format" section still says. It inverts an *interpolating resample*
— ``open_dataset(target_scale=...)`` or
``img_proc.rescale_coords_to_isotropic`` — by dividing out that resample's
zoom. The ``1,4,4`` path never resamples: it is a strided read, so
``coord_original = coord_grid * downsample`` exactly and there is no scale
arithmetic to invert. Reaching for the scale-based function on the strided grid
is how the 4x error gets written: with the ``min(scale)`` convention it divides
z by 4 and leaves y and x at a quarter of their true value, wrong in both axes
and silent. Both conventions are pinned in
``tests/test_coordinate_roundtrip.py``; that file is the reference for which
inverse pairs with which forward transform.

**Cost, measured, because it lands on the submission path.** Sized from the
data rather than guessed: ``estimated_number_of_nodes`` sums to **4.73 M** over
the 199 training samples (mean 23.7 k, median 17.9 k — right-skewed, so sizing
a whole set off the *median* sample understates it by 1.3x). A 473 MB /
9.43 M-row CSV at that total (200 datasets x 23.7 k nodes + 23.5 k edges)
validates in **~22 s** and peaks at **3.5-3.7 GB RSS, 7.4-7.8x the file size**
— the spread is how many parse threads polars gets on a shared box. Kaggle's
~30 GB survives that with room to spare, but the multiplier is worth stating
rather than discovering, because it is *not* the 0.26 GB the frame reports
through ``estimated_size``. Two items dominate and neither is a copy we make:
``pl.read_csv`` alone reaches 1.7 GB — parallel-parse buffers plus allocator
arenas never returned to the OS, so that is a floor every later check builds
on — and the edge-resolution double join adds ~0.8 GB of hash tables. The
frames we *could* avoid holding are worth ~1.2 GB, which is why
``validate_submission``, ``_check_edges`` and ``_check_id_column`` release
theirs explicitly. Deleting the four ``del`` statements and re-running one
357 MB fixture moves the peak from 2.50-2.71 GB to 3.66-3.82 GB — **7.2x
before, 10.4x after** — so they are load-bearing, not tidying, and a cleanup
that removes one is a regression. A typed rather
than all-String read would save a further 0.55 GB but cannot tell ``100.5`` from
``100``, the one thing the string pass exists to catch, so it was rejected. If
this ever has to shrink further the direction is a per-dataset pass, not one
whole-file frame.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import zarr

SUBMISSION_FILENAME = "submission.csv"

COLUMNS: tuple[str, ...] = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
)
INT_COLUMNS: tuple[str, ...] = tuple(c for c in COLUMNS if c not in ("dataset", "row_type"))

# Fields that carry no meaning for a row type and must hold the -1 sentinel.
NODE_SENTINELS: tuple[str, ...] = ("source_id", "target_id")
EDGE_SENTINELS: tuple[str, ...] = ("node_id", "t", "z", "y", "x")

# Below this many nodes in a dataset, the isotropic-coordinate heuristic in
# _check_node_coords is switched off: a handful of nodes can sit in a corner of
# the volume by chance, so the y/x span carries no signal until there are
# enough of them for "spans only a quarter of the volume" to mean something.
_ISOTROPIC_MIN_NODES = 50

_MAX_EXAMPLES = 5


@dataclass(frozen=True)
class Problem:
    """One failed check. ``count`` is the total, ``examples`` a sample of it."""

    code: str
    level: str  # "error" or "warning"
    message: str
    count: int = 1
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"[{self.level.upper()}] {self.code}: {self.message}"
        if self.count > 1:
            head += f" ({self.count} occurrences)"
        if self.examples:
            head += "\n    e.g. " + "\n    e.g. ".join(self.examples)
        return head


@dataclass
class SubmissionReport:
    """Everything wrong with one submission CSV, plus what it contains."""

    csv_path: Path
    problems: list[Problem] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "warning"]

    @property
    def ok(self) -> bool:
        """True when nothing that would break the grader was found."""
        return not self.errors

    def __str__(self) -> str:
        lines = [f"{self.csv_path}: {'PASS' if self.ok else 'FAIL'}"]
        if self.stats:
            lines.append(
                "  {n_datasets} datasets, {n_nodes} nodes, {n_edges} edges".format(**self.stats)
            )
        lines.extend(f"  {p}" for p in self.problems)
        if self.ok and not self.warnings:
            lines.append("  no problems found")
        return "\n".join(lines)


class SubmissionError(AssertionError):
    """Raised by ``assert_valid_submission`` when the report holds errors."""

    def __init__(self, report: SubmissionReport) -> None:
        super().__init__(str(report))
        self.report = report


def assert_valid_submission(
    csv_path: Path | str,
    test_dir: Path | str,
    expect_datasets: Iterable[str] | None = None,
) -> SubmissionReport:
    """``validate_submission``, raising ``SubmissionError`` on any error.

    This is the line to add to a save path — a validator whose return value can
    be ignored is a validator that will be ignored::

        from volt_biohub.submission import assert_valid_submission
        assert_valid_submission(csv_path, test_dir)                     # submission
        assert_valid_submission(csv_path, data_dir, expect_datasets=names)  # CV run

    Warnings are returned, never raised. ``kaggle/submission.py`` shows the
    stronger ordering for the real submission — write to a staging path,
    validate *that*, then rename into place — so a rejected file never leaves a
    plausible-looking ``submission.csv`` for the notebook to hand in.
    """
    report = validate_submission(csv_path, test_dir, expect_datasets)
    if not report.ok:
        raise SubmissionError(report)
    return report


def validate_submission(
    csv_path: Path | str,
    test_dir: Path | str,
    expect_datasets: Iterable[str] | None = None,
) -> SubmissionReport:
    """Check ``csv_path`` against every submission invariant; return a report.

    ``test_dir`` is a directory of ``*.zarr`` volumes. It supplies two things no
    amount of self-consistency checking can: the set of dataset names that must
    be present (the folder names without ``.zarr``), and each volume's shape,
    which turns the catastrophic-but-silent coordinate-space bugs into bounds
    errors.

    ``expect_datasets`` narrows only the *coverage* requirement. Default
    ``None`` means every dataset in ``test_dir`` must appear, which is the real
    submission's rule. A CV run predicts a fold — a handful of names out of the
    199 sitting in the data directory — so without this it would be told 190
    datasets are missing, and a validator that cries wolf on the CV path is one
    nobody runs on the submission path. Shapes and the unknown-name check still
    come from ``test_dir`` either way, so the coordinate-space guard is never
    weakened by narrowing coverage.

    Never raises for a defect in the CSV — that is what the report is for,
    including a file polars cannot parse at all. It does raise
    ``FileNotFoundError`` for a missing ``test_dir``, which is a caller mistake
    rather than a submission defect.
    """
    if isinstance(expect_datasets, str):
        # ``str`` satisfies ``Iterable[str]``, so ``expect_datasets=name`` for a
        # single dataset type-checks and then expands to one missing-dataset
        # error per character. Raised rather than reported for the same reason a
        # missing ``test_dir`` is: it is a caller mistake, not a CSV defect.
        raise TypeError(
            "expect_datasets must be a collection of dataset names, not one "
            f"string; pass [{expect_datasets!r}]"
        )
    csv_path = Path(csv_path)
    test_dir = Path(test_dir)
    report = SubmissionReport(csv_path=csv_path)
    shapes = _test_volume_shapes(test_dir)

    if csv_path.name != SUBMISSION_FILENAME:
        # Advisory: Kaggle enforces the name at the upload location, so a local
        # scratch CSV under another name is legitimate.
        report.problems.append(Problem(
            "filename", "warning",
            f"file is named {csv_path.name!r}; the competition requires exactly "
            f"{SUBMISSION_FILENAME!r} at submission time",
        ))

    if not csv_path.exists():
        report.problems.append(Problem("missing_file", "error", f"no such file: {csv_path}"))
        return report

    header_problem = _check_header(csv_path)
    if header_problem is not None:
        # Every later check indexes by column name, so a bad header is fatal.
        report.problems.append(header_problem)
        return report

    try:
        raw = pl.read_csv(csv_path, infer_schema_length=0)
    except pl.exceptions.PolarsError as exc:
        # Reported rather than propagated, because the module's contract is
        # "one pass surfaces every defect" and a raw polars traceback surfaces
        # none of them — and because this runs inside the Kaggle notebook's
        # save path, where the exception would read as a library failure rather
        # than a malformed submission. Two real defects land here: a row with
        # more fields than the header, and invalid UTF-8 anywhere past the
        # header line.
        detail = str(exc).strip()
        code = "encoding" if "utf-8" in detail.lower() else "unreadable"
        report.problems.append(
            Problem(code, "error", f"the CSV could not be read: {detail}")
        )
        return report
    if raw.height == 0:
        report.problems.append(Problem("empty", "error", "file has a header but no data rows"))
        return report
    n_rows = raw.height  # before unparseable rows are dropped, for the id check
    raw = raw.with_row_index("line", offset=2)  # CSV line number, header included

    typed = _check_integer_columns(raw, report)
    # ``raw`` holds the all-String copy of every integer column, ~0.5 GB at
    # hidden-test scale, and nothing below reads it. Dropped here rather than
    # left to fall out of scope at return: the frames that follow are what the
    # peak is made of, and the allocator can only reuse what has been released.
    del raw
    nodes, edges = _split_row_types(typed, report)
    ids = typed["id"]
    names = typed["dataset"].unique()
    report.stats = {
        "n_datasets": names.len(),
        "n_nodes": nodes.height,
        "n_edges": edges.height,
        "nodes_per_dataset": dict(
            nodes.group_by("dataset").len().sort("dataset").iter_rows()
        ),
    }
    # ``nodes`` and ``edges`` partition ``typed``, so keeping it alive across
    # the heavy checks below doubles the table for nothing.
    del typed

    _check_sentinels(nodes, edges, report)
    _check_datasets(names, shapes, expect_datasets, report)
    _check_node_ids(nodes, report)
    _check_node_coords(nodes, shapes, report)
    _check_duplicate_nodes(nodes, report)
    _check_edges(nodes, edges, report)
    _check_id_column(ids, n_rows, report)
    return report


def _test_volume_shapes(test_dir: Path) -> dict[str, tuple[int, ...]]:
    """Map dataset name -> (T, Z, Y, X) for every ``.zarr`` in the test directory.

    Reads array metadata only, so it costs nothing on the real 80 GB tree. The
    test directory is enumerated rather than taken from ``sample_submission.csv``
    because at notebook rerun the placeholder dataset names are replaced.
    """
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test directory not found: {test_dir}")
    shapes: dict[str, tuple[int, ...]] = {}
    for path in sorted(test_dir.glob("*.zarr")):
        shapes[path.stem] = tuple(zarr.open_group(str(path), mode="r")["0"].shape)
    return shapes


def _check_header(csv_path: Path) -> Problem | None:
    """Exact column names in exact order, as in sample_submission.csv.

    Also the BOM gate: a byte-order mark lands in the first column name and
    shows up here rather than as a mysterious later failure.

    Read as bytes and decoded by hand so this check covers exactly the header
    line. A text-mode ``readline`` decodes a whole buffer chunk, so whether a
    bad byte in row three million surfaced here or much later depended on the
    file size — the body's encoding is checked once, at the read below.
    """
    with csv_path.open("rb") as handle:
        first_bytes = handle.readline()
    try:
        first = first_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return Problem("encoding", "error", f"the header line is not valid UTF-8: {exc}")
    got = tuple(first.strip().split(","))
    if got == COLUMNS:
        return None
    return Problem(
        "header", "error",
        f"header must be exactly {','.join(COLUMNS)}; got {','.join(got)}",
    )


def _check_integer_columns(raw: pl.DataFrame, report: SubmissionReport) -> pl.DataFrame:
    """Reject non-integer text, then cast. Returns only the rows that parsed.

    Text-level rather than a strict-schema read because "32.5" must fail loudly:
    coordinates are integers in the submission, and a float would either be
    silently truncated by a lenient reader or abort the whole read on a strict
    one, telling us nothing about how widespread the problem is.
    """
    is_int = pl.all_horizontal(
        pl.col(c).fill_null("").str.contains(r"^-?\d+$") for c in INT_COLUMNS
    )
    bad = raw.filter(~is_int)
    if bad.height:
        report.problems.append(Problem(
            "non_integer", "error",
            "every column except dataset and row_type must be a plain integer",
            count=bad.height,
            examples=_examples(bad, ["line", *INT_COLUMNS]),
        ))
    good = raw.filter(is_int)
    return good.with_columns(pl.col(c).cast(pl.Int64) for c in INT_COLUMNS)


def _split_row_types(
    typed: pl.DataFrame, report: SubmissionReport
) -> tuple[pl.DataFrame, pl.DataFrame]:
    unknown = typed.filter(~pl.col("row_type").is_in(["node", "edge"]))
    if unknown.height:
        report.problems.append(Problem(
            "row_type", "error",
            "row_type must be 'node' or 'edge'",
            count=unknown.height,
            examples=_examples(unknown, ["line", "dataset", "row_type"]),
        ))
    return (
        typed.filter(pl.col("row_type") == "node"),
        typed.filter(pl.col("row_type") == "edge"),
    )


def _check_sentinels(
    nodes: pl.DataFrame, edges: pl.DataFrame, report: SubmissionReport
) -> None:
    """Unused fields must be -1, per sample_submission.csv."""
    for frame, kind, columns in (
        (nodes, "node", NODE_SENTINELS),
        (edges, "edge", EDGE_SENTINELS),
    ):
        bad = frame.filter(pl.any_horizontal(pl.col(c) != -1 for c in columns))
        if bad.height:
            report.problems.append(Problem(
                f"{kind}_sentinel", "error",
                f"{kind} rows must carry -1 in {', '.join(columns)}",
                count=bad.height,
                examples=_examples(bad, ["line", "dataset", *columns]),
            ))


def _check_datasets(
    names: pl.Series,
    shapes: dict[str, tuple[int, ...]],
    expect_datasets: Iterable[str] | None,
    report: SubmissionReport,
) -> None:
    """Every expected dataset present, and no name the test directory does not know.

    A dataset yielding zero nodes still has to appear, so emit a placeholder
    node for it rather than skipping it — an absent dataset is the failure this
    check exists to catch. The usual way to fail the other direction is a
    ``.zarr`` suffix left on the name.

    Coverage is measured against ``expect_datasets`` when given and against the
    whole of ``test_dir`` otherwise; the unknown-name half always uses
    ``test_dir``, since a name with no volume behind it has no shape to
    bounds-check against and is a defect in either mode.
    """
    present = set(names.to_list())
    expected = set(shapes) if expect_datasets is None else set(expect_datasets)
    missing = sorted(expected - present)
    if missing:
        report.problems.append(Problem(
            "missing_dataset", "error",
            "every test dataset must appear in the CSV, even one predicting zero nodes",
            count=len(missing),
            examples=tuple(missing[:_MAX_EXAMPLES]),
        ))
    unknown = sorted(present - set(shapes))
    if unknown:
        report.problems.append(Problem(
            "unknown_dataset", "error",
            "dataset must match a test folder name without '.zarr'",
            count=len(unknown),
            examples=tuple(unknown[:_MAX_EXAMPLES]),
        ))


def _check_node_ids(nodes: pl.DataFrame, report: SubmissionReport) -> None:
    """Unique *within* each dataset, and never the -1 edge sentinel."""
    duplicated = nodes.filter(nodes.select("dataset", "node_id").is_duplicated())
    if duplicated.height:
        report.problems.append(Problem(
            "duplicate_node_id", "error",
            "node_id must be unique within each dataset (global uniqueness is not required)",
            count=duplicated.height,
            examples=_examples(duplicated.sort("dataset", "node_id"), ["line", "dataset", "node_id"]),
        ))
    negative = nodes.filter(pl.col("node_id") < 0)
    if negative.height:
        report.problems.append(Problem(
            "negative_node_id", "error",
            "node_id must be non-negative; -1 is the edge-row sentinel and cannot be referenced",
            count=negative.height,
            examples=_examples(negative, ["line", "dataset", "node_id"]),
        ))


def _check_node_coords(
    nodes: pl.DataFrame, shapes: dict[str, tuple[int, ...]], report: SubmissionReport
) -> None:
    """Coordinates inside the volume, in ORIGINAL anisotropic voxel space.

    The isotropic/original round trip is the highest-risk arithmetic in the
    project: wrong by 4x in z, no exception raised, score near zero. Bounds
    against the real volume shape catch the direction that overflows; the
    heuristic below catches the direction that silently shrinks.

    Checked across all datasets at once rather than per dataset: a coordinate
    bug is global, and the hidden test set is train-sized, so per-dataset
    problems would bury the report in a couple of hundred copies of one finding.
    """
    if not shapes or not nodes.height:
        return
    axes = ("t", "z", "y", "x")
    extents = pl.DataFrame(
        {
            "dataset": list(shapes),
            **{f"{axis}_extent": [shapes[name][i] for name in shapes]
               for i, axis in enumerate(axes)},
        },
        schema={"dataset": pl.String, **{f"{a}_extent": pl.Int64 for a in axes}},
    )
    # Inner join: a dataset the test directory does not know has no extent to
    # check against and is already reported as unknown.
    known = nodes.join(extents, on="dataset", how="inner")

    for axis in axes:
        bad = known.filter(
            (pl.col(axis) < 0) | (pl.col(axis) >= pl.col(f"{axis}_extent"))
        )
        if bad.height:
            report.problems.append(Problem(
                f"{axis}_out_of_bounds", "error",
                f"{axis} must be in [0, {axis}_extent) for its dataset; coordinates "
                "belong in original anisotropic voxel space",
                count=bad.height,
                examples=_examples(bad, ["line", "dataset", axis, f"{axis}_extent"]),
            ))

    suspect = (
        known.group_by("dataset")
        .agg(
            pl.len().alias("n"),
            pl.col("y").max().alias("y_max"),
            pl.col("x").max().alias("x_max"),
            pl.col("y_extent").first(),
            pl.col("x_extent").first(),
        )
        .filter(
            (pl.col("n") >= _ISOTROPIC_MIN_NODES)
            & (pl.col("y_max") < pl.col("y_extent") / 4)
            & (pl.col("x_max") < pl.col("x_extent") / 4)
        )
        .sort("dataset")
    )
    if suspect.height:
        # A coordinate-space bug is global: it fires on every dataset at once.
        # One or two datasets instead look like a genuinely localised lineage,
        # which is possible in a sparse prediction, so only the global pattern
        # is treated as fatal. That keeps a heuristic from blocking a valid
        # submission while still stopping the 4x-in-z error that scores ~0.
        n_datasets = known["dataset"].n_unique()
        level = "error" if suspect.height >= max(2, n_datasets / 2) else "warning"
        report.problems.append(Problem(
            "suspect_isotropic_coords", level,
            f"{suspect.height} of {n_datasets} datasets span less than a quarter of "
            "the volume in y and x while z is unscaled — the signature of isotropic "
            "1,4,4 coordinates. Multiply z,y,x by the checkpoint's downsample "
            "factor: linking.build_graph(coords, links, upsample=DOWNSAMPLE) on the "
            "scoring path, or kaggle/submission.py's own upsample argument on the "
            "submission path. NOT io.rescale_graph_to_original: that inverts an "
            "interpolating resample, and a strided read has no zoom to invert "
            "(see this module's docstring)",
            count=suspect.height,
            examples=_examples(suspect, ["dataset", "n", "y_max", "x_max", "y_extent"]),
        ))


def _check_duplicate_nodes(nodes: pl.DataFrame, report: SubmissionReport) -> None:
    keys = ("dataset", "t", "z", "y", "x")
    duplicated = nodes.filter(nodes.select(keys).is_duplicated())
    if duplicated.height:
        report.problems.append(Problem(
            "duplicate_node", "error",
            "two nodes share a dataset and position; only one can ever match a GT node",
            count=duplicated.height,
            examples=_examples(duplicated.sort(keys), ["line", *keys]),
        ))


def _check_edges(
    nodes: pl.DataFrame, edges: pl.DataFrame, report: SubmissionReport
) -> None:
    """Endpoints resolve, dt == 1, out-degree <= 2, no duplicates.

    Each of these is silently *repaired* by the scorer rather than rejected —
    a dt != 1 edge is deleted, out-degree above two is truncated to the two
    lowest edge ids, duplicate edges collapsed — so every one is lost score
    that nothing would otherwise report.
    """
    if not edges.height:
        return

    keys = ("dataset", "source_id", "target_id")
    duplicated = edges.filter(edges.select(keys).is_duplicated())
    if duplicated.height:
        report.problems.append(Problem(
            "duplicate_edge", "error",
            "the same source->target pair appears twice; the scorer dedupes it "
            "(metrics.py:61, keeping whichever copy matched) so the extra rows are "
            "wasted, and it is a different defect from the merge collapse below",
            count=duplicated.height,
            examples=_examples(duplicated.sort(keys), ["line", *keys]),
        ))

    # Dedupe the lookup so a duplicated node_id (already reported) cannot fan
    # the join out and turn one defect into thousands of phantom ones.
    lookup = nodes.select("dataset", "node_id", "t").unique(
        subset=["dataset", "node_id"], keep="first"
    )
    resolved = edges.join(
        lookup.rename({"node_id": "source_id", "t": "t_source"}),
        on=["dataset", "source_id"], how="left",
    ).join(
        lookup.rename({"node_id": "target_id", "t": "t_target"}),
        on=["dataset", "target_id"], how="left",
    )

    unresolved = resolved.filter(
        pl.col("t_source").is_null() | pl.col("t_target").is_null()
    )
    if unresolved.height:
        report.problems.append(Problem(
            "unresolved_edge", "error",
            "source_id and target_id must both name a node row in the same dataset",
            count=unresolved.height,
            examples=_examples(unresolved, ["line", *keys]),
        ))

    bad_dt = resolved.filter((pl.col("t_target") - pl.col("t_source")) != 1)
    if bad_dt.height:
        report.problems.append(Problem(
            "edge_dt", "error",
            "every edge must go strictly t -> t+1; the scorer deletes any other edge. "
            "Express gap closing by inserting an interpolated node",
            count=bad_dt.height,
            examples=_examples(bad_dt, ["line", *keys, "t_source", "t_target"]),
        ))
    # The double join is the single largest allocation in the validator (~0.8 GB
    # of hash tables at hidden-test scale) and the degree checks below do not
    # need it. Releasing it here lets them reuse that memory instead of stacking
    # on top of it.
    del resolved, lookup, bad_dt, unresolved

    out_degree = edges.group_by("dataset", "source_id").len().filter(pl.col("len") > 2)
    if out_degree.height:
        report.problems.append(Problem(
            "out_degree", "error",
            "at most 2 outgoing edges per node; the scorer truncates to the two lowest "
            "edge ids, which is insertion-order dependent",
            count=out_degree.height,
            examples=_examples(out_degree.sort("dataset", "source_id"),
                               ["dataset", "source_id", "len"]),
        ))

    in_degree = edges.group_by("dataset", "target_id").len().filter(pl.col("len") > 1)
    if in_degree.height:
        # Not a format violation, so not an error: the scorer tolerates merges.
        # But a cell with two parents is impossible, so it is almost always an
        # id bug worth seeing before upload.
        report.problems.append(Problem(
            "merge", "warning",
            "node has more than one incoming edge (a merge); biologically impossible "
            "and usually an id bug",
            count=in_degree.height,
            examples=_examples(in_degree.sort("dataset", "target_id"),
                               ["dataset", "target_id", "len"]),
        ))


def _check_id_column(ids: pl.Series, n_rows: int, report: SubmissionReport) -> None:
    """``id`` is a throwaway index, but a duplicated one can break a keyed join.

    Takes the column rather than the frame so ``validate_submission`` can
    release the full table before the expensive checks; the ``line`` numbers the
    other checks quote are recoverable from the id itself, since a duplicate id
    is by definition a value and not a position.

    ``n_rows`` is the row count *before* unparseable rows were dropped. The
    consecutive-index check is only meaningful against the whole file, so it is
    skipped when rows are missing — otherwise every ``non_integer`` error would
    drag a false "ids are not consecutive" warning along with it.
    """
    is_dup = ids.is_duplicated()  # hashes 7 M ids at test scale: compute once
    n_dup = int(is_dup.sum())
    if n_dup:
        report.problems.append(Problem(
            "duplicate_id", "error",
            "the id column must be unique",
            count=n_dup,
            examples=tuple(
                f"id={v}" for v in ids.filter(is_dup).unique().sort().head(_MAX_EXAMPLES)
            ),
        ))
    del is_dup
    if ids.len() != n_rows:
        return
    if ids.len() and (ids.min() != 0 or ids.max() != ids.len() - 1):
        report.problems.append(Problem(
            "id_not_consecutive", "warning",
            f"id should be a consecutive index from 0 as in sample_submission.csv; "
            f"got {ids.min()}..{ids.max()} over {ids.len()} rows",
        ))


def _examples(frame: pl.DataFrame, columns: list[str], n: int = _MAX_EXAMPLES) -> tuple[str, ...]:
    head = frame.head(n).select(columns)
    return tuple(
        " ".join(f"{col}={value}" for col, value in zip(columns, row, strict=True))
        for row in head.iter_rows()
    )


def main() -> int:
    """``python -m volt_biohub.submission <csv> <test_dir>`` -- check a CSV by hand.

    The scripts that write submissions should call ``assert_valid_submission``
    directly; this exists so a file can be checked from a shell without one.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Validate a submission CSV.")
    parser.add_argument("csv", type=Path, help="Submission CSV to check.")
    parser.add_argument("test_dir", type=Path, help="Competition test directory.")
    args = parser.parse_args()

    report = validate_submission(args.csv, args.test_dir)
    print(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
