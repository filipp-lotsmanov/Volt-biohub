"""Flatten predicted ``.geff`` graphs into one submission CSV.

Reads each ``.geff`` with **plain zarr** rather than through tracksdata. Two
independent reasons pointed the same way, which is why this is a simplification
and not a second implementation:

1. tracksdata is a git dependency with twenty transitive requirements (ilpy/SCIP,
   a Rust extension, numba). It cannot be bundled for the internet-disabled
   notebook, so anything importing it is unusable there. A ``.geff`` is plain
   Zarr v3 — ``nodes/ids``, ``nodes/props/<k>/values``, ``edges/ids`` — so
   reading one needs nothing but zarr, which is bundled anyway.
2. ``tracksdata``'s ``from_geff`` **drops the z/y/x props of a zero-node graph**
   on read, so this writer raised ``ColumnNotFoundError`` on exactly the empty
   ``.geff`` that ``predict_unet_transformer.run_predictions`` writes for a
   dataset that failed under ``--skip-failures``. The graceful-degradation path
   was therefore a guaranteed total failure. Reading the arrays directly makes an
   empty graph an empty array, and no special case is needed for it.

The layout was checked against both producers — the organizers' ground-truth
files and the ones tracksdata writes — and is byte-identically shaped, only the
prop dtypes differ (int64 for GT, int32/float64 for tracksdata). Coordinates are
rounded to int, which is what the submission format requires and what makes the
round trip through ``csv_to_geffs.py`` exact.

``kaggle/submit.py`` is still the notebook's writer: it goes straight from
``(coords, links)`` to rows and never materialises a graph at all. This script is
the ``.geff``-based reference writer, used by the CV path and by the round-trip
test that pins the two against each other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import zarr

# Column order of the submission CSV (the ``id`` index column is prepended last).
COLUMNS: tuple[str, ...] = (
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
)

AXES: tuple[str, ...] = ("t", "z", "y", "x")

_EMPTY_SCHEMA = {
    c: (pl.String if c in ("dataset", "row_type") else pl.Int64) for c in COLUMNS
}


def read_geff(geff_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Node and edge tables from a ``.geff``, read with plain zarr.

    Returns ``(nodes, edges)`` where *nodes* carries ``node_id, t, z, y, x`` and
    *edges* carries ``source_id, target_id``, all Int64.

    A zero-node graph short-circuits before touching ``nodes/props``: whether a
    writer emits the prop arrays at all for an empty graph is its own business,
    and an empty submission table does not depend on the answer.
    """
    group = zarr.open_group(str(geff_path), mode="r")

    node_ids = np.asarray(group["nodes/ids"]).reshape(-1)
    if node_ids.size == 0:
        nodes = pl.DataFrame(schema={"node_id": pl.Int64,
                                     **dict.fromkeys(AXES, pl.Int64)})
    else:
        nodes = pl.DataFrame({
            "node_id": node_ids.astype(np.int64),
            # Rounded, not truncated: tracksdata stores coordinates as float64
            # and int() would bias every one of them toward the origin.
            **{
                axis: np.rint(
                    np.asarray(group[f"nodes/props/{axis}/values"]).reshape(-1)
                ).astype(np.int64)
                for axis in AXES
            },
        })

    # edges/ids is (E, 2) = (source_id, target_id). An empty graph stores it with
    # shape (0, 2), so the reshape is a guard against a (0,) degenerate rather
    # than a transformation.
    edge_ids = np.asarray(group["edges/ids"]).reshape(-1, 2)
    edges = pl.DataFrame({
        "source_id": edge_ids[:, 0].astype(np.int64),
        "target_id": edge_ids[:, 1].astype(np.int64),
    })
    return nodes, edges


def graph_to_rows(nodes: pl.DataFrame, edges: pl.DataFrame, name: str) -> pl.DataFrame:
    """Flatten one dataset's tables into node rows then edge rows.

    Unused per-row-type fields hold the ``-1`` sentinel, matching
    ``sample_submission.csv``.

    A dataset with no nodes and no edges gets **one placeholder node at the
    origin**. That is a submission-format requirement, not a convenience: every
    test dataset must be *present* in the CSV, a dataset can only appear via a
    row, and a dataset absent from the file invalidates the whole submission —
    so a failed video would otherwise cost all the others. ``(0, 0, 0)`` is
    in-bounds for every volume, and one node against a median ``N_true`` of
    17,909 moves the node-count penalty by nothing. ``kaggle/submit.py`` applies
    the same policy at the volume centre, where it does know the shape.
    """
    if nodes.height == 0 and edges.height == 0:
        return pl.DataFrame({
            "dataset": [name], "row_type": ["node"], "node_id": [1],
            "t": [0], "z": [0], "y": [0], "x": [0],
            "source_id": [-1], "target_id": [-1],
        }, schema=_EMPTY_SCHEMA)

    node_rows = nodes.select(
        pl.lit(name).alias("dataset"),
        pl.lit("node").alias("row_type"),
        pl.col("node_id"),
        pl.col("t"), pl.col("z"), pl.col("y"), pl.col("x"),
        pl.lit(-1, dtype=pl.Int64).alias("source_id"),
        pl.lit(-1, dtype=pl.Int64).alias("target_id"),
    )
    edge_rows = edges.select(
        pl.lit(name).alias("dataset"),
        pl.lit("edge").alias("row_type"),
        pl.lit(-1, dtype=pl.Int64).alias("node_id"),
        pl.lit(-1, dtype=pl.Int64).alias("t"),
        pl.lit(-1, dtype=pl.Int64).alias("z"),
        pl.lit(-1, dtype=pl.Int64).alias("y"),
        pl.lit(-1, dtype=pl.Int64).alias("x"),
        pl.col("source_id"),
        pl.col("target_id"),
    )
    return pl.concat([node_rows, edge_rows])


def geffs_to_csv(in_dir: Path | str, csv_path: Path | str) -> Path:
    """Flatten every ``.geff`` in ``in_dir`` into one submission CSV; return its path."""
    in_dir = Path(in_dir)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    geffs = sorted(in_dir.glob("*.geff"))
    frames: list[pl.DataFrame] = []
    for geff in geffs:
        nodes, edges = read_geff(geff)
        frames.append(graph_to_rows(nodes, edges, geff.stem))
        note = " (empty -> placeholder node)" if not nodes.height and not edges.height else ""
        print(f"{geff.stem}: {nodes.height} nodes, {edges.height} edges{note}")

    table = pl.concat(frames) if frames else pl.DataFrame(schema=_EMPTY_SCHEMA)
    table = table.with_row_index("id")

    table.write_csv(csv_path)
    print(f"\nWrote {table.height} rows from {len(geffs)} geffs to {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten .geff graphs into a submission CSV.")
    parser.add_argument("--in-dir", type=Path, required=True, help="Directory of .geff files.")
    parser.add_argument("--csv", type=Path, required=True, help="Output submission CSV path.")
    args = parser.parse_args()

    geffs_to_csv(args.in_dir, args.csv)


if __name__ == "__main__":
    main()
