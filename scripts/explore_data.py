"""Phase A exploration for the biohub cell-tracking competition data.

Reads zarr image metadata and geff ground-truth graphs for every dataset in
train/ and test/. Loads no image volumes by default, so a full run over ~199
datasets is metadata-bound rather than IO-bound.

Outputs (written to --out):
    datasets.csv        one row per dataset, all per-dataset metrics
    summary.json        aggregates, consistency checks, overlap findings
    attrs_sample.json   raw zarr + geff attrs for a few datasets, verbatim

Usage:
    uv run python scripts/explore_data.py
    uv run python scripts/explore_data.py --out reports/eda
    uv run python scripts/explore_data.py --content-probe 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree

# scripts/ is not a package; import the shared path config the same way the
# other scripts do.
sys.path.insert(0, str(Path(__file__).parent))
from dataspec import DATA_ROOT

# Physical voxel size asserted by the competition description, used only as a
# fallback and as the value to compare against what the files actually declare.
DEFAULT_SCALE_ZYX = (1.625, 0.40625, 0.40625)

# Distance at which the leaderboard metric stops crediting a node match.
# This gates predicted-node-to-GT-node assignment. It is NOT a bound on how far
# a cell may move between frames; edge displacement is compared against it only
# for scale.
METRIC_MATCH_UM = 7.0

# Baseline pool_kernel_um: the NMS suppression radius in the released pipeline.
BASELINE_NMS_UM = 3.0

PERCENTILES = [1, 5, 25, 50, 75, 90, 99]


# ---------------------------------------------------------------- discovery


def find_datasets(split_dir: Path) -> list[dict]:
    """Return one record per .zarr directory, pairing it with its .geff if present."""
    if not split_dir.is_dir():
        return []
    out = []
    for zarr_path in sorted(split_dir.glob("*.zarr")):
        name = zarr_path.stem
        geff_path = split_dir / f"{name}.geff"
        prefix, _, suffix = name.partition("_")
        out.append(
            {
                "name": name,
                "split": split_dir.name,
                "prefix": prefix,
                "suffix": suffix,
                "zarr_path": zarr_path,
                "geff_path": geff_path if geff_path.exists() else None,
                "has_geff": geff_path.exists(),
            }
        )
    return out


def dir_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total / (1024**2)


# ------------------------------------------------------------ image metadata


def _find_multiscales(attrs: dict) -> tuple[list | None, str]:
    """Locate the OME multiscales block.

    OME-Zarr 0.4 puts it at the top level; 0.5 (zarr v3) nests it under "ome".
    The released loader (io._parse_scale) only checks the top level, so the
    location determines whether that loader sees the real voxel scale or
    silently falls back to its hardcoded default.
    """
    if "multiscales" in attrs:
        return attrs["multiscales"], "top_level"
    ome = attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        return ome["multiscales"], "under_ome"
    return None, "missing"


def _extract_transforms(multiscales: list | None) -> tuple[dict, dict, list]:
    """Pull scale and translation out of the first dataset's transform list.

    Returns (scale_by_type, translation_by_type, raw_transform_list). Both dicts
    are empty when the corresponding transform type is absent.
    """
    if not multiscales:
        return {}, {}, []
    try:
        transforms = multiscales[0]["datasets"][0]["coordinateTransformations"]
    except (KeyError, IndexError, TypeError):
        return {}, {}, []
    scale, translation = {}, {}
    for t in transforms:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "scale":
            scale = {"values": list(t.get("scale", []))}
        elif t.get("type") == "translation":
            translation = {"values": list(t.get("translation", []))}
    return scale, translation, transforms


def read_image_meta(zarr_path: Path) -> dict:
    """Read shape, dtype, chunking, voxel scale, offset and stored quantiles."""
    row: dict = {"zarr_error": None}
    try:
        group = zarr.open_group(zarr_path, mode="r")
    except Exception as exc:
        row["zarr_error"] = f"{type(exc).__name__}: {exc}"
        return row

    attrs = dict(group.attrs)
    row["attrs_top_level_keys"] = "|".join(sorted(attrs.keys()))

    try:
        levels = sorted(group.array_keys())
    except Exception:
        levels = []
    row["n_multiscale_levels"] = len(levels)
    row["multiscale_levels"] = "|".join(levels)

    level0 = levels[0] if levels else "0"
    try:
        arr = group[level0]
        row["shape"] = tuple(int(s) for s in arr.shape)
        row["dtype"] = str(arr.dtype)
        row["chunks"] = tuple(int(c) for c in arr.chunks)
        row["zarr_format"] = getattr(arr, "metadata", None).zarr_format if hasattr(arr, "metadata") else None
    except Exception as exc:
        row["zarr_error"] = f"level0 read: {type(exc).__name__}: {exc}"
        return row

    if len(row["shape"]) == 4:
        row["T"], row["Z"], row["Y"], row["X"] = row["shape"]
    row["ram_float32_gb"] = float(np.prod(row["shape"])) * 4 / 1024**3

    multiscales, location = _find_multiscales(attrs)
    row["multiscales_location"] = location
    # The released io._parse_scale only reads attrs["multiscales"]; anything else
    # means it falls through to its hardcoded DEFAULT_SCALE without warning.
    row["io_parse_scale_sees_metadata"] = location == "top_level"

    scale, translation, raw = _extract_transforms(multiscales)
    row["n_coordinate_transformations"] = len(raw)
    row["transform_types"] = "|".join(
        t.get("type", "?") for t in raw if isinstance(t, dict)
    )

    if scale.get("values"):
        vals = scale["values"]
        row["scale_source"] = "attrs"
        row["scale_z"], row["scale_y"], row["scale_x"] = (float(v) for v in vals[-3:])
    else:
        row["scale_source"] = "fallback_default"
        row["scale_z"], row["scale_y"], row["scale_x"] = DEFAULT_SCALE_ZYX

    row["scale_matches_competition_doc"] = bool(
        np.allclose(
            [row["scale_z"], row["scale_y"], row["scale_x"]],
            DEFAULT_SCALE_ZYX,
            rtol=1e-6,
        )
    )

    row["has_translation"] = bool(translation.get("values"))
    if row["has_translation"]:
        tv = [float(v) for v in translation["values"]]
        row["translation_raw"] = json.dumps(tv)
        row["offset_t"] = tv[-4] if len(tv) >= 4 else np.nan
        row["offset_z"], row["offset_y"], row["offset_x"] = (
            tv[-3],
            tv[-2],
            tv[-1],
        )
        row["translation_is_zero"] = bool(np.allclose(tv, 0.0))
    else:
        row["translation_raw"] = None
        for k in ("offset_t", "offset_z", "offset_y", "offset_x"):
            row[k] = np.nan
        row["translation_is_zero"] = None

    quant = attrs.get("image_statistics", {})
    quant = quant.get("quantiles", {}) if isinstance(quant, dict) else {}
    row["has_stored_quantiles"] = bool(quant)
    row["quantile_keys"] = "|".join(sorted(quant.keys(), key=_safe_float))
    row["quantile_values"] = json.dumps(
        {k: quant[k] for k in sorted(quant.keys(), key=_safe_float)}
    ) if quant else None
    # io._process_on_gpu looks up q_min=0.000 and q_max=1.000 by numeric key
    # match. If those exact quantiles are not stored, it silently falls back to
    # min/max of a 1-in-50 subsample instead of a robust percentile clip.
    row["io_quantile_lookup_hits"] = _lookup_hits(quant, [0.0, 1.0])
    row["stored_quantile_min"] = _safe_float(row["quantile_keys"].split("|")[0]) if quant else np.nan

    return row


def _safe_float(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("inf")


def _lookup_hits(quant: dict, wanted: list[float], tol: float = 1e-6) -> int:
    hits = 0
    for w in wanted:
        for k in quant:
            if abs(_safe_float(k) - w) <= tol:
                hits += 1
                break
    return hits


# ------------------------------------------------------------- graph reading


def read_graph(geff_path: Path) -> dict:
    """Read node ids, node props and edge pairs from a geff store.

    Props are discovered rather than assumed, so unexpected extras (track_id,
    seg_id, radius) get reported instead of silently dropped.
    """
    out: dict = {"geff_error": None}
    try:
        g = zarr.open_group(geff_path, mode="r")
    except Exception as exc:
        out["geff_error"] = f"{type(exc).__name__}: {exc}"
        return out

    out["geff_attrs"] = dict(g.attrs)

    try:
        out["node_ids"] = np.asarray(g["nodes/ids"][:])
    except Exception as exc:
        out["geff_error"] = f"nodes/ids: {type(exc).__name__}: {exc}"
        return out

    props: dict[str, np.ndarray] = {}
    prop_names: list[str] = []
    missing_masks: list[str] = []
    try:
        prop_group = g["nodes/props"]
        for pname in sorted(prop_group.group_keys()):
            prop_names.append(pname)
            try:
                props[pname] = np.asarray(prop_group[f"{pname}/values"][:])
            except Exception:
                pass
            try:
                if "missing" in set(prop_group[pname].array_keys()):
                    missing_masks.append(pname)
            except Exception:
                pass
    except Exception:
        pass

    out["node_props"] = props
    out["node_prop_names"] = prop_names
    out["props_with_missing_mask"] = missing_masks

    try:
        edges = np.asarray(g["edges/ids"][:])
        out["edges"] = edges.reshape(-1, 2) if edges.size else np.empty((0, 2), dtype=np.int64)
    except Exception:
        out["edges"] = np.empty((0, 2), dtype=np.int64)

    try:
        out["edge_prop_names"] = sorted(g["edges/props"].group_keys())
    except Exception:
        out["edge_prop_names"] = []

    return out


def _tracklet_lengths(node_ids, edges) -> list[int]:
    """Lengths of maximal non-branching chains.

    A tracklet breaks at a division (out-degree > 1) and at a merge
    (in-degree > 1), which is what matters for judging whether a single missed
    detection is expensive.
    """
    children = defaultdict(list)
    parents = defaultdict(list)
    for s, t in edges:
        children[s].append(t)
        parents[t].append(s)

    starts = [
        n for n in node_ids
        if len(parents[n]) != 1 or len(children[parents[n][0]]) > 1
    ]
    lengths = []
    for s in starts:
        n, length = s, 1
        while len(children[n]) == 1:
            nxt = children[n][0]
            if len(parents[nxt]) != 1:
                break
            n = nxt
            length += 1
        lengths.append(length)
    return lengths


def _connected_component_count(node_ids, edges) -> int:
    parent = {int(n): int(n) for n in node_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in edges:
        s, t = int(s), int(t)
        if s in parent and t in parent:
            rs, rt = find(s), find(t)
            if rs != rt:
                parent[rs] = rt
    return len({find(int(n)) for n in node_ids})


def graph_stats(
    graph: dict,
    scale_zyx: tuple[float, float, float],
    n_frames: int | None = None,
) -> dict:
    """Topology and physical-distance statistics for one ground-truth graph.

    Parameters
    ----------
    graph
        Output of :func:`read_graph`.
    scale_zyx
        Physical voxel size, used for every distance reported in microns.
    n_frames
        Image ``T``, needed to separate "per annotated frame" from "per frame
        of video". Omit and only the former is reported.
    """
    row: dict = {"geff_error": graph.get("geff_error")}
    if row["geff_error"]:
        return row

    node_ids = graph["node_ids"]
    edges = graph["edges"]
    props = graph["node_props"]

    row["n_nodes"] = len(node_ids)
    row["n_edges"] = len(edges)
    row["node_prop_names"] = "|".join(graph["node_prop_names"])
    row["edge_prop_names"] = "|".join(graph["edge_prop_names"])
    row["props_with_missing_mask"] = "|".join(graph["props_with_missing_mask"])

    required = ("t", "z", "y", "x")
    if not all(k in props for k in required):
        row["geff_error"] = f"missing coord props, found {graph['node_prop_names']}"
        return row

    t = np.asarray(props["t"]).astype(np.float64)
    zyx = np.stack(
        [np.asarray(props[k]).astype(np.float64) for k in ("z", "y", "x")], axis=1
    )

    row["t_min"], row["t_max"] = float(t.min()), float(t.max())
    row["n_timepoints_with_nodes"] = len(np.unique(t))
    counts = np.bincount(t.astype(np.int64) - int(t.min()))
    counts = counts[counts > 0]
    # Two different denominators, and they differ for every dataset whose
    # annotation does not span the movie (29 start after t=0, 25 end before
    # t=99). The first answers "how crowded is an annotated frame", the second
    # "how many annotated cells per frame of video" -- the detector's per-frame
    # target uses the second, density arguments use the first.
    row["nodes_per_annotated_t_mean"] = float(counts.mean())
    row["nodes_per_t_min"] = int(counts.min())
    row["nodes_per_t_max"] = int(counts.max())
    if n_frames:
        row["nodes_per_T_mean"] = float(len(node_ids) / n_frames)
        row["frac_frames_annotated"] = float(row["n_timepoints_with_nodes"] / n_frames)

    for i, ax in enumerate(("z", "y", "x")):
        row[f"bbox_{ax}_min"] = float(zyx[:, i].min())
        row[f"bbox_{ax}_max"] = float(zyx[:, i].max())

    # Degrees. The metric defines a division as out-degree >= 2; in-degree > 1
    # would be a merge, which should not occur in nuclei lineages.
    id_to_idx = {int(n): i for i, n in enumerate(node_ids)}
    out_deg = np.zeros(len(node_ids), dtype=np.int64)
    in_deg = np.zeros(len(node_ids), dtype=np.int64)
    dt_list = []
    disp_vec_um = []
    scale_arr = np.asarray(scale_zyx, dtype=np.float64)

    for s, tgt in edges:
        si, ti = id_to_idx.get(int(s)), id_to_idx.get(int(tgt))
        if si is None or ti is None:
            continue
        out_deg[si] += 1
        in_deg[ti] += 1
        dt_list.append(t[ti] - t[si])
        # Keep the vector, not just its norm. The match gate is isotropic in
        # microns but the sampling is not (z is 4x coarser), so whether a long
        # edge is axial or lateral decides whether a gated linker can even
        # propose it.
        disp_vec_um.append((zyx[ti] - zyx[si]) * scale_arr)

    row["n_divisions"] = int((out_deg >= 2).sum())
    row["n_out_degree_ge3"] = int((out_deg >= 3).sum())
    row["max_out_degree"] = int(out_deg.max()) if len(out_deg) else 0
    row["n_in_degree_violations"] = int((in_deg > 1).sum())
    row["n_isolated_nodes"] = int(((out_deg == 0) & (in_deg == 0)).sum())
    row["division_rate"] = row["n_divisions"] / row["n_nodes"] if row["n_nodes"] else np.nan

    dt_arr = np.asarray(dt_list) if dt_list else np.empty(0)
    row["n_edges_dt_ne_1"] = int((dt_arr != 1).sum()) if dt_arr.size else 0
    row["max_edge_dt"] = float(dt_arr.max()) if dt_arr.size else np.nan

    vec = np.asarray(disp_vec_um) if disp_vec_um else np.empty((0, 3))
    if vec.size:
        disp = np.linalg.norm(vec, axis=1)
        dz = np.abs(vec[:, 0])
        dxy = np.linalg.norm(vec[:, 1:], axis=1)
        for p in PERCENTILES:
            row[f"disp_um_p{p}"] = float(np.percentile(disp, p))
            row[f"disp_z_um_p{p}"] = float(np.percentile(dz, p))
            row[f"disp_xy_um_p{p}"] = float(np.percentile(dxy, p))
        row["disp_um_max"] = float(disp.max())
        row["disp_um_mean"] = float(disp.mean())
        row["disp_z_um_max"] = float(dz.max())
        row["disp_xy_um_max"] = float(dxy.max())
        row["frac_edges_disp_over_7um"] = float((disp > METRIC_MATCH_UM).mean())
        # Share of total squared displacement carried by the axial component.
        # An isotropic random walk in physical space gives 1/3; well above that
        # means the motion (or the annotation error) is z-dominated, which is
        # what a z-only jump from a mis-linked track looks like.
        ss = float((disp**2).sum())
        row["frac_disp_var_axial"] = float((dz**2).sum() / ss) if ss > 0 else np.nan
        # Same question restricted to the edges a 7 um gate would reject: if
        # those are overwhelmingly axial they are more likely annotation noise
        # than motion the linker should be asked to span.
        far = disp > METRIC_MATCH_UM
        row["n_edges_over_7um"] = int(far.sum())
        if far.any():
            ss_far = float((disp[far] ** 2).sum())
            row["frac_disp_var_axial_over_7um"] = float((dz[far] ** 2).sum() / ss_far)
            row["disp_z_um_max_over_7um"] = float(dz[far].max())
        else:
            row["frac_disp_var_axial_over_7um"] = np.nan
            row["disp_z_um_max_over_7um"] = np.nan
    else:
        row["disp_um_max"] = np.nan
        row["frac_edges_disp_over_7um"] = np.nan
        row["frac_disp_var_axial"] = np.nan
        row["n_edges_over_7um"] = 0

    # Within-timepoint nearest-neighbour distance in physical units. Ground truth
    # is sparsely labelled, so this is an upper bound on true cell density.
    nn = []
    for tv in np.unique(t):
        pts = zyx[t == tv] * scale_arr
        if len(pts) < 2:
            continue
        d, _ = cKDTree(pts).query(pts, k=2)
        nn.append(d[:, 1])
    nn = np.concatenate(nn) if nn else np.empty(0)
    if nn.size:
        for p in PERCENTILES:
            row[f"nn_um_p{p}"] = float(np.percentile(nn, p))
        row["nn_um_min"] = float(nn.min())
        row["frac_nn_under_baseline_nms"] = float((nn < BASELINE_NMS_UM).mean())
        # GT nodes closer than the 7 um match gate to another GT node at the same
        # timepoint are where bipartite assignment can pair a prediction with the
        # wrong GT node. Low values mean matching is essentially unambiguous.
        row["frac_gt_nn_under_match_gate"] = float((nn < METRIC_MATCH_UM).mean())
    else:
        row["nn_um_min"] = np.nan
        row["frac_nn_under_baseline_nms"] = np.nan
        row["frac_gt_nn_under_match_gate"] = np.nan

    lengths = _tracklet_lengths([int(n) for n in node_ids], edges)
    if lengths:
        la = np.asarray(lengths)
        row["n_tracklets"] = len(la)
        row["tracklet_len_median"] = float(np.median(la))
        row["tracklet_len_p90"] = float(np.percentile(la, 90))
        row["tracklet_len_max"] = int(la.max())
        row["frac_tracklets_len1"] = float((la == 1).mean())
    row["n_lineages"] = _connected_component_count([int(n) for n in node_ids], edges)

    return row


def division_events(
    graph: dict,
    scale_zyx: tuple[float, float, float],
    dataset: str,
    n_frames: int | None = None,
) -> list[dict]:
    """One record per ground-truth division, with the geometry a head must learn.

    The division term is worth 0.1 of the score and there are only ~151 events
    in the whole training corpus, so aggregate columns waste them: at that
    count every event can be inspected individually. Each record carries the
    quantities that decide whether an event is detectable and whether the
    metric can credit it.

    ``division_metrics.extract_divisions`` builds each event as
    ``parent -> divider -> child -> grandchild``, so ``has_parent`` and
    ``n_grandchildren`` say how much of that context the ground truth actually
    provides. Grandchild evidence is a fallback for an unmatched child rather
    than a requirement, but an event missing it has strictly less evidence
    available and is the harder case.
    """
    if graph.get("geff_error"):
        return []
    props = graph.get("node_props", {})
    if not all(k in props for k in ("t", "z", "y", "x")):
        return []

    node_ids = [int(n) for n in graph["node_ids"]]
    idx = {n: i for i, n in enumerate(node_ids)}
    t = np.asarray(props["t"]).astype(np.float64)
    zyx = np.stack([np.asarray(props[k]).astype(np.float64) for k in ("z", "y", "x")], axis=1)
    scale_arr = np.asarray(scale_zyx, dtype=np.float64)

    children = defaultdict(list)
    parents = defaultdict(list)
    for s, tgt in graph["edges"]:
        s, tgt = int(s), int(tgt)
        if s in idx and tgt in idx:
            children[s].append(tgt)
            parents[tgt].append(s)

    def um(a: int, b: int) -> float:
        return float(np.linalg.norm((zyx[idx[b]] - zyx[idx[a]]) * scale_arr))

    out = []
    for parent in node_ids:
        kids = children[parent]
        if len(kids) < 2:
            continue
        # Order daughters so "a" is always the one closer to the parent; without
        # a canonical order the two displacement columns are not comparable
        # across events.
        kids = sorted(kids, key=lambda k: um(parent, k))
        a, b = kids[0], kids[1]
        d_ab = um(a, b)
        vec_ab = (zyx[idx[b]] - zyx[idx[a]]) * scale_arr
        rec = {
            "dataset": dataset,
            "parent_id": parent,
            "t": float(t[idx[parent]]),
            "n_children": len(kids),
            "z": float(zyx[idx[parent], 0]),
            "y": float(zyx[idx[parent], 1]),
            "x": float(zyx[idx[parent], 2]),
            "has_parent": bool(parents[parent]),
            "parent_track_len_before": 0,
            # Sister separation one frame after the split. This is the signal a
            # division head keys on, and it must clear the 7 um match gate for
            # the two daughters to be matched to distinct ground-truth nodes.
            "sister_sep_um": d_ab,
            "sister_sep_z_um": float(abs(vec_ab[0])),
            "sister_sep_xy_um": float(np.linalg.norm(vec_ab[1:])),
            "sister_sep_over_match_gate": bool(d_ab > METRIC_MATCH_UM),
            "disp_parent_to_near_um": um(parent, a),
            "disp_parent_to_far_um": um(parent, b),
            "n_grandchildren": sum(len(children[k]) for k in kids),
            "both_children_continue": all(len(children[k]) >= 1 for k in kids),
        }
        # How much history precedes the event: a head that needs context cannot
        # fire on a division in the first few annotated frames.
        n, steps = parent, 0
        while len(parents[n]) == 1 and steps < 1000:
            n = parents[n][0]
            steps += 1
        rec["parent_track_len_before"] = steps
        if n_frames:
            rec["frames_after"] = float(n_frames - 1 - t[idx[parent]])
            rec["t_frac"] = float(t[idx[parent]] / max(n_frames - 1, 1))
        out.append(rec)
    return out


# ----------------------------------------------------------- cross-dataset


def geff_node_count_target(geff_attrs: dict) -> dict:
    """Read ``estimated_number_of_nodes`` from the geff metadata extra.

    This is ``T_true`` in the adjusted edge Jaccard:

        adj = max(0, J * (1 - 0.1 * (N_pred - T_true) / T_true))

    It is a coarse estimate of *all* cells present, not the annotated subset, so
    it is two to three orders of magnitude larger than the GT node count. It is
    the number a detector should aim its total output at, and it is absent from
    the test split, so it has to be estimated at inference time.
    """
    extra = (geff_attrs or {}).get("geff", {}).get("extra", {}) or {}
    val = extra.get("estimated_number_of_nodes")
    try:
        val = float(val)
    except (TypeError, ValueError):
        val = float("nan")
    return {"t_true": val, "geff_extra_keys": "|".join(sorted(extra.keys()))}


def penalty_multiplier(n_pred: float, t_true: float, alpha: float = 0.1) -> float:
    """The factor the leaderboard applies to a sample's edge Jaccard."""
    if not (t_true > 0):
        return float("nan")
    return max(0.0, 1.0 - alpha * (n_pred - t_true) / t_true)


def geff_axis_offsets(geff_attrs: dict) -> dict:
    """Extract per-axis offset/min/max from geff metadata, if recorded.

    This is a second candidate location for crop provenance, independent of the
    OME translation transform.
    """
    out = {}
    axes = (geff_attrs or {}).get("geff", {}).get("axes", [])
    for ax in axes:
        if not isinstance(ax, dict):
            continue
        name = ax.get("name")
        if name in ("t", "z", "y", "x"):
            out[f"geff_{name}_offset"] = ax.get("offset")
            out[f"geff_{name}_min"] = ax.get("min")
            out[f"geff_{name}_max"] = ax.get("max")
    return out


def geff_scale_check(geff_attrs: dict, zarr_scale: tuple[float, float, float]) -> dict:
    """Compare the per-axis scale the GEFF declares against the image's own.

    The GEFF axes carry their own ``scale`` field, independent of the OME
    transform. Every physical distance in this report, and the metric's 7 um
    gate, assumes the two agree. A silent disagreement would put node
    coordinates in a different physical frame from the voxels, so it is worth
    one cheap comparison rather than an assumption.
    """
    out: dict = {"geff_scale_matches_zarr": None}
    axes = (geff_attrs or {}).get("geff", {}).get("axes", [])
    got = {}
    for ax in axes:
        if isinstance(ax, dict) and ax.get("name") in ("z", "y", "x"):
            got[ax["name"]] = ax.get("scale")
    if len(got) == 3 and all(v is not None for v in got.values()):
        declared = [float(got[k]) for k in ("z", "y", "x")]
        out["geff_scale_zyx"] = "|".join(f"{v:g}" for v in declared)
        if all(s is not None for s in zarr_scale):
            out["geff_scale_matches_zarr"] = bool(
                np.allclose(declared, [float(s) for s in zarr_scale], rtol=1e-6)
            )
    return out


def overlap_report(df: pd.DataFrame) -> dict:
    """Assess whether same-prefix crops occupy overlapping regions.

    Uses the OME translation when present. Without it, coordinates are local to
    each crop and the question cannot be settled from metadata alone.
    """
    rep: dict = {}
    have_offsets = df["has_translation"].fillna(False).astype(bool)
    rep["datasets_with_translation"] = int(have_offsets.sum())
    rep["datasets_total"] = len(df)

    geff_off_cols = [c for c in df.columns if c.endswith("_offset")]
    rep["geff_offset_columns_present"] = geff_off_cols
    rep["geff_offsets_all_null"] = bool(
        all(df[c].isna().all() for c in geff_off_cols)
    ) if geff_off_cols else None

    usable = df[have_offsets]
    # An offset of zero is a legitimate crop position; what makes the offsets
    # uninformative is all crops in a prefix sharing the same one.
    informative = []
    for prefix, grp in usable.groupby("prefix"):
        if len(grp) < 2:
            continue
        offs = grp[["offset_z", "offset_y", "offset_x"]].to_numpy(dtype=float)
        if len(np.unique(offs, axis=0)) > 1:
            informative.append(prefix)
    rep["prefixes_with_distinct_offsets"] = informative

    if not informative:
        rep["verdict"] = (
            "inconclusive: no prefix has two or more crops with distinct spatial "
            "offsets, so node coordinates are local to each crop. Use "
            "--content-probe to test overlap by comparing voxel content instead."
        )
        return rep

    findings = []
    for prefix, grp in usable[usable["prefix"].isin(informative)].groupby("prefix"):
        boxes = []
        for _, r in grp.iterrows():
            boxes.append(
                (
                    r["name"],
                    np.array([r["offset_z"], r["offset_y"], r["offset_x"]]),
                    np.array([r["Z"], r["Y"], r["X"]]) * np.array(
                        [r["scale_z"], r["scale_y"], r["scale_x"]]
                    ),
                )
            )
        n_pairs = n_overlap = 0
        ious = []
        for (_, o1, s1), (_, o2, s2) in combinations(boxes, 2):
            n_pairs += 1
            lo = np.maximum(o1, o2)
            hi = np.minimum(o1 + s1, o2 + s2)
            inter = np.prod(np.clip(hi - lo, 0, None))
            if inter > 0:
                n_overlap += 1
                union = np.prod(s1) + np.prod(s2) - inter
                ious.append(inter / union if union else 0.0)
        findings.append(
            {
                "prefix": prefix,
                "n_datasets": len(grp),
                "n_pairs": n_pairs,
                "n_overlapping_pairs": n_overlap,
                "frac_overlapping": n_overlap / n_pairs if n_pairs else 0.0,
                "iou_median": float(np.median(ious)) if ious else 0.0,
                "iou_max": float(np.max(ious)) if ious else 0.0,
            }
        )
    rep["per_prefix"] = findings
    total_pairs = sum(f["n_pairs"] for f in findings)
    total_ov = sum(f["n_overlapping_pairs"] for f in findings)
    rep["verdict"] = (
        f"{total_ov}/{total_pairs} same-prefix pairs overlap spatially. "
        + (
            "Crops are effectively disjoint; a random split is honest."
            if total_ov == 0
            else "Crops overlap; group overlapping crops when building folds, or "
            "accept that a random split mirrors however Kaggle carved the test set."
        )
    )
    return rep


def duplicate_check(records: list[dict]) -> list[dict]:
    """Compare same-named datasets across splits.

    Reads a strided sample of voxels, so it is cheap even for large volumes.
    """
    by_name = defaultdict(dict)
    for r in records:
        by_name[r["name"]][r["split"]] = r

    out = []
    for name, splits in sorted(by_name.items()):
        if len(splits) < 2:
            continue
        entry = {"name": name, "splits": sorted(splits.keys())}
        digests, shapes = {}, {}
        for split, rec in splits.items():
            try:
                g = zarr.open_group(rec["zarr_path"], mode="r")
                key = sorted(g.array_keys())[0]
                arr = g[key]
                T = arr.shape[0]
                sample = arr[:: max(1, T // 3), ::4, ::8, ::8]
                digests[split] = hashlib.md5(
                    np.ascontiguousarray(sample).tobytes()
                ).hexdigest()
                shapes[split] = tuple(int(s) for s in arr.shape)
            except Exception as exc:
                digests[split] = f"error: {type(exc).__name__}"
                shapes[split] = None
        entry["shapes"] = {k: str(v) for k, v in shapes.items()}
        entry["digests"] = digests
        entry["identical"] = len(set(digests.values())) == 1 and len(set(map(str, shapes.values()))) == 1
        out.append(entry)
    return out


def content_probe(df: pd.DataFrame, records: list[dict], n_pairs: int, seed: int = 0) -> dict:
    """Test crop overlap by voxel content when no offsets are recorded.

    Samples same-prefix, same-shape pairs and looks for an exact match on a
    strided central patch. An exact match means the same region of the same
    source volume.
    """
    rng = np.random.default_rng(seed)
    path_by_name = {(r["split"], r["name"]): r["zarr_path"] for r in records}
    train = df[df["split"] == "train"].copy()

    candidates = []
    for _, grp in train.groupby(["prefix", "shape"]):
        names = grp["name"].tolist()
        if len(names) >= 2:
            candidates.extend(combinations(names, 2))
    if not candidates:
        return {"verdict": "no same-prefix, same-shape pairs to compare"}

    idx = rng.choice(len(candidates), size=min(n_pairs, len(candidates)), replace=False)
    picked = [candidates[i] for i in idx]

    cache: dict[str, np.ndarray] = {}

    def patch(name: str) -> np.ndarray | None:
        if name in cache:
            return cache[name]
        try:
            g = zarr.open_group(path_by_name[("train", name)], mode="r")
            arr = g[sorted(g.array_keys())[0]]
            T, Z, Y, X = arr.shape
            p = arr[
                T // 2,
                :: max(1, Z // 8),
                Y // 4 : Y // 4 + min(64, Y),
                X // 4 : X // 4 + min(64, X),
            ]
            cache[name] = np.ascontiguousarray(p)
        except Exception:
            cache[name] = None
        return cache[name]

    n_ident = n_cmp = 0
    examples = []
    for a, b in picked:
        pa, pb = patch(a), patch(b)
        if pa is None or pb is None or pa.shape != pb.shape:
            continue
        n_cmp += 1
        if np.array_equal(pa, pb):
            n_ident += 1
            if len(examples) < 5:
                examples.append([a, b])

    return {
        "pairs_compared": n_cmp,
        "pairs_identical_patch": n_ident,
        "examples": examples,
        "verdict": (
            "no identical central patches found; crops appear to be distinct regions"
            if n_ident == 0
            else f"{n_ident}/{n_cmp} sampled pairs share an identical central patch, "
            "so some crops cover the same region"
        ),
    }


# ------------------------------------------------------------------- driver


def json_safe(obj):
    """Replace non-finite floats with None, recursively.

    Several derived statistics are legitimately undefined -- an axial variance
    share when every separation is zero, a percentile over an empty selection --
    and numpy yields NaN for those. ``json.dump`` writes bare ``NaN`` and
    ``Infinity`` tokens, which Python reads back but which are not valid JSON:
    jq, schema validators and other languages' parsers all reject them. Emitting
    ``null`` keeps the distinction between "undefined" and "zero" while staying
    parseable.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return v if not isinstance(v, float) or math.isfinite(v) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def pooled_percentiles(df: pd.DataFrame, prefix: str) -> dict:
    """Weighted-ish pooled view of a per-dataset percentile family."""
    out = {}
    for p in PERCENTILES:
        col = f"{prefix}_p{p}"
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals):
                out[f"median_of_p{p}"] = float(vals.median())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Directory containing train/ and test/. Default: "
                         "dataspec.DATA_ROOT, which honours $VOLT_DATA_DIR.")
    ap.add_argument("--out", default="reports/eda", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N datasets per split")
    ap.add_argument("--skip-duplicate-check", action="store_true")
    ap.add_argument("--content-probe", type=int, default=0, metavar="N",
                    help="sample N same-prefix pairs and compare voxel patches")
    ap.add_argument("--attrs-samples", type=int, default=4)
    args = ap.parse_args(argv)

    root: Path = args.data_root or DATA_ROOT
    if not root.is_dir():
        print(f"data root not found: {root.resolve()}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for split in ("train", "test"):
        found = find_datasets(root / split)
        records.extend(found[: args.limit] if args.limit else found)
    if not records:
        print(f"no .zarr datasets found under {root.resolve()}", file=sys.stderr)
        return 1

    print(f"found {len(records)} datasets under {root}")

    rows, attrs_sample, div_rows = [], {}, []
    for i, rec in enumerate(records, 1):
        if i % 25 == 0 or i == len(records):
            print(f"  {i}/{len(records)}")

        row = {k: v for k, v in rec.items() if k not in ("zarr_path", "geff_path")}
        row["disk_mb"] = dir_size_mb(rec["zarr_path"])
        row.update(read_image_meta(rec["zarr_path"]))

        if rec["geff_path"] is not None:
            graph = read_graph(rec["geff_path"])
            scale = (row.get("scale_z"), row.get("scale_y"), row.get("scale_x"))
            row.update(graph_stats(graph, scale, n_frames=row.get("T")))
            row.update(geff_axis_offsets(graph.get("geff_attrs", {})))
            row.update(geff_scale_check(graph.get("geff_attrs", {}), scale))
            row.update(geff_node_count_target(graph.get("geff_attrs", {})))
            div_rows.extend(
                division_events(graph, scale, rec["name"], n_frames=row.get("T"))
            )
            if row.get("t_true", 0) > 0 and row.get("n_nodes") is not None:
                row["gt_nodes_over_t_true"] = row["n_nodes"] / row["t_true"]
                row["t_true_per_frame"] = row["t_true"] / max(row.get("T", 1), 1)
            if len(attrs_sample) < args.attrs_samples:
                try:
                    zg = zarr.open_group(rec["zarr_path"], mode="r")
                    attrs_sample[rec["name"]] = {
                        "zarr_attrs": dict(zg.attrs),
                        "geff_attrs": graph.get("geff_attrs", {}),
                    }
                except Exception:
                    pass
        rows.append(row)

    df = pd.DataFrame(rows)
    df["shape"] = df["shape"].astype(str)
    df.to_csv(args.out / "datasets.csv", index=False)

    train = df[df["split"] == "train"]
    labelled = train[train["has_geff"] & train["geff_error"].isna()] if "geff_error" in train else train

    summary = {
        "counts": {
            "total_datasets": len(df),
            "by_split": df["split"].value_counts().to_dict(),
            "by_prefix": df["prefix"].value_counts().to_dict(),
            "by_split_prefix": {
                f"{s}/{p}": int(n)
                for (s, p), n in df.groupby(["split", "prefix"]).size().items()
            },
            "train_with_geff": int(train["has_geff"].sum()),
            "train_without_geff": int((~train["has_geff"]).sum()),
        },
        "geometry": {
            "distinct_shapes": df["shape"].value_counts().to_dict(),
            "distinct_chunks": df["chunks"].astype(str).value_counts().to_dict(),
            "distinct_dtypes": df["dtype"].value_counts().to_dict(),
            "total_disk_gb": float(df["disk_mb"].sum() / 1024),
            "max_ram_float32_gb": float(df["ram_float32_gb"].max()),
        },
        "scale_and_offsets": {
            "multiscales_location": df["multiscales_location"].value_counts().to_dict(),
            "io_parse_scale_sees_metadata": df["io_parse_scale_sees_metadata"].value_counts().to_dict(),
            "scale_source": df["scale_source"].value_counts().to_dict(),
            "distinct_scales": (
                df[["scale_z", "scale_y", "scale_x"]]
                .astype(str)
                .agg("|".join, axis=1)
                .value_counts()
                .to_dict()
            ),
            "all_scales_match_competition_doc": bool(df["scale_matches_competition_doc"].all()),
            "transform_types": df["transform_types"].value_counts().to_dict(),
            "datasets_with_translation": int(df["has_translation"].sum()),
            "geff_scale_matches_zarr": (
                df["geff_scale_matches_zarr"].value_counts(dropna=False).to_dict()
                if "geff_scale_matches_zarr" in df
                else "not checked"
            ),
        },
        "normalization": {
            "has_stored_quantiles": df["has_stored_quantiles"].value_counts().to_dict(),
            "quantile_keys": df["quantile_keys"].value_counts().to_dict(),
            "io_quantile_lookup_hits": df["io_quantile_lookup_hits"].value_counts().to_dict(),
            "note": (
                "io._process_on_gpu requests q=0.000 and q=1.000. Fewer than 2 hits "
                "means it falls back to min/max of a 1-in-50 subsample rather than a "
                "robust percentile clip."
            ),
        },
    }

    if len(labelled) and "n_nodes" in labelled.columns and labelled["n_nodes"].notna().any():
        labelled = labelled[labelled["n_nodes"].notna()]
        summary["ground_truth"] = {
            "total_nodes": int(labelled["n_nodes"].sum()),
            "total_edges": int(labelled["n_edges"].sum()),
            "total_divisions": int(labelled["n_divisions"].sum()),
            "divisions_per_dataset_median": float(labelled["n_divisions"].median()),
            "datasets_with_zero_divisions": int((labelled["n_divisions"] == 0).sum()),
            "division_rate_median": float(labelled["division_rate"].median()),
            "nodes_per_dataset_median": float(labelled["n_nodes"].median()),
            "nodes_per_annotated_frame_median": float(labelled["nodes_per_annotated_t_mean"].median()),
            "nodes_per_video_frame_median": (
                float(labelled["nodes_per_T_mean"].median()) if "nodes_per_T_mean" in labelled else None
            ),
            "frac_frames_annotated_median": (
                float(labelled["frac_frames_annotated"].median())
                if "frac_frames_annotated" in labelled
                else None
            ),
            "datasets_gt_starts_after_t0": int((labelled["t_min"] > 0).sum()),
            "datasets_gt_ends_before_last_frame": int((labelled["t_max"] < labelled["T"] - 1).sum()),
            "node_prop_names": labelled["node_prop_names"].value_counts().to_dict(),
            "edge_prop_names": labelled["edge_prop_names"].value_counts().to_dict(),
            "integrity": {
                "datasets_with_out_degree_ge3": int((labelled["n_out_degree_ge3"] > 0).sum()),
                "datasets_with_in_degree_violations": int((labelled["n_in_degree_violations"] > 0).sum()),
                "total_in_degree_violations": int(labelled["n_in_degree_violations"].sum()),
                "datasets_with_edges_dt_ne_1": int((labelled["n_edges_dt_ne_1"] > 0).sum()),
                "max_edge_dt": float(labelled["max_edge_dt"].max()),
                "total_isolated_nodes": int(labelled["n_isolated_nodes"].sum()),
            },
            "tracklets": {
                "median_of_median_length": (
                    float(labelled["tracklet_len_median"].median())
                    if "tracklet_len_median" in labelled
                    else None
                ),
                "max_length": (
                    int(labelled["tracklet_len_max"].max())
                    if "tracklet_len_max" in labelled
                    else None
                ),
                "median_frac_length_1": (
                    float(labelled["frac_tracklets_len1"].median())
                    if "frac_tracklets_len1" in labelled
                    else None
                ),
            },
        }
        summary["hyperparameter_evidence"] = {
            "edge_displacement_um": pooled_percentiles(labelled, "disp_um"),
            "max_edge_displacement_um": float(labelled["disp_um_max"].max()),
            "median_frac_edges_disp_over_7um": float(labelled["frac_edges_disp_over_7um"].median()),
            "displacement_note": (
                "7 um is the node-match gate, not a bound on cell motion. This "
                "row says how often true frame-to-frame displacement exceeds "
                "that distance, which indicates the range the linker must span."
            ),
            "axial_displacement_um": pooled_percentiles(labelled, "disp_z_um"),
            "lateral_displacement_um": pooled_percentiles(labelled, "disp_xy_um"),
            "frac_disp_variance_axial_median": float(labelled["frac_disp_var_axial"].median()),
            "frac_disp_variance_axial_over_7um_median": (
                float(labelled["frac_disp_var_axial_over_7um"].median())
                if "frac_disp_var_axial_over_7um" in labelled
                else None
            ),
            "n_edges_over_7um_total": (
                int(labelled["n_edges_over_7um"].sum()) if "n_edges_over_7um" in labelled else None
            ),
            "anisotropy_note": (
                "z is sampled 4x coarser than y/x, so an isotropic gate in "
                "microns is not isotropic in voxels. An isotropic random walk "
                "puts 1/3 of its squared displacement on the axial component; "
                "materially more than that, especially among the edges a 7 um "
                "gate would reject, points at axial motion or axial annotation "
                "error rather than lateral motion the linker must span."
            ),
            "nn_distance_um": pooled_percentiles(labelled, "nn_um"),
            "min_nn_distance_um": float(labelled["nn_um_min"].min()),
            "median_frac_nn_under_3um": float(labelled["frac_nn_under_baseline_nms"].median()),
            "datasets_with_nn_undefined": int(labelled["nn_um_min"].isna().sum()),
            "nn_undefined_note": (
                "A dataset has no within-frame nearest-neighbour distance when "
                "it never annotates two cells in the same frame. Those datasets "
                "carry no density or matching-ambiguity evidence and are excluded "
                "from every nn_* aggregate here."
            ),
            "note": (
                "Displacement percentiles bound how far the linker must associate; "
                "compare the high percentiles against the 7.0 um metric gate. NN "
                "percentiles bound pool_kernel_um: mass below 3.0 um means the "
                "baseline NMS radius is deleting real cells."
            ),
        }

    if "t_true" in labelled.columns and labelled["t_true"].notna().any():
        tt = labelled[labelled["t_true"] > 0]
        sec = {
            "note": (
                "t_true is estimated_number_of_nodes from the geff extra: the "
                "coarse count of ALL cells, which the adjusted edge Jaccard "
                "measures N_pred against. It does not exist in the test split, "
                "so it must be estimated at inference time."
            ),
            "t_true_total": float(tt["t_true"].sum()),
            "t_true_min": float(tt["t_true"].min()),
            "t_true_median": float(tt["t_true"].median()),
            "t_true_max": float(tt["t_true"].max()),
            "t_true_per_frame_median": float(tt["t_true_per_frame"].median()),
            "gt_nodes_over_t_true_median": float(tt["gt_nodes_over_t_true"].median()),
            "gt_nodes_over_t_true_min": float(tt["gt_nodes_over_t_true"].min()),
            "gt_nodes_over_t_true_max": float(tt["gt_nodes_over_t_true"].max()),
            "by_prefix": {
                p: {
                    "n": len(g),
                    "t_true_median": float(g["t_true"].median()),
                    "gt_frac_median": float(g["gt_nodes_over_t_true"].median()),
                }
                for p, g in tt.groupby("prefix")
            },
        }
        # What the penalty multiplier would be at various detection counts.
        sec["multiplier_at_fraction_of_t_true"] = {
            f"{f:.2f}x": round(penalty_multiplier(f, 1.0), 4)
            for f in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 11.0)
        }
        # Can t_true be predicted from metadata alone? Correlate it against the
        # stored intensity quantiles, which cost nothing to read.
        try:
            from scipy.stats import spearmanr
            qrows = []
            for _, r in tt.iterrows():
                if isinstance(r.get("quantile_values"), str):
                    qrows.append(json.loads(r["quantile_values"]))
                else:
                    qrows.append({})
            qdf = pd.DataFrame(qrows).reset_index(drop=True)
            target = tt["t_true"].reset_index(drop=True)
            corr = {}
            for col in qdf.columns:
                v = pd.to_numeric(qdf[col], errors="coerce")
                ok = v.notna() & target.notna()
                # spearmanr on a constant column returns NaN; skip rather than
                # emit NaN, which is not valid strict JSON.
                if ok.sum() > 10 and v[ok].nunique() > 1:
                    rho, pv = spearmanr(v[ok], target[ok])
                    if rho == rho:
                        corr[f"quantile_{col}"] = {"spearman": round(float(rho), 3),
                                                   "p": float(f"{pv:.2e}")}
            if {"0.9", "0.1"}.issubset(qdf.columns):
                span = pd.to_numeric(qdf["0.9"], errors="coerce") - pd.to_numeric(qdf["0.1"], errors="coerce")
                ok = span.notna() & target.notna()
                if ok.sum() > 10 and span[ok].nunique() > 1:
                    rho, pv = spearmanr(span[ok], target[ok])
                    if rho == rho:
                        corr["p90_minus_p10"] = {"spearman": round(float(rho), 3),
                                                 "p": float(f"{pv:.2e}")}
            sec["t_true_vs_intensity_spearman"] = corr
            sec["predictability_note"] = (
                "A strong correlation here means t_true can be estimated from "
                "zarr metadata alone at test time, enabling per-dataset "
                "det_threshold calibration. Weak correlations mean it has to "
                "come from the detector output or from image content."
            )
        except Exception as exc:
            sec["t_true_vs_intensity_spearman"] = f"skipped: {type(exc).__name__}"
        summary["node_count_target"] = sec

    if "frac_gt_nn_under_match_gate" in labelled.columns:
        amb = labelled["frac_gt_nn_under_match_gate"].dropna()
        if len(amb):
            summary["matching_ambiguity"] = {
                "note": (
                    "Fraction of GT nodes with another GT node inside the 7 um "
                    "match gate at the same timepoint. These are the only places "
                    "bipartite assignment can pair a prediction with the wrong "
                    "GT node."
                ),
                "median": float(amb.median()),
                "max": float(amb.max()),
                "datasets_with_any": int((amb > 0).sum()),
                "datasets_total": len(amb),
            }

    if div_rows:
        ddf = pd.DataFrame(div_rows)
        ddf.to_csv(args.out / "divisions.csv", index=False)
        # The whole population fits in one table, so report the distribution
        # rather than a single count. Every number here constrains a division
        # head: the separation column says whether the metric can even resolve
        # the daughters, and the context columns bound achievable recall.
        summary["division_events"] = {
            "note": (
                "One row per ground-truth division in divisions.csv. The "
                "division term is 0.1 of the score and this is the entire "
                "population of events, so it is small enough to inspect "
                "individually rather than summarise."
            ),
            "n_events": len(ddf),
            "n_datasets_with_events": int(ddf["dataset"].nunique()),
            "n_children_counts": ddf["n_children"].value_counts().to_dict(),
            "sister_separation_um": {
                f"p{p}": float(np.percentile(ddf["sister_sep_um"], p)) for p in PERCENTILES
            },
            "frac_sisters_over_match_gate": float(ddf["sister_sep_over_match_gate"].mean()),
            "sister_separation_axial_share_median": float(
                (ddf["sister_sep_z_um"] ** 2
                 / (ddf["sister_sep_um"] ** 2).replace(0, np.nan)).median()
            ),
            "parent_to_daughter_um": {
                "near_median": float(ddf["disp_parent_to_near_um"].median()),
                "far_median": float(ddf["disp_parent_to_far_um"].median()),
                "far_p90": float(ddf["disp_parent_to_far_um"].quantile(0.9)),
            },
            "context_available": {
                "frac_with_annotated_parent": float(ddf["has_parent"].mean()),
                "frac_with_both_children_continuing": float(ddf["both_children_continue"].mean()),
                "n_grandchildren_median": float(ddf["n_grandchildren"].median()),
                "parent_track_len_before_median": float(ddf["parent_track_len_before"].median()),
                "note": (
                    "division_metrics.extract_divisions builds each event as "
                    "parent -> divider -> children -> grandchildren. Grandchild "
                    "evidence is a fallback for an unmatched child, not a "
                    "requirement, but events lacking it have strictly less "
                    "evidence available and are the harder half."
                ),
            },
            "timing": {
                "t_frac_median": float(ddf["t_frac"].median()) if "t_frac" in ddf else None,
                "n_in_first_5_frames": int((ddf["t"] < 5).sum()),
                "n_in_last_5_frames": (
                    int((ddf["frames_after"] < 5).sum()) if "frames_after" in ddf else None
                ),
            },
        }

    summary["crop_overlap"] = overlap_report(train)

    # Name collisions across splits cost nothing to detect and invalidate every
    # test-split number if present, so this check is not skippable.
    collisions = sorted(set(df.loc[df.duplicated("name", keep=False), "name"]))
    summary["split_hygiene"] = {
        "duplicate_names_across_splits": collisions,
        "n_duplicate_names": len(collisions),
        "verdict": (
            "clean: no dataset name appears in more than one split"
            if not collisions
            else f"{len(collisions)} dataset name(s) appear in both splits. The local "
            "test/ directory is not an independent test set, so every test-split "
            "number in this report describes copies of training data. Re-run with "
            "--content-probe or the duplicate check to confirm byte identity."
        ),
    }

    if not args.skip_duplicate_check:
        summary["train_test_duplicates"] = duplicate_check(records)
    else:
        summary["train_test_duplicates"] = "skipped (--skip-duplicate-check)"

    if args.content_probe:
        summary["content_probe"] = content_probe(df, records, args.content_probe)

    with open(args.out / "summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2, default=str)
    if attrs_sample:
        with open(args.out / "attrs_sample.json", "w") as f:
            json.dump(json_safe(attrs_sample), f, indent=2, default=str)

    print(f"\nwrote {args.out/'datasets.csv'}")
    print(f"wrote {args.out/'summary.json'}")
    if div_rows:
        print(f"wrote {args.out/'divisions.csv'} ({len(div_rows)} events)")
    if attrs_sample:
        print(f"wrote {args.out/'attrs_sample.json'}")
    print("\n--- key findings ---")
    for section in ("counts", "split_hygiene", "scale_and_offsets", "normalization",
                    "ground_truth", "division_events", "node_count_target",
                    "matching_ambiguity", "hyperparameter_evidence",
                    "crop_overlap", "content_probe"):
        if section in summary:
            print(f"\n[{section}]")
            print(json.dumps(json_safe(summary[section]), indent=2, default=str)[:1800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
