"""Data integrity checks for the biohub cell-tracking dataset.

Validates assumptions that the pipeline and the leaderboard metric rely on.
Every finding carries a severity:

    FAIL  violates something the metric or pipeline assumes. Results computed
          on this data would be wrong, not merely noisy.
    WARN  unusual but tolerable. Worth knowing about; not a blocker.
    INFO  observation, no action implied.

Exits non-zero when any FAIL is recorded, so this can gate a pipeline run.

Fast checks read metadata and graphs only. Voxel-reading checks are opt-in via
--deep, because the zarr chunking is one whole 3D volume per timepoint, so
touching every frame of every dataset decompresses roughly 800 MB per dataset.

Run:
    uv run python scripts/check_integrity.py
    uv run python scripts/check_integrity.py --deep 12
    uv run python scripts/check_integrity.py --deep-all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zarr

# Values the competition description asserts. Checked rather than assumed.
EXPECTED_SCALE_ZYX = (1.625, 0.40625, 0.40625)
EXPECTED_NDIM = 4
EXPECTED_NODE_PROPS = {"t", "z", "y", "x"}
EXPECTED_QUANTILE_KEYS = {0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 1.0}

# metrics.md: a GT division has exactly two outgoing edges; the scorer caps
# predicted out-degree at 2 and drops edges not spanning exactly one timepoint.
MAX_OUT_DEGREE = 2
MAX_IN_DEGREE = 1
REQUIRED_EDGE_DT = 1

# Physical displacement above which a frame-to-frame link is implausible for a
# cell and more likely an identity swap or a stage shift.
DISPLACEMENT_WARN_UM = 20.0

# A frame whose mean intensity falls below this fraction of the dataset median
# is treated as dropped or corrupted.
DEAD_FRAME_FRACTION = 0.25


def json_safe(obj):
    """Replace non-finite floats with None, recursively.

    ``json.dump`` writes bare ``NaN`` and ``Infinity`` tokens for non-finite
    floats. Python reads those back, but they are not valid JSON, so jq, schema
    validators and other languages' parsers reject the file. Measured
    intensities and ratios here can be undefined when a selection is empty.
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


class Report:
    """Collects findings and prints a severity-ordered summary."""

    def __init__(self) -> None:
        self.findings: list[dict] = []
        self.checks_run: Counter = Counter()

    def add(self, severity: str, check: str, dataset: str | None, message: str,
            detail: dict | None = None) -> None:
        self.findings.append({
            "severity": severity, "check": check,
            "dataset": dataset, "message": message, "detail": detail or {},
        })

    def ran(self, check: str) -> None:
        self.checks_run[check] += 1

    def counts(self) -> Counter:
        return Counter(f["severity"] for f in self.findings)

    @property
    def failed(self) -> bool:
        return self.counts()["FAIL"] > 0


# ------------------------------------------------------------------ metadata


def check_image_metadata(name: str, zarr_path: Path, rep: Report) -> dict:
    """Shape, dtype, voxel scale and stored intensity statistics."""
    out: dict = {"name": name}
    rep.ran("image_metadata")
    try:
        group = zarr.open_group(zarr_path, mode="r")
    except Exception as exc:
        rep.add("FAIL", "zarr_opens", name, f"cannot open zarr: {type(exc).__name__}: {exc}")
        return out

    levels = sorted(group.array_keys())
    if not levels:
        rep.add("FAIL", "zarr_has_level_0", name, "zarr group contains no arrays")
        return out
    if "0" not in levels:
        rep.add("FAIL", "zarr_has_level_0", name,
                f"no multiscale level '0'; found {levels}")
        return out

    arr = group["0"]
    out["shape"] = tuple(int(s) for s in arr.shape)
    out["dtype"] = str(arr.dtype)
    out["chunks"] = tuple(int(c) for c in arr.chunks)

    if arr.ndim != EXPECTED_NDIM:
        rep.add("FAIL", "image_is_4d", name,
                f"expected 4 dimensions (T,Z,Y,X), got {arr.ndim}: {out['shape']}")
        return out
    if any(s <= 0 for s in out["shape"]):
        rep.add("FAIL", "image_nonempty", name, f"degenerate shape {out['shape']}")

    attrs = dict(group.attrs)

    ms = attrs.get("multiscales")
    if ms is None:
        where = "under 'ome'" if isinstance(attrs.get("ome"), dict) else "nowhere"
        rep.add("FAIL", "multiscales_top_level", name,
                f"'multiscales' absent from top-level attrs (found {where}); "
                "io._parse_scale would silently fall back to its hardcoded scale")
    else:
        try:
            transforms = ms[0]["datasets"][0]["coordinateTransformations"]
            scale = next(t["scale"] for t in transforms if t.get("type") == "scale")
            out["scale"] = tuple(float(v) for v in scale[-3:])
            if not np.allclose(out["scale"], EXPECTED_SCALE_ZYX, rtol=1e-6):
                rep.add("FAIL", "scale_as_documented", name,
                        f"voxel scale {out['scale']} != documented {EXPECTED_SCALE_ZYX}; "
                        "every micrometre computation downstream would be wrong")
        except (KeyError, IndexError, TypeError, StopIteration) as exc:
            rep.add("FAIL", "scale_parseable", name,
                    f"cannot read scale transform: {type(exc).__name__}")

    stats = attrs.get("image_statistics", {})
    quant = stats.get("quantiles", {}) if isinstance(stats, dict) else {}
    if not quant:
        rep.add("WARN", "quantiles_present", name,
                "no image_statistics.quantiles; the loader would recompute them")
    else:
        try:
            pairs = sorted((float(k), float(v)) for k, v in quant.items())
        except (TypeError, ValueError):
            rep.add("FAIL", "quantiles_numeric", name, "non-numeric quantile entries")
            pairs = []
        keys = {k for k, _ in pairs}
        missing = EXPECTED_QUANTILE_KEYS - keys
        if missing:
            rep.add("WARN", "quantile_keys_complete", name,
                    f"missing quantile keys {sorted(missing)}")
        vals = [v for _, v in pairs]
        if vals != sorted(vals):
            # A non-monotone quantile table means the stored statistics are
            # corrupt, and normalization would produce a negative or inverted
            # range.
            rep.add("FAIL", "quantiles_monotone", name,
                    f"quantile values not non-decreasing: {vals}")
        if pairs:
            out["q_lo"], out["q_hi"] = vals[0], vals[-1]
            if vals[0] == vals[-1]:
                rep.add("FAIL", "quantiles_nondegenerate", name,
                        f"all quantiles equal ({vals[0]}); normalization divides by zero")
        hi = max(vals) if vals else 0
        if out.get("dtype") == "uint16" and hi >= 65535:
            rep.add("WARN", "not_saturated", name,
                    "maximum reaches the uint16 ceiling; the sensor likely clipped")
    return out


# --------------------------------------------------------------------- graph


def read_graph(geff_path: Path) -> dict:
    g = zarr.open_group(geff_path, mode="r")
    props = {}
    prop_names = sorted(g["nodes/props"].group_keys())
    for pname in prop_names:
        try:
            props[pname] = np.asarray(g[f"nodes/props/{pname}/values"][:])
        except Exception:
            pass
    edges = np.asarray(g["edges/ids"][:])
    return {
        "ids": np.asarray(g["nodes/ids"][:]),
        "props": props,
        "prop_names": prop_names,
        "edges": edges.reshape(-1, 2) if edges.size else np.empty((0, 2), dtype=np.int64),
        "attrs": dict(g.attrs),
    }


def check_graph(name: str, geff_path: Path, image_meta: dict, rep: Report) -> dict:
    """Node validity, edge topology, and metadata/data consistency."""
    out: dict = {"name": name}
    rep.ran("graph")
    try:
        gr = read_graph(geff_path)
    except Exception as exc:
        rep.add("FAIL", "geff_opens", name, f"cannot read geff: {type(exc).__name__}: {exc}")
        return out

    ids, props, edges = gr["ids"], gr["props"], gr["edges"]
    out["n_nodes"], out["n_edges"] = int(len(ids)), int(len(edges))

    if len(ids) == 0:
        rep.add("FAIL", "graph_nonempty", name, "geff contains no nodes")
        return out

    extra = set(gr["prop_names"]) - EXPECTED_NODE_PROPS
    missing = EXPECTED_NODE_PROPS - set(gr["prop_names"])
    if missing:
        rep.add("FAIL", "node_props_complete", name, f"missing node props {sorted(missing)}")
        return out
    if extra:
        rep.add("INFO", "node_props_expected", name,
                f"extra node props present: {sorted(extra)}")

    if len(np.unique(ids)) != len(ids):
        dup = len(ids) - len(np.unique(ids))
        rep.add("FAIL", "node_ids_unique", name, f"{dup} duplicate node id(s)")

    t = np.asarray(props["t"])
    zyx = {k: np.asarray(props[k]) for k in ("z", "y", "x")}

    for key, a in [("t", t), *zyx.items()]:
        if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all():
            rep.add("FAIL", "coords_finite", name,
                    f"{int((~np.isfinite(a)).sum())} non-finite value(s) in '{key}'")

    # Coordinates must index the volume they annotate. An out-of-bounds node
    # cannot be matched, and its presence usually means a convention mismatch
    # (axis order, or an off-by-one origin) rather than a stray annotation.
    shape = image_meta.get("shape")
    if shape and len(shape) == 4:
        T, Z, Y, X = shape
        for key, a, hi in (("t", t, T), ("z", zyx["z"], Z),
                           ("y", zyx["y"], Y), ("x", zyx["x"], X)):
            bad = int(((a < 0) | (a >= hi)).sum())
            if bad:
                rep.add("FAIL", "coords_in_bounds", name,
                        f"{bad} node(s) with '{key}' outside [0,{hi}); "
                        f"range is {a.min()}..{a.max()}",
                        {"axis": key, "limit": hi,
                         "min": float(a.min()), "max": float(a.max())})

    # Declared axis bounds should agree with the stored values. A mismatch means
    # the metadata was written from different data than the props.
    for ax in gr["attrs"].get("geff", {}).get("axes", []) or []:
        nm = ax.get("name")
        if nm not in ("t", "z", "y", "x"):
            continue
        a = t if nm == "t" else zyx[nm]
        for field, actual in (("min", float(a.min())), ("max", float(a.max()))):
            declared = ax.get(field)
            if declared is not None and not np.isclose(float(declared), actual):
                rep.add("WARN", "geff_axes_match_props", name,
                        f"axis '{nm}' declares {field}={declared} but props give {actual}")
        declared_scale = ax.get("scale")
        if nm in ("z", "y", "x") and declared_scale is not None:
            expected = EXPECTED_SCALE_ZYX[("z", "y", "x").index(nm)]
            if not np.isclose(float(declared_scale), expected):
                rep.add("WARN", "geff_scale_matches_zarr", name,
                        f"axis '{nm}' declares scale {declared_scale}, expected {expected}")

    est = (gr["attrs"].get("geff", {}).get("extra", {}) or {}).get(
        "estimated_number_of_nodes")
    if est is None:
        rep.add("FAIL", "t_true_present", name,
                "estimated_number_of_nodes absent; the adjusted edge Jaccard "
                "has no target node count for this sample")
    else:
        out["t_true"] = float(est)
        if out["t_true"] <= 0:
            rep.add("FAIL", "t_true_positive", name,
                    f"estimated_number_of_nodes = {est}")
        elif out["t_true"] < out["n_nodes"]:
            # T_true counts all cells, so it cannot be below the annotated subset.
            rep.add("FAIL", "t_true_exceeds_gt", name,
                    f"estimated_number_of_nodes ({est:.0f}) is below the annotated "
                    f"node count ({out['n_nodes']})")

    # --- edges ---
    id_set = set(int(v) for v in ids)
    idx = {int(v): i for i, v in enumerate(ids)}
    if len(edges):
        flat = edges.reshape(-1)
        unknown = [int(v) for v in np.unique(flat) if int(v) not in id_set]
        if unknown:
            rep.add("FAIL", "edges_reference_known_nodes", name,
                    f"{len(unknown)} edge endpoint(s) refer to unknown node ids",
                    {"examples": unknown[:5]})

        self_loops = int((edges[:, 0] == edges[:, 1]).sum())
        if self_loops:
            rep.add("FAIL", "no_self_edges", name, f"{self_loops} self-edge(s)")

        pairs = [tuple(int(v) for v in e) for e in edges]
        n_dup = len(pairs) - len(set(pairs))
        if n_dup:
            rep.add("WARN", "no_duplicate_edges", name,
                    f"{n_dup} duplicate source->target pair(s); the scorer "
                    "deduplicates these")

        valid = [(s, tg) for s, tg in pairs if s in idx and tg in idx]
        if valid:
            dt = np.array([t[idx[tg]] - t[idx[s]] for s, tg in valid], dtype=float)
            n_bad_dt = int((dt != REQUIRED_EDGE_DT).sum())
            if n_bad_dt:
                # The scorer discards these silently rather than penalising them,
                # so they are dead weight in a prediction and a red flag in GT.
                rep.add("FAIL", "edges_span_one_timepoint", name,
                        f"{n_bad_dt} edge(s) with t_target - t_source != 1 "
                        f"(range {dt.min():.0f}..{dt.max():.0f}); the metric "
                        "discards these")

            out_deg, in_deg = Counter(), Counter()
            for s, tg in valid:
                out_deg[s] += 1
                in_deg[tg] += 1
            over_out = [n for n, d in out_deg.items() if d > MAX_OUT_DEGREE]
            over_in = [n for n, d in in_deg.items() if d > MAX_IN_DEGREE]
            if over_out:
                rep.add("FAIL", "out_degree_at_most_2", name,
                        f"{len(over_out)} node(s) with more than {MAX_OUT_DEGREE} "
                        "children; the scorer keeps only the two lowest edge ids",
                        {"examples": over_out[:5]})
            if over_in:
                rep.add("FAIL", "in_degree_at_most_1", name,
                        f"{len(over_in)} node(s) with more than one parent "
                        "(a merge, which nuclei lineages cannot produce)",
                        {"examples": over_in[:5]})

            divisions = [n for n, d in out_deg.items() if d == 2]
            out["n_divisions"] = len(divisions)
            # metrics.md: daughters of a division sit at the parent's timepoint + 1.
            for n in divisions:
                kids = [tg for s, tg in valid if s == n]
                kt = {int(t[idx[k]]) for k in kids}
                if len(kt) != 1 or kt.pop() != int(t[idx[n]]) + 1:
                    rep.add("WARN", "division_daughters_same_frame", name,
                            f"division node {n} has daughters at differing timepoints")

            scale = np.asarray(EXPECTED_SCALE_ZYX)
            disp = np.array([
                np.linalg.norm((np.array([zyx["z"][idx[tg]], zyx["y"][idx[tg]],
                                          zyx["x"][idx[tg]]], dtype=float)
                                - np.array([zyx["z"][idx[s]], zyx["y"][idx[s]],
                                            zyx["x"][idx[s]]], dtype=float)) * scale)
                for s, tg in valid
            ])
            out["disp_um_max"] = float(disp.max()) if disp.size else 0.0
            n_far = int((disp > DISPLACEMENT_WARN_UM).sum())
            if n_far:
                rep.add("WARN", "displacement_plausible", name,
                        f"{n_far} edge(s) exceed {DISPLACEMENT_WARN_UM:g} um "
                        f"(max {disp.max():.1f} um); likely identity swaps or a "
                        "stage shift rather than cell motion",
                        {"n_far": n_far, "max_um": float(disp.max())})

        isolated = [n for n in id_set
                    if out_deg.get(n, 0) == 0 and in_deg.get(n, 0) == 0] if valid else []
        if isolated:
            rep.add("WARN", "no_isolated_nodes", name,
                    f"{len(isolated)} node(s) with no edges at all")
    else:
        rep.add("WARN", "graph_has_edges", name, "geff contains nodes but no edges")

    return out


# ---------------------------------------------------------------- deep checks


def check_volume(name: str, zarr_path: Path, geff_path: Path | None,
                 image_meta: dict, rep: Report) -> dict:
    """Voxel-level checks: dropped frames, and intensity at annotated positions."""
    out: dict = {"name": name}
    rep.ran("volume")
    try:
        arr = zarr.open_group(zarr_path, mode="r")["0"]
    except Exception as exc:
        rep.add("FAIL", "zarr_opens", name, f"cannot open for deep check: {exc}")
        return out

    T = arr.shape[0]
    means = np.empty(T, dtype=np.float64)
    for t in range(T):
        # Strided read: the chunk is a whole volume, so this costs one chunk
        # decompression per frame regardless of the stride.
        means[t] = float(np.asarray(arr[t, ::4, ::8, ::8]).mean())
    out["frame_mean_min"] = float(means.min())
    out["frame_mean_median"] = float(np.median(means))

    med = np.median(means)
    if med > 0:
        dead = np.flatnonzero(means < DEAD_FRAME_FRACTION * med)
        if dead.size:
            rep.add("WARN", "no_dropped_frames", name,
                    f"{dead.size} frame(s) below {DEAD_FRAME_FRACTION:.0%} of the "
                    f"median frame intensity; likely dropped or corrupted",
                    {"frames": dead.tolist()[:20],
                     "means": [round(float(means[i]), 1) for i in dead[:20]]})
        out["dead_frames"] = dead.tolist()

    const = [t for t in range(T) if float(np.asarray(arr[t, ::8, ::16, ::16]).std()) == 0.0]
    if const:
        rep.add("FAIL", "frames_not_constant", name,
                f"{len(const)} frame(s) have zero variance", {"frames": const[:20]})

    # Intensity at annotated positions versus matched random positions. If GT
    # coordinates were transposed or offset, the annotated positions would sit
    # on background and this contrast would vanish. It is the cheapest available
    # test of the coordinate convention.
    if geff_path is not None:
        try:
            gr = read_graph(geff_path)
            t_arr = np.asarray(gr["props"]["t"])
            zc = np.asarray(gr["props"]["z"]).astype(int)
            yc = np.asarray(gr["props"]["y"]).astype(int)
            xc = np.asarray(gr["props"]["x"]).astype(int)
            rng = np.random.default_rng(0)
            picks = np.unique(t_arr)[:: max(1, len(np.unique(t_arr)) // 5)][:5]
            gt_vals, rand_vals = [], []
            for tv in picks:
                m = t_arr == tv
                if not m.any():
                    continue
                vol = np.asarray(arr[int(tv)])
                gt_vals.append(vol[zc[m], yc[m], xc[m]].astype(float))
                n = int(m.sum())
                rand_vals.append(vol[
                    rng.integers(0, vol.shape[0], n),
                    rng.integers(0, vol.shape[1], n),
                    rng.integers(0, vol.shape[2], n),
                ].astype(float))
            if gt_vals:
                gt_v = np.concatenate(gt_vals)
                rd_v = np.concatenate(rand_vals)
                out["gt_median_intensity"] = float(np.median(gt_v))
                out["random_median_intensity"] = float(np.median(rd_v))
                ratio = float(np.median(gt_v) / max(np.median(rd_v), 1e-6))
                out["gt_over_random"] = ratio
                if ratio < 1.0:
                    rep.add("FAIL", "gt_positions_are_bright", name,
                            f"annotated positions are dimmer than random ones "
                            f"(ratio {ratio:.2f}); coordinate convention is "
                            "probably misread")
                elif ratio < 1.5:
                    rep.add("WARN", "gt_positions_are_bright", name,
                            f"annotated positions only {ratio:.2f}x brighter than "
                            "random; weaker contrast than expected")
        except Exception as exc:
            rep.add("WARN", "gt_intensity_check", name,
                    f"could not sample GT intensities: {type(exc).__name__}")
    return out


# -------------------------------------------------------------------- driver


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/raw", type=Path)
    ap.add_argument("--out", default="reports/integrity", type=Path)
    ap.add_argument("--deep", type=int, default=0, metavar="N",
                    help="run voxel-level checks on N datasets (~4s each)")
    ap.add_argument("--deep-all", action="store_true",
                    help="run voxel-level checks on every dataset")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on WARN as well as FAIL")
    args = ap.parse_args(argv)

    if not args.data_root.is_dir():
        print(f"data root not found: {args.data_root.resolve()}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    rep = Report()
    per_dataset: list[dict] = []

    records = []
    for split in ("train", "test"):
        sdir = args.data_root / split
        if not sdir.is_dir():
            rep.add("WARN", "split_present", None, f"no '{split}' directory")
            continue
        for zp in sorted(sdir.glob("*.zarr")):
            gp = sdir / f"{zp.stem}.geff"
            records.append({"name": zp.stem, "split": split, "zarr": zp,
                            "geff": gp if gp.exists() else None})

    if not records:
        print(f"no .zarr datasets under {args.data_root.resolve()}", file=sys.stderr)
        return 2
    print(f"checking {len(records)} datasets\n")

    shapes: Counter = Counter()
    for i, r in enumerate(records, 1):
        if i % 25 == 0 or i == len(records):
            print(f"  {i}/{len(records)}")
        meta = check_image_metadata(r["name"], r["zarr"], rep)
        meta["split"] = r["split"]
        if "shape" in meta:
            shapes[str(meta["shape"])] += 1

        if r["split"] == "train":
            if r["geff"] is None:
                rep.add("FAIL", "train_has_labels", r["name"],
                        "training dataset has no paired .geff")
            else:
                meta.update(check_graph(r["name"], r["geff"], meta, rep))
        elif r["geff"] is not None:
            rep.add("INFO", "test_unlabelled", r["name"],
                    "test dataset unexpectedly has a .geff")
        per_dataset.append(meta)

    if len(shapes) > 1:
        rep.add("WARN", "shapes_uniform", None,
                f"datasets do not share one shape: {dict(shapes)}")

    deep_targets = []
    if args.deep_all:
        deep_targets = records
    elif args.deep > 0:
        by_split = defaultdict(list)
        for r in records:
            by_split[r["split"]].append(r)
        step = max(1, len(by_split["train"]) // max(1, args.deep))
        deep_targets = by_split["train"][::step][: args.deep]

    if deep_targets:
        print(f"\ndeep checks on {len(deep_targets)} dataset(s)")
        lookup = {m["name"]: m for m in per_dataset}
        for i, r in enumerate(deep_targets, 1):
            if i % 5 == 0 or i == len(deep_targets):
                print(f"  {i}/{len(deep_targets)}")
            d = check_volume(r["name"], r["zarr"], r["geff"],
                             lookup.get(r["name"], {}), rep)
            lookup.get(r["name"], {}).update(d)

    counts = rep.counts()
    payload = {
        "datasets_checked": len(records),
        "checks_run": dict(rep.checks_run),
        "severity_counts": {k: counts[k] for k in ("FAIL", "WARN", "INFO")},
        "findings": rep.findings,
        "per_dataset": per_dataset,
    }
    with open(args.out / "integrity.json", "w") as f:
        json.dump(json_safe(payload), f, indent=2, default=str)

    print(f"\nwrote {args.out / 'integrity.json'}")
    print(f"\n=== {counts['FAIL']} FAIL, {counts['WARN']} WARN, "
          f"{counts['INFO']} INFO ===")

    by_check: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in rep.findings:
        by_check[(f["severity"], f["check"])].append(f)
    for sev in ("FAIL", "WARN", "INFO"):
        rows = [(c, v) for (s, c), v in by_check.items() if s == sev]
        if not rows:
            continue
        print(f"\n[{sev}]")
        for check, items in sorted(rows, key=lambda kv: -len(kv[1])):
            names = [i["dataset"] for i in items if i["dataset"]]
            shown = ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")
            print(f"  {check}: {len(items)} dataset(s)  {shown}")
            print(f"    e.g. {items[0]['message']}")

    if not rep.findings:
        print("\nno findings: every checked assumption holds")

    if rep.failed:
        return 1
    if args.strict and counts["WARN"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
