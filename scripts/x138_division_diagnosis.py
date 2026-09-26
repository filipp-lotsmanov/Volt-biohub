#!/usr/bin/env python
"""Where do x138's missed divisions die? One bucket per GT division, from x138's own run.

Every division decomposition in this repo was measured on OUR pipeline. x138's
division pass differs in the one place that matters: a DeepCenter veto after the
geometric gates, which rejected 482 of 557 geometric candidates on the visible
movies. So before building anything on the gates, measure x138 itself.

Inputs come from one run of the diagnosis build (`build_x138_variant.py
--division-diagnosis DIR`) through the post-processing-only harness:

* the run's ``submission.csv`` -- labelled TP/FN per GT division by the official
  scorer, the same call `division_audit.py` makes;
* ``DIR/{movie}_graph.npz`` -- the graph the division pass received;
* ``DIR/{movie}_cands.csv`` -- every candidate pair with the value of EVERY gate;
* ``DIR/{movie}_added.csv`` -- the forks the pass added;
* x138's own ``edge_cache/{movie}.npz`` -- every detector peak above 0.3.

Each missed GT division gets exactly one bucket, decided in this order:

    P   the divider was never detected
    E0  the divider has no child, so the pass never considers it
    E2  the divider already has two children (fork present, not scored)
    D2  the divider is linked to a cell that is not one of its daughters
    D   the second daughter is linked to another parent
    A   the second daughter is there and unclaimed, but was not admitted:
        the detail names every gate it fails, or "passed every gate" (cap or collision)
    B   no node for the second daughter, but a free detector peak exists there:
        below the detection threshold, or above it and dropped before the pass
    C   no node and no free peak: either nothing, or only a peak that belongs to an
        existing node (usually the sister's -- the detector saw one blob)

A peak is "free" by the rule x138's own gap filler uses: more than 2.0 um from every
node of its frame. It must also be nearer the missing daughter than any node, so
the sister's peak, displaced from her node by the coordinate head, is not mistaken
for the daughter's.

It also labels every logged candidate pair true/false and prints, per gate, how
many true and false pairs fail ONLY that gate -- i.e. what dropping that gate
alone would admit, before caps. That is the cost side of every gate change.

Two caveats, both conservative in the same direction as `division_audit.py`.
Matching is nearest node within 7 um, not the scorer's optimal assignment, so a
node reported "detected" may still lose the assignment; one reported missing is
missing. And the analysis looks at the exact GT split frame, while the scorer
also accepts a fork one frame early or late -- the TP labels come from the
scorer, so this only affects which bucket an FN lands in, never the counts of TP.

Usage
-----
    uv run python kaggle/forks/build_x138_variant.py /tmp/x138/biohub-x138.ipynb \
        /tmp/k_divdiag --slug volt-x138-divdiag --division-diagnosis /kaggle/working/volt_divdiag
    # run /tmp/k_divdiag through the post-processing-only harness on fold 0 and fold 1
    uv run python scripts/x138_division_diagnosis.py --csv <run>/submission.csv \
        --diag-dir <run>/volt_divdiag --cache-dir <run>/edge_cache \
        --out reports/x138_division_diagnosis.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle"))

# x138's VOXEL_SCALE_UM, and this repo's ORIGINAL_SCALE: both work in original voxels.
SCALE = np.asarray((1.625, 0.40625, 0.40625))
TOL_UM = 7.0
FREE_UM = 2.0   # x138's GAPFILL_EXCLUDE_UM: a peak this close to a node is that node's
GATES = ("exist", "par", "sis", "nn", "div", "dc", "sym")   # x138's order of evaluation
GATE_NAMES = {
    "exist": "existing-child distance", "par": "parent distance", "sis": "sister distance",
    "nn": "not the nearest orphan", "div": "forward divergence", "dc": "DeepCenter veto",
    "sym": "symmetry",
}
C_NOTHING = "no node and no peak"
C_OWNED = "no separate peak; the only peak nearby belongs to an existing node"
BUCKETS = {
    "P": "divider not detected",
    "E0": "divider has no child",
    "E2": "divider already has two children",
    "D2": "divider linked to a non-daughter",
    "D": "second daughter linked to another parent",
    "A": "second daughter found, unclaimed, not admitted",
    "B": "no node; a free detector peak exists",
    "C": "no node and no free peak",
}
FLOAT_COLS = ("d_exist", "d_par", "d_sis", "div_margin", "symmetry", "dc_score", "rank_score")
INT_COLS = ("t", "source_id", "child_id", "cand_id", "nn_rank", "child_succ", "cand_succ",
            *(f"pass_{g}" for g in GATES))


class FramePoints:
    """Nearest point within a tolerance, per frame, in micrometres."""

    def __init__(self, t: np.ndarray, zyx_vox: np.ndarray, payload: np.ndarray):
        self.payload = np.asarray(payload)
        um = np.asarray(zyx_vox, dtype=np.float64).reshape(-1, 3) * SCALE
        t = np.asarray(t, dtype=np.int64)
        self.t_of = dict(zip(self.payload.tolist(), t.tolist(), strict=True))
        self.frames: dict[int, tuple[cKDTree, np.ndarray]] = {}
        for tt in np.unique(t):
            rows = np.nonzero(t == tt)[0]
            self.frames[int(tt)] = (cKDTree(um[rows]), rows)

    def nearest(self, t: int, um: np.ndarray, tol: float = TOL_UM, exclude: set = frozenset()):
        hit = self.frames.get(int(t))
        if hit is None:
            return None
        tree, rows = hit
        k = min(len(rows), len(exclude) + 1)
        d, idx = tree.query(um, k=k)
        for dd, ii in zip(np.atleast_1d(d), np.atleast_1d(idx), strict=True):
            if dd > tol:
                return None
            value = self.payload[rows[int(ii)]].item()
            if value not in exclude:
                return value
        return None

    def nearest_dist(self, t: int, um: np.ndarray) -> np.ndarray:
        """Distance from each row of *um* to the nearest point of frame *t* (inf if none)."""
        um = np.asarray(um, dtype=np.float64).reshape(-1, 3)
        hit = self.frames.get(int(t))
        if hit is None:
            return np.full(len(um), np.inf)
        d, _ = hit[0].query(um, k=1)
        return np.asarray(d, dtype=np.float64).reshape(-1)


@dataclass
class PreGraph:
    """The graph x138's division pass received."""

    points: FramePoints
    um_of: dict[int, np.ndarray]
    children: dict[int, list[int]]
    parent: dict[int, int]

    @classmethod
    def from_arrays(cls, node_id, t, zyx, edge_src, edge_tgt) -> PreGraph:
        node_id = np.asarray(node_id, dtype=np.int64)
        zyx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
        children: dict[int, list[int]] = defaultdict(list)
        parent: dict[int, int] = {}
        for a, b in zip(np.asarray(edge_src).tolist(), np.asarray(edge_tgt).tolist(), strict=True):
            children[int(a)].append(int(b))
            parent.setdefault(int(b), int(a))
        um_of = {int(i): p * SCALE for i, p in zip(node_id.tolist(), zyx, strict=True)}
        return cls(FramePoints(t, zyx, node_id), um_of, dict(children), parent)

    @classmethod
    def load(cls, path: Path) -> PreGraph:
        g = np.load(path)
        return cls.from_arrays(g["node_id"], g["t"], g["zyx"], g["edge_src"], g["edge_tgt"])


class LowPool:
    """x138's dump of every detector peak above its low-detection threshold (0.3).

    The dump holds peaks ABOVE the detection threshold too, so every node's own peak
    is in it; `free_peak` is what separates a missed cell from its neighbour.
    """

    def __init__(self, coords: np.ndarray, score: np.ndarray):
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 4)
        t = coords[:, 0].astype(np.int64)
        um = coords[:, 1:] * SCALE
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        self.frames: dict[int, tuple[cKDTree, np.ndarray, np.ndarray]] = {}
        for tt in np.unique(t):
            rows = np.nonzero(t == tt)[0]
            self.frames[int(tt)] = (cKDTree(um[rows]), um[rows], score[rows])

    def free_peak(self, t: int, um: np.ndarray, nodes: FramePoints,
                  tol: float = TOL_UM) -> tuple[float | None, bool]:
        """(score of the nearest free peak within *tol*, whether an owned peak is there).

        Free: more than FREE_UM from every node of frame *t*, and nearer *um* than to
        any of them.
        """
        hit = self.frames.get(int(t))
        if hit is None:
            return None, False
        tree, peak_um, peak_score = hit
        rows = tree.query_ball_point(np.asarray(um, dtype=np.float64), tol)
        if not rows:
            return None, False
        rows = np.asarray(sorted(rows), dtype=np.int64)
        to_target = np.linalg.norm(peak_um[rows] - um, axis=1)
        to_node = nodes.nearest_dist(t, peak_um[rows])
        free = (to_node > FREE_UM) & (to_target < to_node)
        if not free.any():
            return None, True
        best = rows[free][np.argmin(to_target[free])]
        return float(peak_score[best]), False

    @classmethod
    def load(cls, cache_dir: Path | None, name: str) -> LowPool | None:
        if cache_dir is None:
            return None
        path = cache_dir / f"{name}.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            if "low_coords" not in z.files:
                return None
            return cls(z["low_coords"], z["low_score"])


def load_candidates(path: Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {k: int(raw[k]) for k in INT_COLS}
            row.update({k: float(raw[k]) for k in FLOAT_COLS})
            out[(row["source_id"], row["cand_id"])] = row
    return out


def load_added(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {(int(r["source_id"]), int(r["target_id"])) for r in csv.DictReader(fh)}


def classify_division(
    t: int,
    parent_um: np.ndarray,
    daughters: list[tuple[int, np.ndarray]],
    pre: PreGraph,
    low: LowPool | None,
    cands: dict[tuple[int, int], dict],
    added: set[tuple[int, int]],
    det_threshold: float = 0.965,
    tol: float = TOL_UM,
) -> dict:
    """Bucket one GT division by where its second daughter was lost.

    *daughters* are the two GT daughters as ``(t, um)``. Returns ``bucket``, ``detail``,
    ``pair`` (divider node, second-daughter node) when
    both exist in the pass's input, ``failing`` gates, ``weak_score`` for bucket B, and
    the logged gate values of the pair.
    """
    out = {"bucket": None, "detail": "", "pair": None, "failing": [], "weak_score": None,
           "values": {}}
    div_n = pre.points.nearest(t, parent_um, tol)
    if div_n is None:
        return {**out, "bucket": "P", "detail": BUCKETS["P"]}
    kids = pre.children.get(div_n, [])
    if not kids:
        return {**out, "bucket": "E0", "detail": BUCKETS["E0"]}
    if len(kids) >= 2:
        return {**out, "bucket": "E2", "detail": BUCKETS["E2"]}
    c1 = kids[0]
    c1_t = int(pre.points.t_of[c1])
    gaps = [float(np.linalg.norm(pre.um_of[c1] - d_um)) if d_t == c1_t else np.inf
            for d_t, d_um in daughters]
    first = int(np.argmin(gaps))
    if gaps[first] > tol:
        return {**out, "bucket": "D2", "detail": BUCKETS["D2"]}
    second_t, second_um = daughters[1 - first]
    n2 = pre.points.nearest(second_t, second_um, tol, exclude={c1})
    if n2 is None:
        if low is None:
            return {**out, "bucket": "C", "detail": "no node (no low-detection cache given)"}
        score, owned = low.free_peak(second_t, second_um, pre.points, tol)
        if score is None:
            return {**out, "bucket": "C", "detail": C_OWNED if owned else C_NOTHING}
        where = ("below the detection threshold" if score < det_threshold
                 else "above the threshold, dropped before the pass (ILP or relink)")
        return {**out, "bucket": "B", "detail": f"peak {score:.3f}, {where}", "weak_score": score}
    if n2 in pre.parent:
        return {**out, "bucket": "D", "detail": BUCKETS["D"]}
    pair = (div_n, n2)
    row = cands.get(pair)
    if row is None:
        return {**out, "bucket": "A", "detail": "outside the logged envelope", "pair": pair}
    failing = [g for g in GATES if not row[f"pass_{g}"]]
    values = {k: row[k] for k in FLOAT_COLS} | {"nn_rank": row["nn_rank"]}
    if failing:
        detail = "refused by " + " + ".join(GATE_NAMES[g] for g in failing)
    elif pair in added:
        detail = "fork added, but not scored as this division"
    else:
        detail = "passed every gate; lost to a cap or a collision"
    return {**out, "bucket": "A", "detail": detail, "pair": pair, "failing": failing,
            "values": values}


def gate_ablation(cand_rows: list[dict], true_pairs: set[tuple[str, int, int]]
                  ) -> dict[str, Counter]:
    """Per gate, true and false pairs failing ONLY that gate; plus pairs passing all.

    Rows carry ``dataset``; *true_pairs* holds ``(dataset, source_id, cand_id)``.
    """
    out: dict[str, Counter] = {g: Counter() for g in (*GATES, "all_pass")}
    for row in cand_rows:
        key = (row["dataset"], row["source_id"], row["cand_id"])
        label = "true" if key in true_pairs else "false"
        failing = [g for g in GATES if not row[f"pass_{g}"]]
        if not failing:
            out["all_pass"][label] += 1
        elif len(failing) == 1:
            out[failing[0]][label] += 1
    return out


def embryo_of(name: str) -> str:
    return name.split("_", 1)[0]


def diagnose_dataset(name, pred_nodes, pred_edges, gt, pre, low, cands, added,
                     det_threshold: float = 0.965):
    """Official TP/FN per GT division, then a bucket for every FN.

    Returns the rows, the true candidate pairs as ``(name, divider, daughter)`` -- node
    ids repeat across movies -- and the number of false-positive forks.
    """
    from csv_to_geffs import build_graph_from_rows
    from division_audit import _nodes_um
    from volt_biohub.division_metrics import score_divisions
    from volt_biohub.scoring import ORIGINAL_SCALE

    # Same order as division_audit.py: the graph the scorer grades is not reused.
    scored = score_divisions(build_graph_from_rows(pred_nodes, pred_edges), gt,
                             ORIGINAL_SCALE, TOL_UM)
    gt_t, gt_um, gt_ids = _nodes_um(gt)
    pos = {i: (int(gt_t[k]), gt_um[k]) for k, i in enumerate(gt_ids)}

    rows, true_pairs = [], set()
    for div_node, ok in scored.scores.items():
        kids = [k for k in gt.successors(div_node) if k in pos]
        if div_node not in pos or len(kids) != 2:
            continue
        t, p_um = pos[div_node]
        res = classify_division(t, p_um, [pos[k] for k in kids], pre, low, cands, added,
                                det_threshold)
        if res["pair"] is not None:
            true_pairs.add((name, *res["pair"]))
        rows.append({
            "dataset": name, "embryo": embryo_of(name), "gt_div_node": int(div_node), "t": t,
            "scored_tp": bool(ok),
            "bucket": "TP" if ok else res["bucket"],
            "detail": "" if ok else res["detail"],
            "first_gate": res["failing"][0] if (res["failing"] and not ok) else "",
            "n_failing": len(res["failing"]) if not ok else 0,
            "weak_score": res["weak_score"],
            **{k: res["values"].get(k) for k in (*FLOAT_COLS, "nn_rank")},
        })
    return rows, true_pairs, len(scored.fp_forks)


def summarise(rows: list[dict], cand_rows: list[dict], true_pairs: set, n_fp: dict[str, int],
              det_threshold: float) -> str:
    lines = []
    groups = [*sorted({r["embryo"] for r in rows}), "all"]
    for g in groups:
        sel = [r for r in rows if g in ("all", r["embryo"])]
        if not sel:
            continue
        tp = sum(r["scored_tp"] for r in sel)
        fn = [r for r in sel if not r["scored_tp"]]
        fp = sum(v for k, v in n_fp.items() if g in ("all", k))
        lines += [f"\n=== {g}: {len(sel)} GT divisions, TP {tp} ({tp / len(sel):.1%}), "
                  f"FN {len(fn)}, FP forks {fp}, divJ {tp / max(tp + len(fn) + fp, 1):.4f} ==="]
        counts = Counter(r["bucket"] for r in fn)
        for b in BUCKETS:
            if counts[b]:
                lines.append(f"  {b:3s} {counts[b]:4d}  {counts[b] / len(fn):6.1%}  {BUCKETS[b]}")
        a_rows = [r for r in fn if r["bucket"] == "A"]
        if a_rows:
            lines.append("  A, first gate that refuses (x138's order):")
            for gate, n in Counter(r["first_gate"] or "(none)" for r in a_rows).most_common():
                lines.append(f"      {GATE_NAMES.get(gate, gate):28s} {n}")
            only = Counter(r["first_gate"] for r in a_rows if r["n_failing"] == 1)
            if only:
                lines.append("  A, refused by ONE gate only (move that gate and it passes):")
                for gate, n in only.most_common():
                    lines.append(f"      {GATE_NAMES[gate]:28s} {n}")
            rest = Counter(r["detail"] for r in a_rows if not r["first_gate"])
            for detail, n in rest.items():
                lines.append(f"  A, {detail}: {n}")
        c_rows = Counter(r["detail"] for r in fn if r["bucket"] == "C")
        for detail, n in c_rows.most_common():
            lines.append(f"  C, {detail}: {n}")
        b_rows = [r for r in fn if r["bucket"] == "B"]
        if b_rows:
            s = np.asarray([r["weak_score"] for r in b_rows])
            lines.append(f"  B, peak score p25/p50/p75 {np.percentile(s, 25):.2f} / "
                         f"{np.median(s):.2f} / {np.percentile(s, 75):.2f}; "
                         f"below {det_threshold}: {int((s < det_threshold).sum())} of {len(s)}")
        pairs = [c for c in cand_rows if g in ("all", c["embryo"])]
        if pairs:
            ab = gate_ablation(pairs, true_pairs)
            lines.append("  candidate pairs failing ONLY this gate (what dropping it admits, "
                         "before caps; false pairs are uncharged counts):")
            for gate in GATES:
                c = ab[gate]
                if c["true"] or c["false"]:
                    lines.append(f"      {GATE_NAMES[gate]:28s} true {c['true']:4d}   "
                                 f"false {c['false']:6d}")
            lines.append(f"      {'pass every gate':28s} true {ab['all_pass']['true']:4d}   "
                         f"false {ab['all_pass']['false']:6d}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import polars as pl

    from dataspec import DATASET_PATH
    from volt_biohub.scoring import load_gt

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="the diagnosis run's submission.csv")
    p.add_argument("--diag-dir", type=Path, required=True, help="the --division-diagnosis DIR")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="x138's edge_cache; without it buckets B and C cannot be told apart")
    p.add_argument("--data-dir", type=Path, default=None, help="GT .geff directory")
    p.add_argument("--det-threshold", type=float, default=0.965)
    p.add_argument("--out", type=Path, default=Path("reports/x138_division_diagnosis.csv"))
    a = p.parse_args(argv)
    data_dir = a.data_dir or DATASET_PATH

    df = pl.read_csv(a.csv, columns=["dataset", "row_type", "node_id", "t", "z", "y", "x",
                                     "source_id", "target_id"])
    rows, cand_rows, true_pairs, n_fp = [], [], set(), {}
    for (name,), group in sorted(df.group_by("dataset"), key=lambda kv: kv[0]):
        graph_path = a.diag_dir / f"{name}_graph.npz"
        if not (data_dir / f"{name}.geff").exists() or not graph_path.exists():
            print(f"  skip {name}: no GT or no diagnosis dump", flush=True)
            continue
        cands = load_candidates(a.diag_dir / f"{name}_cands.csv")
        r, tp_pairs, fp = diagnose_dataset(
            name,
            group.filter(pl.col("row_type") == "node"),
            group.filter(pl.col("row_type") == "edge"),
            load_gt(data_dir, name), PreGraph.load(graph_path), LowPool.load(a.cache_dir, name),
            cands, load_added(a.diag_dir / f"{name}_added.csv"), a.det_threshold)
        rows += r
        n_fp[embryo_of(name)] = n_fp.get(embryo_of(name), 0) + fp
        true_pairs |= tp_pairs
        cand_rows += [{**c, "dataset": name, "embryo": embryo_of(name)} for c in cands.values()]
        print(f"  {name}: {sum(x['scored_tp'] for x in r)}/{len(r)} TP, {fp} FP forks, "
              f"{len(cands)} candidate pairs", flush=True)
    if not rows:
        print("no GT divisions found in these datasets")
        return 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(a.out)
    # The labelled pool: the training and evaluation table for any new admission rule.
    cand_out = a.out.with_name(a.out.stem + "_candidates.csv")
    pl.DataFrame([{**c, "is_true": (c["dataset"], c["source_id"], c["cand_id"]) in true_pairs}
                  for c in cand_rows]).write_csv(cand_out)
    print(summarise(rows, cand_rows, true_pairs, n_fp, a.det_threshold))
    print(f"\nwrote {a.out} and {cand_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
