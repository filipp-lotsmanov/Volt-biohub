"""x138 division diagnosis: one bucket per missed GT division.

The buckets decide what gets built next (gates, detection, linking), so each one is
pinned on a hand-made geometry where the right answer is known, and the whole script
is run once against the official division scorer.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tracksdata as td

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle" / "forks"))

import build_x138_variant as builder

import x138_division_diagnosis as dx

SCALE = np.asarray((1.625, 0.40625, 0.40625))


def vox(z_um, y_um, x_um):
    """Micrometres -> original voxels, the unit every dump and CSV uses."""
    return tuple(np.asarray((z_um, y_um, x_um)) / SCALE)


def um(z_um, y_um, x_um):
    return np.asarray((z_um, y_um, x_um), dtype=np.float64)


# One division on the y axis: divider at y=50 (t=5), daughters at y=53 and y=47 (t=6).
PARENT = um(10, 50, 50)
DAUGHTERS = [(6, um(10, 53, 50)), (6, um(10, 47, 50))]
DIV, C1, N2, OTHER = 1, 2, 3, 4


def pregraph(nodes: dict[int, tuple[int, tuple]], edges: list[tuple[int, int]]) -> dx.PreGraph:
    ids = list(nodes)
    return dx.PreGraph.from_arrays(
        ids, [nodes[i][0] for i in ids], [vox(*nodes[i][1]) for i in ids],
        [a for a, _ in edges], [b for _, b in edges])


def cand_row(source=DIV, cand=N2, **fails) -> dict:
    row = {"t": 5, "source_id": source, "child_id": C1, "cand_id": cand, "nn_rank": 0,
           "child_succ": 1, "cand_succ": 1, "d_exist": 3.0, "d_par": 3.0, "d_sis": 6.0,
           "div_margin": 3.0, "symmetry": 0.0, "dc_score": 0.9, "rank_score": 3.9}
    row |= {f"pass_{g}": 0 if fails.get(g) else 1 for g in dx.GATES}
    return row


def base_nodes():
    return {DIV: (5, (10, 50, 50)), C1: (6, (10, 53, 50)), N2: (6, (10, 47.5, 50))}


def classify(pre, cands=None, added=frozenset(), low=None, daughters=DAUGHTERS):
    cands = {} if cands is None else cands
    return dx.classify_division(5, PARENT, daughters, pre, low, cands, set(added))


def low_pool(peaks: list[tuple[int, tuple, float]]) -> dx.LowPool:
    coords = np.asarray([(t, *vox(*p)) for t, p, _ in peaks], dtype=np.float64)
    return dx.LowPool(coords, np.asarray([s for _, _, s in peaks]))


class TestBuckets:
    def test_divider_not_detected(self):
        pre = pregraph({DIV: (5, (10, 80, 50)), C1: (6, (10, 83, 50))}, [(DIV, C1)])
        assert classify(pre)["bucket"] == "P"

    def test_divider_without_child(self):
        pre = pregraph(base_nodes(), [])
        assert classify(pre)["bucket"] == "E0"

    def test_divider_with_two_children(self):
        pre = pregraph(base_nodes(), [(DIV, C1), (DIV, N2)])
        assert classify(pre)["bucket"] == "E2"

    def test_divider_linked_to_a_non_daughter(self):
        nodes = base_nodes() | {C1: (6, (10, 70, 50))}
        assert classify(pregraph(nodes, [(DIV, C1)]))["bucket"] == "D2"

    def test_second_daughter_claimed_by_another_parent(self):
        nodes = base_nodes() | {OTHER: (5, (10, 40, 50))}
        res = classify(pregraph(nodes, [(DIV, C1), (OTHER, N2)]))
        assert res["bucket"] == "D"

    def test_refused_names_every_failing_gate_in_x138_order(self):
        pre = pregraph(base_nodes(), [(DIV, C1)])
        res = classify(pre, {(DIV, N2): cand_row(sym=True, dc=True)})
        assert res["bucket"] == "A"
        assert res["failing"] == ["dc", "sym"]
        assert res["detail"] == "refused by DeepCenter veto + symmetry"
        assert res["pair"] == (DIV, N2)
        assert res["values"]["d_par"] == 3.0

    def test_passed_every_gate_but_not_added(self):
        pre = pregraph(base_nodes(), [(DIV, C1)])
        res = classify(pre, {(DIV, N2): cand_row()})
        assert (res["bucket"], res["failing"]) == ("A", [])
        assert "cap or a collision" in res["detail"]

    def test_passed_and_added_but_not_scored(self):
        pre = pregraph(base_nodes(), [(DIV, C1)])
        res = classify(pre, {(DIV, N2): cand_row()}, added={(DIV, N2)})
        assert res["detail"] == "fork added, but not scored as this division"

    def test_outside_the_logged_envelope(self):
        pre = pregraph(base_nodes(), [(DIV, C1)])
        res = classify(pre, {})
        assert (res["bucket"], res["detail"]) == ("A", "outside the logged envelope")
        assert res["pair"] == (DIV, N2)

    def test_the_existing_child_is_never_the_second_daughter(self):
        # Only C1 is near either daughter; C1 must not be offered as the orphan sister.
        nodes = {DIV: (5, (10, 50, 50)), C1: (6, (10, 50, 50))}
        res = classify(pregraph(nodes, [(DIV, C1)]),
                       daughters=[(6, um(10, 52, 50)), (6, um(10, 48, 50))])
        assert res["bucket"] == "C"

    def test_second_daughter_looked_up_in_its_own_frame(self):
        nodes = base_nodes() | {N2: (7, (10, 47.5, 50))}
        pre = pregraph(nodes, [(DIV, C1)])
        assert classify(pre)["bucket"] == "C"          # nothing at t=6
        res = classify(pre, daughters=[DAUGHTERS[0], (7, um(10, 47, 50))])
        assert (res["bucket"], res["pair"]) == ("A", (DIV, N2))


class TestPeakBuckets:
    def pre(self):
        nodes = base_nodes()
        del nodes[N2]
        return pregraph(nodes, [(DIV, C1)])

    def test_free_peak_below_threshold(self):
        res = classify(self.pre(), low=low_pool([(6, (10, 47, 50), 0.6)]))
        assert res["bucket"] == "B"
        assert res["weak_score"] == pytest.approx(0.6)
        assert "below the detection threshold" in res["detail"]

    def test_free_peak_above_threshold(self):
        res = classify(self.pre(), low=low_pool([(6, (10, 47, 50), 0.98)]))
        assert res["bucket"] == "B"
        assert "dropped before the pass" in res["detail"]

    def test_sister_peak_is_not_the_daughters(self):
        # The dump holds every node's own peak: the sister's, 6 um from the daughter,
        # is inside the tolerance and must not be read as a missed cell.
        res = classify(self.pre(), low=low_pool([(6, (10, 53, 50), 0.99)]))
        assert (res["bucket"], res["detail"]) == ("C", dx.C_OWNED)

    def test_sister_peak_displaced_from_her_node_is_still_hers(self):
        # 2.4 um from the node (past x138's 2.0) but nearer the node than the daughter.
        res = classify(self.pre(), low=low_pool([(6, (10, 50.6, 50), 0.99)]))
        assert (res["bucket"], res["detail"]) == ("C", dx.C_OWNED)

    def test_nearest_free_peak_wins(self):
        low = low_pool([(6, (10, 53, 50), 0.99), (6, (10, 44, 50), 0.4), (6, (10, 47.5, 50), 0.7)])
        res = classify(self.pre(), low=low)
        assert res["weak_score"] == pytest.approx(0.7)

    def test_no_peak(self):
        res = classify(self.pre(), low=low_pool([(6, (10, 120, 50), 0.9), (5, (10, 47, 50), 0.9)]))
        assert (res["bucket"], res["detail"]) == ("C", dx.C_NOTHING)

    def test_no_cache_is_said_not_guessed(self):
        res = classify(self.pre(), low=None)
        assert res["bucket"] == "C"
        assert "no low-detection cache" in res["detail"]


class TestGateAblation:
    def test_counts_single_gate_failures_and_keys_pairs_by_dataset(self):
        rows = [
            {**cand_row(dc=True), "dataset": "m1"},                 # true, dc only
            {**cand_row(dc=True), "dataset": "m2"},                 # same ids, other movie: false
            {**cand_row(cand=9, dc=True, sym=True), "dataset": "m1"},   # two gates: not counted
            {**cand_row(cand=8), "dataset": "m1"},                  # passes all, false
        ]
        ab = dx.gate_ablation(rows, {("m1", DIV, N2)})
        assert ab["dc"] == {"true": 1, "false": 1}
        assert sum(ab["sym"].values()) == 0
        assert ab["all_pass"] == {"false": 1}


def test_candidate_columns_match_the_builder_dump():
    cols = re.search(r"_VOLT_DIVDIAG_COLUMNS = \[(.*?)\]", builder._DIVDIAG_CODE, re.S).group(1)
    logged = re.findall(r'"(\w+)"', cols)
    assert sorted(logged) == sorted((*dx.INT_COLS, *dx.FLOAT_COLS))
    assert [c[5:] for c in logged if c.startswith("pass_")] == list(dx.GATES)


# ---- end to end, through the official division scorer --------------------------------

# Two GT divisions 60 um apart. Division 1 is recovered by a fork the pass added;
# division 2's orphan sister was refused by the divergence gate.
GT_POS = {
    "gp": (4, (10, 50, 50)), "d": (5, (10, 50, 50)),
    "c1": (6, (10, 53, 50)), "c2": (6, (10, 47, 50)),
    "g1": (7, (10, 55, 50)), "g2": (7, (10, 45, 50)),
}
GT_EDGES = [("gp", "d"), ("d", "c1"), ("d", "c2"), ("c1", "g1"), ("c2", "g2")]


def shifted(pos, dy):
    return {k: (t, (z, y + dy, x)) for k, (t, (z, y, x)) in pos.items()}


def gt_graph():
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, 0.0)
    for dy in (0, 60):
        ids = {k: g.add_node({"t": t, **dict(zip("zyx", vox(*p), strict=True))})
               for k, (t, p) in shifted(GT_POS, dy).items()}
        for a, b in GT_EDGES:
            g.add_edge(ids[a], ids[b], {})
    return g


def prediction():
    """The final CSV's nodes (t, um) and edges, the pass's input edges, and the two forks."""
    nodes, edges = {}, []
    for base, dy in ((10, 0), (20, 60)):
        ids = {}
        for k, (key, (t, p)) in enumerate(shifted(GT_POS, dy).items()):
            ids[key] = base + k
            nodes[base + k] = (t, p)
        edges += [(ids[a], ids[b]) for a, b in GT_EDGES]
    fork_1, fork_2 = (11, 13), (21, 23)          # d -> c2 of each division
    pre_edges = [e for e in edges if e not in (fork_1, fork_2)]
    final_edges = [e for e in edges if e != fork_2]
    return nodes, pre_edges, final_edges, fork_1, fork_2


def frames(nodes, edges):
    ids = list(nodes)
    zyx = np.asarray([vox(*nodes[i][1]) for i in ids])
    node_df = pl.DataFrame({"node_id": ids, "t": [nodes[i][0] for i in ids],
                            "z": zyx[:, 0], "y": zyx[:, 1], "x": zyx[:, 2]})
    edge_df = pl.DataFrame({"source_id": [a for a, _ in edges], "target_id": [b for _, b in edges]},
                           schema={"source_id": pl.Int64, "target_id": pl.Int64})
    return node_df, edge_df


def test_diagnose_dataset_against_the_official_scorer():
    nodes, pre_edges, final_edges, fork_1, fork_2 = prediction()
    cands = {fork_1: cand_row(*fork_1), fork_2: cand_row(*fork_2, div=True)}
    rows, true_pairs, n_fp = dx.diagnose_dataset(
        "m", *frames(nodes, final_edges), gt_graph(), pregraph(nodes, pre_edges), None, cands,
        {fork_1})
    assert sorted(r["bucket"] for r in rows) == ["A", "TP"]
    miss = next(r for r in rows if not r["scored_tp"])
    assert (miss["first_gate"], miss["n_failing"]) == ("div", 1)
    assert true_pairs == {("m", *fork_1), ("m", *fork_2)}
    assert n_fp == 0


def test_main_end_to_end(tmp_path):
    nodes, pre_edges, final_edges, fork_1, fork_2 = prediction()
    name = "aaaa_movie"
    data_dir, diag = tmp_path / "data", tmp_path / "diag"
    data_dir.mkdir()
    diag.mkdir()
    gt_graph().to_geff(str(data_dir / f"{name}.geff"))

    ids = sorted(nodes)
    np.savez_compressed(
        diag / f"{name}_graph.npz", node_id=np.asarray(ids),
        t=np.asarray([nodes[i][0] for i in ids]),
        zyx=np.asarray([vox(*nodes[i][1]) for i in ids]),
        edge_src=np.asarray([a for a, _ in pre_edges]),
        edge_tgt=np.asarray([b for _, b in pre_edges]))
    with open(diag / f"{name}_cands.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[*dx.INT_COLS, *dx.FLOAT_COLS])
        w.writeheader()
        w.writerow(cand_row(*fork_1))
        w.writerow(cand_row(*fork_2, div=True))
    with open(diag / f"{name}_added.csv", "w", newline="") as fh:
        fh.write(f"source_id,target_id\n{fork_1[0]},{fork_1[1]}\n")

    node_df, edge_df = frames(nodes, final_edges)
    sub = pl.concat([
        node_df.select(pl.lit("node").alias("row_type"), pl.col("node_id"), pl.col("t"),
                       *(pl.col(c).round().cast(pl.Int64) for c in "zyx"),
                       pl.lit(-1).alias("source_id"), pl.lit(-1).alias("target_id")),
        edge_df.select(pl.lit("edge").alias("row_type"),
                       *(pl.lit(-1).alias(c) for c in ("node_id", "t", "z", "y", "x")),
                       pl.col("source_id"), pl.col("target_id")),
    ], how="vertical_relaxed").with_columns(pl.lit(name).alias("dataset")).with_row_index("id")
    sub.write_csv(tmp_path / "submission.csv")

    out = tmp_path / "report.csv"
    assert dx.main(["--csv", str(tmp_path / "submission.csv"), "--diag-dir", str(diag),
                    "--data-dir", str(data_dir), "--out", str(out)]) == 0
    report = pl.read_csv(out)
    assert sorted(report["bucket"].to_list()) == ["A", "TP"]
    labelled = pl.read_csv(out.with_name("report_candidates.csv"))
    assert labelled["is_true"].to_list() == [True, True]
