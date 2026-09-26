"""The division-diagnosis patch for x138: read-only, and the gate values it logs are right.

x138's notebook is third-party and never committed, so the patch is applied to a
stand-in for cell 5 with the same names and call signature. The stand-in's pass is a
simple one; what is under test is the wrapper around it: the pass must return exactly
what it returns unpatched, see only its own caches, and survive a failing logger.
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent / "kaggle" / "forks"))

import build_x138_variant as builder

STAND_IN = '''
SAFE_DIV_MAX_UM = 9.0
SAFE_DIV_SISTER_MAX_UM = 14.0
SAFE_DIV_SISTER_SYMMETRY_TAU = 0.6
SAFE_DIV_EXISTING_CHILD_MAX_UM = 10.0
SAFE_DIV_DIVERGE_UM = 2.25
SAFE_DIV_REQUIRE_DIVERGENCE = True
SAFE_DIV_REQUIRE_MUTUAL_NN = True
USE_DEEPCENTER_VETO = True
DEEPCENTER_SAFE_DIV_VETO = True
DEEPCENTER_SAFE_DIV_THRESHOLD = 0.25
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def node_point(node):
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def _position_um(node):
    return np.array(node_point(node)) * np.array(VOXEL_SCALE_UM)


def edge_distance_um(a, b):
    return float(np.linalg.norm(_position_um(a) - _position_um(b)))


def add_safe_divisions_postlink(nodes_by_id, edges, stats, dataset=None,
                                deepcenter_bundle=None, frame_cache=None,
                                deepcenter_cache=None):
    """Stand-in: fork every source with one child to its nearest orphan within 9 um."""
    has_parent = {e["target_id"] for e in edges}
    kids = {}
    for e in edges:
        kids.setdefault(e["source_id"], []).append(e["target_id"])
    out = list(edges)
    for s, ks in sorted(kids.items()):
        if len(ks) != 1:
            continue
        t = nodes_by_id[s]["t"]
        orphans = [i for i, n in nodes_by_id.items() if n["t"] == t + 1 and i not in has_parent]
        near = sorted((edge_distance_um(nodes_by_id[s], nodes_by_id[i]), i) for i in orphans)
        if near and near[0][0] <= SAFE_DIV_MAX_UM:
            deepcenter_score_point(dataset, t + 1, node_point(nodes_by_id[near[0][1]]),
                                   deepcenter_bundle, frame_cache, deepcenter_cache)
            out.append({"source_id": s, "target_id": near[0][1], "safe_division": True})
            has_parent.add(near[0][1])
            stats["added"] += 1
    return out


def filter_short_track_components(nodes_by_id, edges):
    return edges
'''


def um_node(t, z, y, x):
    """A node at micrometre position (z, y, x), stored in original voxels as x138 does."""
    return {"t": t, "z": z / 1.625, "y": y / 0.40625, "x": x / 0.40625}


def movie():
    """Divider 1 (t=0) -> child 2 (t=1); orphans 3 (true sister) and 4 (farther); both
    daughters have one successor at t=2, diverging by more than 2.25 um."""
    nodes = {
        1: um_node(0, 10, 50, 50),
        2: um_node(1, 10, 53, 50), 3: um_node(1, 10, 47, 50), 4: um_node(1, 10, 40, 50),
        5: um_node(2, 10, 56, 50), 6: um_node(2, 10, 43, 50),
    }
    edges = [{"source_id": 1, "target_id": 2}, {"source_id": 2, "target_id": 5},
             {"source_id": 3, "target_id": 6}]
    return nodes, edges


def namespace(src, dc):
    ns = {"np": np, "cKDTree": cKDTree, "math": math, "os": __import__("os"), "Path": Path}
    exec(compile(src, "stand_in_cell5", "exec"), ns)
    ns["deepcenter_score_point"] = dc
    return ns


def run(src, dc, tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("VOLT_DIVDIAG_DIR", str(tmp_path / "diag"))
    ns = namespace(src, dc)
    nodes, edges = movie()
    stats = {"added": 0}
    frame_cache, heat_cache = {}, {}
    out = ns["add_safe_divisions_postlink"](nodes, edges, stats, dataset="m",
                                            deepcenter_bundle="bundle", frame_cache=frame_cache,
                                            deepcenter_cache=heat_cache, **kw)
    return out, stats, frame_cache, heat_cache


def recording_dc(scores):
    """DeepCenter stand-in: a fixed score per y position; writes into the caches it is given."""
    def dc(dataset, t, point, bundle, frame_cache, heat_cache):
        frame_cache[(t, "frame")] = 1
        heat_cache[(t, round(point[1]))] = 1
        return scores.get(round(point[1] * 0.40625))
    return dc


def cands(tmp_path):
    with open(tmp_path / "diag" / "m_cands.csv", newline="") as fh:
        return {int(r["cand_id"]): r for r in csv.DictReader(fh)}


def patched():
    return builder.patch_division_diagnosis(STAND_IN, "/nonexistent")


def test_output_stats_and_caches_are_untouched(tmp_path, monkeypatch):
    dc = recording_dc({47: 0.9, 40: 0.9})
    plain = run(STAND_IN, dc, tmp_path / "a", monkeypatch)
    wrapped = run(patched(), dc, tmp_path / "b", monkeypatch)
    assert wrapped == plain                       # edges, stats, and both cache dicts
    assert not (tmp_path / "a" / "diag").exists()


def test_dumps_graph_candidates_and_added_forks(tmp_path, monkeypatch):
    out, *_ = run(patched(), recording_dc({47: 0.9, 40: 0.9}), tmp_path, monkeypatch)
    diag = tmp_path / "diag"
    g = np.load(diag / "m_graph.npz")
    assert g["node_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert list(zip(g["edge_src"].tolist(), g["edge_tgt"].tolist(), strict=True)) == [
        (1, 2), (2, 5), (3, 6)]
    with open(diag / "m_added.csv", newline="") as fh:
        added = [(int(r["source_id"]), int(r["target_id"])) for r in csv.DictReader(fh)]
    assert added == [(e["source_id"], e["target_id"]) for e in out if e.get("safe_division")]
    assert added == [(1, 3)]


def test_gate_values_and_flags(tmp_path, monkeypatch):
    run(patched(), recording_dc({47: 0.9, 40: 0.1}), tmp_path, monkeypatch)
    rows = cands(tmp_path)
    sister, far = rows[3], rows[4]
    assert float(sister["d_exist"]) == pytest.approx(3.0)
    assert float(sister["d_par"]) == pytest.approx(3.0)
    assert float(sister["d_sis"]) == pytest.approx(6.0)
    assert float(sister["div_margin"]) == pytest.approx(13.0 - 6.0)
    assert int(sister["nn_rank"]) == 0
    assert all(int(sister[f"pass_{g}"]) for g in ("exist", "par", "sis", "nn", "div", "dc", "sym"))
    # Orphan 4: 10 um from the divider -- refused by the 9.0 gate but inside the 14 um
    # envelope, so it is logged -- second nearest to the child, no successor, DeepCenter 0.1.
    assert float(far["d_par"]) == pytest.approx(10.0)
    assert float(far["d_sis"]) == pytest.approx(13.0)
    assert int(far["nn_rank"]) == 1
    assert (int(far["pass_exist"]), int(far["pass_sis"])) == (1, 1)
    assert (int(far["pass_par"]), int(far["pass_nn"]), int(far["pass_div"]),
            int(far["pass_dc"])) == (0, 0, 0, 0)
    assert int(far["cand_succ"]) == 0 and far["div_margin"] == "nan"
    assert float(far["symmetry"]) == pytest.approx(7.0 / 6.5)
    assert int(far["pass_sym"]) == 0


def test_missing_deepcenter_score_is_accepted(tmp_path, monkeypatch):
    run(patched(), recording_dc({}), tmp_path, monkeypatch)
    sister = cands(tmp_path)[3]
    assert sister["dc_score"] == "nan" and int(sister["pass_dc"]) == 1


def test_a_failing_logger_never_costs_the_run(tmp_path, monkeypatch, capsys):
    calls = []

    def exploding(dataset, t, point, bundle, frame_cache, heat_cache):
        calls.append(frame_cache)
        if len(calls) == 1:                       # the logger scores first; the pass second
            raise RuntimeError("heatmap unavailable")
        return 0.9

    plain = run(STAND_IN, recording_dc({47: 0.9}), tmp_path / "a", monkeypatch)
    out, stats, *_ = run(patched(), exploding, tmp_path / "b", monkeypatch)
    assert (out, stats) == plain[:2]
    assert "VOLT division diagnosis FAILED (output unaffected)" in capsys.readouterr().out


def test_anchor_must_appear_exactly_once():
    with pytest.raises(SystemExit, match="found 0 times"):
        builder.patch_division_diagnosis("def something_else(): pass\n", "/d")
    with pytest.raises(SystemExit, match="found 2 times"):
        builder.patch_division_diagnosis(builder._DIVDIAG_ANCHOR * 2, "/d")
