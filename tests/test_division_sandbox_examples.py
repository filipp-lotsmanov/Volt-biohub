"""Freeze edge and division metrics on small, hand-crafted graph examples.

These in-memory cases mirror the interactive division sandbox without relying
on ignored ``visualize/sandbox_examples/*.json`` files. They exercise both the
low-level sandbox pipeline and the public :func:`evaluate` entry point.
"""

from __future__ import annotations

from typing import NamedTuple

import polars as pl
import pytest
import tracksdata as td

from volt_biohub.division_metrics import evaluate_divisions, score_divisions
from volt_biohub.metrics import _evaluate, _evaluate_matched_graph, _jaccard, evaluate


class GraphSpec(NamedTuple):
    """Minimal graph representation used by the mock sandbox cases."""

    nodes: dict[str, tuple[int, float]]  # name -> (t, y)
    edges: list[tuple[str, str]]


class SandboxMetrics(NamedTuple):
    edge_tp: int
    edge_fp: int
    edge_fn: int
    div_tp: int
    div_fp: int
    div_fn: int


class SandboxCase(NamedTuple):
    gt: GraphSpec
    pred: GraphSpec
    expected: SandboxMetrics
    max_distance: float = 1.0


_GT_DIVISION = GraphSpec(
    nodes={
        "P": (0, 0.0),
        "D": (1, 0.0),
        "C1": (2, 5.0),
        "C2": (2, -5.0),
        "G1": (3, 5.0),
        "G2": (3, -5.0),
    },
    edges=[
        ("P", "D"),
        ("D", "C1"),
        ("D", "C2"),
        ("C1", "G1"),
        ("C2", "G2"),
    ],
)


def _hack2_case() -> SandboxCase:
    """Create the exact ``hack2.json`` topology as self-contained test data."""
    return SandboxCase(
        gt=GraphSpec(
            nodes={
                "61": (1, 10.0),
                "62": (2, 10.0),
                "63": (3, 0.0),
                "64": (3, 20.0),
                "66": (4, 20.0),
                "98": (4, 0.0),
                "99": (1, -40.0),
                "100": (2, -40.0),
                "101": (3, -25.0),
                "102": (4, -25.0),
                "103": (3, -45.0),
                "104": (4, -45.0),
            },
            edges=[
                ("61", "62"),
                ("62", "63"),
                ("62", "64"),
                ("64", "66"),
                ("63", "98"),
                ("99", "100"),
                ("100", "101"),
                ("101", "102"),
                ("100", "103"),
                ("103", "104"),
            ],
        ),
        pred=GraphSpec(
            nodes={
                "105": (0, 15.0),
                "106": (2, 15.0),
                "107": (3, 30.0),
                "108": (3, 5.0),
                "109": (4, 5.0),
                "110": (4, 30.0),
                "114": (4, -20.0),
                "115": (3, -50.0),
                "116": (4, -50.0),
                "121": (5, -50.0),
                "122": (1, 55.0),
                "124": (2, 55.0),
                "130": (3, 55.0),
                "132": (1, 15.0),
                "153": (2, -35.0),
            },
            edges=[
                ("106", "107"),
                ("108", "109"),
                ("107", "110"),
                ("115", "116"),
                ("116", "121"),
                ("122", "124"),
                ("124", "108"),
                ("124", "130"),
                ("130", "114"),
                ("105", "132"),
                ("132", "106"),
                ("105", "122"),
                ("122", "153"),
                ("106", "115"),
            ],
        ),
        expected=SandboxMetrics(
            edge_tp=5,
            edge_fp=4,
            edge_fn=5,
            div_tp=0,
            div_fp=4,
            div_fn=2,
        ),
        max_distance=10.0,
    )


CASES: dict[str, SandboxCase] = {
    "hack2": _hack2_case(),
    "perfect_division": SandboxCase(
        gt=_GT_DIVISION,
        pred=_GT_DIVISION,
        expected=SandboxMetrics(
            edge_tp=5, edge_fp=0, edge_fn=0,
            div_tp=1, div_fp=0, div_fn=0,
        ),
    ),
    "missed_division": SandboxCase(
        gt=_GT_DIVISION,
        pred=GraphSpec(
            nodes={
                "P": (0, 0.0),
                "D": (1, 0.0),
                "C1": (2, 5.0),
                "G1": (3, 5.0),
            },
            edges=[("P", "D"), ("D", "C1"), ("C1", "G1")],
        ),
        expected=SandboxMetrics(
            edge_tp=3, edge_fp=0, edge_fn=2,
            div_tp=0, div_fp=0, div_fn=1,
        ),
    ),
    "delayed_local_division": SandboxCase(
        gt=_GT_DIVISION,
        pred=GraphSpec(
            nodes={
                "P": (0, 0.0),
                "D": (1, 0.0),
                "M": (2, 0.0),
                "G1": (3, 5.0),
                "G2": (3, -5.0),
            },
            edges=[("P", "D"), ("D", "M"), ("M", "G1"), ("M", "G2")],
        ),
        expected=SandboxMetrics(
            edge_tp=1, edge_fp=3, edge_fn=4,
            div_tp=1, div_fp=0, div_fn=0,
        ),
    ),
    "dummy_branch_same_lineage": SandboxCase(
        gt=_GT_DIVISION,
        pred=GraphSpec(
            nodes={
                "P": (0, 0.0),
                "D": (1, 0.0),
                "C1": (2, 5.0),
                "X": (2, 50.0),
                "G2": (3, -5.0),
            },
            # Both GT daughter matches lie on D's C1 branch. X is a dummy
            # branch, so D must be rejected as TP and counted as FP.
            edges=[("P", "D"), ("D", "C1"), ("D", "X"), ("C1", "G2")],
        ),
        expected=SandboxMetrics(
            edge_tp=2, edge_fp=2, edge_fn=3,
            div_tp=0, div_fp=1, div_fn=1,
        ),
    ),
    "spurious_linear_division": SandboxCase(
        gt=GraphSpec(
            nodes={"A": (0, 20.0), "B": (1, 20.0), "C": (2, 20.0)},
            edges=[("A", "B"), ("B", "C")],
        ),
        pred=GraphSpec(
            nodes={
                "A": (0, 20.0),
                "B": (1, 20.0),
                "C1": (2, 20.0),
                "C2": (2, 25.0),
            },
            edges=[("A", "B"), ("B", "C1"), ("B", "C2")],
        ),
        expected=SandboxMetrics(
            edge_tp=2, edge_fp=1, edge_fn=0,
            div_tp=0, div_fp=1, div_fn=0,
        ),
    ),
    "cross_component_children": SandboxCase(
        gt=GraphSpec(
            nodes={
                "A0": (0, 0.0),
                "A1": (1, 0.0),
                "B0": (0, 20.0),
                "B1": (1, 20.0),
            },
            edges=[("A0", "A1"), ("B0", "B1")],
        ),
        pred=GraphSpec(
            nodes={"P": (0, 0.0), "C1": (1, 0.0), "C2": (1, 20.0)},
            edges=[("P", "C1"), ("P", "C2")],
        ),
        expected=SandboxMetrics(
            edge_tp=1, edge_fp=1, edge_fn=1,
            div_tp=0, div_fp=1, div_fn=0,
        ),
    ),
    "cross_component_grandchild_fallback": SandboxCase(
        gt=GraphSpec(
            nodes={
                "A1": (1, 0.0),
                "A2": (2, 0.0),
                "B1": (1, 20.0),
                "B2": (2, 20.0),
            },
            edges=[("A1", "A2"), ("B1", "B2")],
        ),
        pred=GraphSpec(
            nodes={
                "F": (0, 10.0),
                "A": (1, 0.0),
                "U": (1, 40.0),
                "B": (2, 20.0),
            },
            edges=[("F", "A"), ("F", "U"), ("U", "B")],
        ),
        expected=SandboxMetrics(
            edge_tp=0, edge_fp=1, edge_fn=2,
            div_tp=0, div_fp=1, div_fn=0,
        ),
        max_distance=1.0,
    ),
    "disconnected_daughter": SandboxCase(
        gt=_GT_DIVISION,
        pred=GraphSpec(
            nodes=_GT_DIVISION.nodes,
            edges=[("P", "D"), ("D", "C1"), ("C1", "G1"), ("C2", "G2")],
        ),
        expected=SandboxMetrics(
            edge_tp=4, edge_fp=0, edge_fn=1,
            div_tp=0, div_fp=0, div_fn=1,
        ),
    ),
}


@pytest.fixture(scope="module", autouse=True)
def division_clip_fixture() -> None:
    """Override the unrelated dataset-backed autouse fixture for these tests."""


def _graph_from_spec_with_ids(
    spec: GraphSpec,
) -> tuple[td.graph.InMemoryGraph, dict[str, int]]:
    """Build a y-only graph and return its test-name-to-node-ID mapping."""
    graph = td.graph.InMemoryGraph()
    graph.add_node_attr_key("z", pl.Float64, 0.0)
    graph.add_node_attr_key("y", pl.Float64, 0.0)
    graph.add_node_attr_key("x", pl.Float64, 0.0)
    node_ids = {
        name: graph.add_node({"t": t, "z": 0.0, "y": y, "x": 0.0})
        for name, (t, y) in spec.nodes.items()
    }
    for source, target in spec.edges:
        graph.add_edge(node_ids[source], node_ids[target], {})
    return graph, node_ids


def _graph_from_spec(spec: GraphSpec) -> td.graph.InMemoryGraph:
    """Build the same y-only graph used by the interactive sandbox."""
    graph, _ = _graph_from_spec_with_ids(spec)
    return graph


def _compute_sandbox_metrics(
    pred: td.graph.BaseGraph,
    gt: td.graph.BaseGraph,
    max_distance: float,
) -> tuple[SandboxMetrics, float]:
    """Replicate the sandbox metric pipeline."""
    pred_copy = pred.copy()
    _evaluate(pred_copy, gt, "jaccard", None, max_distance)
    edge_attrs = _evaluate_matched_graph(pred_copy, gt)
    edge_tp = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    edge_fp = int(edge_attrs["pred_valid"].sum()) - edge_tp
    edge_fn = gt.num_edges() - edge_tp

    divisions = evaluate_divisions(pred, gt, max_distance=max_distance)
    counts = SandboxMetrics(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        div_tp=divisions.tp,
        div_fp=divisions.fp,
        div_fn=divisions.fn,
    )
    return counts, _jaccard(edge_tp, edge_fp, edge_fn)


def _build_case(name: str) -> tuple[SandboxCase, td.graph.BaseGraph, td.graph.BaseGraph]:
    case = CASES[name]
    return case, _graph_from_spec(case.pred), _graph_from_spec(case.gt)


@pytest.mark.parametrize("name", sorted(CASES))
def test_mock_sandbox_case_matches_frozen_counts(name: str) -> None:
    case, pred, gt = _build_case(name)

    counts, edge_jaccard = _compute_sandbox_metrics(pred, gt, case.max_distance)

    assert counts == case.expected, f"{name}: {counts} != {case.expected}"
    expected_jaccard = _jaccard(
        case.expected.edge_tp,
        case.expected.edge_fp,
        case.expected.edge_fn,
    )
    assert edge_jaccard == pytest.approx(expected_jaccard)


@pytest.mark.parametrize("name", sorted(CASES))
def test_evaluate_agrees_with_mock_sandbox(name: str) -> None:
    """The public evaluator must agree with the sandbox-style pipeline."""
    case, pred, gt = _build_case(name)

    result = evaluate(pred, gt, max_distance=case.max_distance)

    assert (result.edge_tp, result.edge_fp, result.edge_fn) == (
        case.expected.edge_tp,
        case.expected.edge_fp,
        case.expected.edge_fn,
    )
    assert (result.division_tp, result.division_fp, result.division_fn) == (
        case.expected.div_tp,
        case.expected.div_fp,
        case.expected.div_fn,
    )


def test_hack2_classifies_all_four_predicted_forks_as_fp() -> None:
    """Freeze the exact exploit topology's fork identities, not only its count."""
    case = CASES["hack2"]
    pred, pred_ids = _graph_from_spec_with_ids(case.pred)
    gt = _graph_from_spec(case.gt)

    result = score_divisions(pred, gt, max_distance=case.max_distance)
    fp_names = {
        name for name, node_id in pred_ids.items()
        if node_id in result.fp_forks
    }

    assert fp_names == {"105", "106", "122", "124"}
