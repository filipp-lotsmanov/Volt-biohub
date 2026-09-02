import warnings
from typing import NamedTuple

import polars as pl
import tracksdata as td


class DivisionCounts(NamedTuple):
    """Counts for division event evaluation."""

    tp: int
    fn: int
    fp: int


class DivisionScores(NamedTuple):
    """Result of :func:`score_divisions`.

    Attributes
    ----------
    scores : dict[int, int]
        Mapping from GT dividing-node ID to 1 (recovered) or 0 (not).
    tp_forks : set[int]
        Predicted dividing nodes paired to GT divisions.
    fp_forks : set[int]
        Predicted dividing nodes that were considered for a GT division
        but did not become a true positive, including local-topology
        rejects, bipartite leftovers, evaluable spurious forks, malformed
        local branches, and forks whose branch evidence spans distinct GT
        components.
    """

    scores: dict[int, int]
    tp_forks: set[int]
    fp_forks: set[int]


def _reset_matching_attrs(graph: td.graph.BaseGraph) -> None:
    """Reset any pre-existing match attrs in place so a fresh ``.match()`` isn't
    contaminated by stale values carried in from a previous matching pass."""
    node_keys = graph.node_attr_keys()
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in node_keys:
        node_ids = graph.node_ids()
        if len(node_ids) > 0:
            reset: dict = {td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: -1}
            if td.DEFAULT_ATTR_KEYS.MATCH_SCORE in node_keys:
                reset[td.DEFAULT_ATTR_KEYS.MATCH_SCORE] = 0.0
            graph.update_node_attrs(node_ids=node_ids, attrs=reset)
    if td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK in graph.edge_attr_keys():
        edge_ids = graph.edge_ids()
        if len(edge_ids) > 0:
            graph.update_edge_attrs(
                edge_ids=edge_ids,
                attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: False},
            )


def extract_divisions(
    graph: td.graph.BaseGraph,
) -> dict[int, td.graph.BaseGraph]:
    """Extract individual division events as separate subgraphs.

    Each division event includes the parent of the dividing node, the
    dividing node, its children, and the grandchildren::

        parent → divider → child1 → grandchild1
                         → child2 → grandchild2

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The input tracking graph.

    Returns
    -------
    dict[int, td.graph.BaseGraph]
        Mapping from dividing node ID to a subgraph containing the
        parent, divider, children, and grandchildren.
    """
    divisions: dict[int, td.graph.BaseGraph] = {}
    for div_node in graph.dividing_nodes():
        parents = graph.predecessors(div_node)
        children = graph.successors(div_node)
        grandchildren = [gc for child in children for gc in graph.successors(child)]
        keep = [*parents, div_node, *children, *grandchildren]
        divisions[div_node] = graph.filter(node_ids=keep).subgraph()
    return divisions


def match_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> dict[int, td.graph.BaseGraph]:
    """Match the predicted graph against each GT division subgraph.

    Extracts division events from *gt_graph* via :func:`extract_divisions`,
    then runs ``pred_graph.match(gt_div, ...)`` for each one independently.
    A fresh copy of *pred_graph* is used per division so matchings don't
    interfere.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    dict[int, td.graph.BaseGraph]
        Mapping from GT dividing-node ID to the matched copy of
        *pred_graph* for that division.
    """
    from tracksdata.metrics import DistanceMatching

    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    gt_divisions = extract_divisions(gt_graph)
    matched: dict[int, td.graph.BaseGraph] = {}

    from tracksdata.options import get_options, set_options

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        for div_node, gt_div in gt_divisions.items():
            pred_copy = pred_graph.copy()
            _reset_matching_attrs(pred_copy)
            with warnings.catch_warnings():
                from scipy.sparse import SparseEfficiencyWarning

                warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
                pred_copy.match(gt_div, matching=matching)
            matched[div_node] = pred_copy
    finally:
        set_options(show_progress=prev_show_progress)

    return matched


def _match_full(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> td.graph.BaseGraph:
    """Match the full pred graph against the full GT graph, return the matched copy."""
    from tracksdata.metrics import DistanceMatching

    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    pred_copy = pred_graph.copy()
    _reset_matching_attrs(pred_copy)

    from tracksdata.options import get_options, set_options

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            from scipy.sparse import SparseEfficiencyWarning

            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            pred_copy.match(gt_graph, matching=matching)
    finally:
        set_options(show_progress=prev_show_progress)

    return pred_copy


def _matched_node_attrs(graph: td.graph.BaseGraph) -> pl.DataFrame:
    """Return pred/GT node-ID pairs for matched prediction nodes."""
    node_attrs = graph.node_attrs(
        attr_keys=[
            td.DEFAULT_ATTR_KEYS.NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        ],
    )
    return node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )


def _matched_division_nodes(
    matched_attrs: pl.DataFrame,
    gt_div: td.graph.BaseGraph,
    divider_id: int,
) -> tuple[set[int], list[set[int]]] | None:
    """Group matched pred nodes by their role in a GT division window.

    The parent side contains the GT divider (the parent cell) and its
    immediate predecessor (the grandparent). Each daughter side contains
    one GT child and its immediate successors (the grandchildren).
    """
    if matched_attrs.is_empty():
        return None

    node_to_gt = dict(
        zip(
            matched_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
            matched_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list(),
            strict=True,
        )
    )
    gt_children = gt_div.successors(divider_id)
    if len(gt_children) < 2:
        return None

    gt_parent_ids = {divider_id, *gt_div.predecessors(divider_id)}
    parent_ids = {pred_id for pred_id, gt_id in node_to_gt.items() if gt_id in gt_parent_ids}
    daughter_ids = [
        {pred_id for pred_id, gt_id in node_to_gt.items() if gt_id in {child, *gt_div.successors(child)}}
        for child in gt_children
    ]
    if not parent_ids or sum(bool(ids) for ids in daughter_ids) < 2:
        return None
    return parent_ids, daughter_ids


def _is_strongly_connected_division(
    pred_graph: td.graph.BaseGraph,
    pred_div: int,
    parent_ids: set[int],
    daughter_ids: list[set[int]],
) -> bool:
    """Check a predicted division's local directed topology.

    The prediction window mirrors :func:`extract_divisions`: an immediate
    predecessor (grandparent), *pred_div* (parent), its children, and their
    children (grandchildren). The parent match must be the fork itself or
    its immediate predecessor. Matches from at least two GT daughter
    lineages must occur in two distinct predicted child lineages.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    pred_div : int
        Candidate predicted dividing node (the parent/fork).
    parent_ids : set[int]
        Prediction node IDs matched to the GT parent side (grandparent or
        dividing parent).
    daughter_ids : list[set[int]]
        Prediction node IDs matched to each GT daughter lineage (child or
        grandchild), grouped by lineage.

    Returns
    -------
    bool
        Whether the local prediction topology connects the parent side to
        at least two distinct daughter lineages through *pred_div*.
    """
    pred_parent_ids = {pred_div, *pred_graph.predecessors(pred_div)}
    if pred_parent_ids.isdisjoint(parent_ids):
        return False

    pred_lineages = [{child, *pred_graph.successors(child)} for child in pred_graph.successors(pred_div)]
    lineage_edges = {
        gt_lineage: {
            pred_lineage for pred_lineage, pred_ids in enumerate(pred_lineages) if not matched_ids.isdisjoint(pred_ids)
        }
        for gt_lineage, matched_ids in enumerate(daughter_ids)
    }
    return len(_bipartite_max_matching(list(lineage_edges), lineage_edges)) >= 2


def _bipartite_max_matching(
    left: list[int],
    edges: dict[int, set[int]],
) -> dict[int, int]:
    """Maximum-cardinality bipartite matching via DFS augmenting paths.

    *edges* maps each left-side vertex to the set of adjacent right-side
    vertices. Returns only the matched pairs as a ``left → right`` dict.
    """
    match_r: dict[int, int] = {}
    match_l: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in edges.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in left:
        augment(u, set())

    return match_l


def score_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DivisionScores:
    """Score each GT division: 1 if the prediction recovers it, 0 otherwise.

    For each GT division, the predicted graph is matched against its
    parent/divider/children/grandchildren window. Candidate pred forks are
    restricted to the matched parent-side nodes and their immediate
    successors. A candidate is valid only when its local topology contains
    a matched parent and matches from two GT daughter lineages on distinct
    predicted child branches. A fork is rejected when two direct-child
    branches have nearest matched evidence in distinct reliable GT components.
    An unmatched child may use unambiguous grandchild evidence as a fallback;
    matched children take precedence over downstream matches.

    A maximum-cardinality bipartite matching is then computed so each pred
    fork serves at most one GT division, and each GT division is paired
    with at most one pred fork. A GT division scores 1 only if paired;
    rejected candidates and valid candidates left unpaired are returned as
    false-positive forks.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    DivisionScores
        The per-division scores and the predicted forks classified as true
        positives or false positives. False-positive forks include local
        topology rejects, cross-GT-component branches, locally merged branches,
        evaluable spurious forks, and valid candidates left unmatched by the
        bipartite pairing.
    """
    matched = match_divisions(
        pred_graph,
        gt_graph,
        scale,
        max_distance,
    )
    gt_divisions = extract_divisions(gt_graph)
    pred_div_nodes = {
        node_id for node_id in pred_graph.node_ids()
        if pred_graph.out_degree(node_id) >= 2
    }
    evaluable_forks, cross_component_forks, malformed_forks = (
        _pred_division_fork_sets(pred_graph, gt_graph, scale, max_distance)
    )
    invalid_forks = cross_component_forks | malformed_forks

    candidates: dict[int, set[int]] = {}
    considered: set[int] = set()
    for div_node, matched_pred in matched.items():
        matched_nodes = _matched_division_nodes(_matched_node_attrs(matched_pred), gt_divisions[div_node], div_node)
        if matched_nodes is None:
            candidates[div_node] = set()
            continue

        parent_ids, daughter_ids = matched_nodes
        local_nodes = parent_ids | {
            successor for parent_id in parent_ids for successor in matched_pred.successors(parent_id)
        }
        local_forks = local_nodes & pred_div_nodes
        considered |= local_forks
        candidates[div_node] = {
            pred_div
            for pred_div in local_forks - invalid_forks
            if _is_strongly_connected_division(matched_pred, pred_div, parent_ids, daughter_ids)
        }

    pairing = _bipartite_max_matching(list(candidates), candidates)
    scores = {div: int(div in pairing) for div in candidates}
    tp_forks = set(pairing.values())
    # Use a set union so forks supported by multiple FP rules are counted once.
    # Invalid forks were excluded from the pairing above and therefore cannot
    # also be true positives.
    fp_forks = (considered | evaluable_forks | invalid_forks) - tp_forks
    return DivisionScores(scores=scores, tp_forks=tp_forks, fp_forks=fp_forks)


def _gt_weak_component_ids(graph: td.graph.BaseGraph) -> dict[int, int]:
    """Map each GT node to its weakly connected component ID."""
    component_ids: dict[int, int] = {}
    for seed in graph.node_ids():
        if seed in component_ids:
            continue
        component_ids[seed] = seed
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in graph.successors(current) + graph.predecessors(current):
                if neighbor not in component_ids:
                    component_ids[neighbor] = seed
                    stack.append(neighbor)
    return component_ids


def _branch_component_evidence(
    graph: td.graph.BaseGraph,
    pred_div: int,
    child: int,
    pred_to_gt: dict[int, int],
    gt_component: dict[int, int],
) -> tuple[int | None, bool]:
    """Return one GT component for a predicted child branch.

    Direct-child evidence takes precedence over grandchildren so downstream
    errors do not invalidate a correctly matched division. Grandchildren are
    fallback evidence only when the child is unmatched. The boolean marks a
    locally merged branch that cannot be assigned uniquely to this fork.
    """
    if set(graph.predecessors(child)) != {pred_div}:
        return None, True
    if child in pred_to_gt:
        return gt_component[pred_to_gt[child]], False

    grandchildren = graph.successors(child)
    if any(set(graph.predecessors(node)) != {child} for node in grandchildren):
        return None, True

    components = {
        gt_component[pred_to_gt[node]]
        for node in grandchildren
        if node in pred_to_gt
    }
    if len(components) == 1:
        return next(iter(components)), False
    return None, False


def _pred_division_fork_sets(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> tuple[set[int], set[int], set[int]]:
    """Return evaluable, cross-component, and malformed predicted forks.

    Cross-component evidence must come from distinct direct-child branches.
    A matched child identifies its branch; otherwise an unambiguous matched
    grandchild may identify it. Merged local branches are malformed.
    """
    matched_pred = _match_full(pred_graph, gt_graph, scale, max_distance)
    matched_attrs = _matched_node_attrs(matched_pred)
    pred_to_gt = dict(
        zip(
            matched_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
            matched_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list(),
            strict=True,
        )
    )

    pred_forks = {
        node_id for node_id in matched_pred.node_ids()
        if matched_pred.out_degree(node_id) >= 2
    }
    evaluable_forks = {
        pred_id for pred_id in pred_forks
        if pred_id in pred_to_gt and gt_graph.out_degree(pred_to_gt[pred_id]) >= 1
    }

    gt_component = _gt_weak_component_ids(gt_graph)
    cross_component_forks: set[int] = set()
    malformed_forks: set[int] = set()
    for pred_id in pred_forks:
        branch_evidence: list[int] = []
        for child in matched_pred.successors(pred_id):
            component, malformed = _branch_component_evidence(
                matched_pred, pred_id, child, pred_to_gt, gt_component
            )
            if malformed:
                malformed_forks.add(pred_id)
                break
            if component is not None:
                branch_evidence.append(component)
        else:
            if len(set(branch_evidence)) >= 2:
                cross_component_forks.add(pred_id)

    return evaluable_forks, cross_component_forks, malformed_forks


def count_matched_pred_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> int:
    """Count predicted division nodes whose matched GT node is annotated.

    Matches the full predicted graph against the full GT graph.  Among
    predicted nodes that were matched to a GT node, counts how many are
    dividing (out-degree >= 2) in the prediction *and* whose matched GT
    node has at least one child.  A matched GT node with no children marks
    the end of the annotation — we can't tell whether the cell actually
    divided there, so such predicted divisions are excluded from the count
    (and therefore from the FP tally).

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    int
        Number of matched predicted division nodes.
    """
    evaluable_forks, _, _ = _pred_division_fork_sets(
        pred_graph, gt_graph, scale, max_distance
    )
    return len(evaluable_forks)


def evaluate_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DivisionCounts:
    """Compute TP, FN, and FP counts for division events.

    - **TP**: GT divisions correctly recovered in the prediction
      (matched nodes connected and forking).
    - **FN**: GT divisions not recovered.
    - **FP**: Spurious predicted divisions, including forks matched to an
      annotated GT node, local-topology rejects, bipartite leftovers, and
      forks whose distinct child branches have nearest matched evidence in
      distinct GT components, and forks with locally merged branches. Fork IDs
      are unioned, so a fork supported by multiple rules counts once.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    DivisionCounts
        Named tuple with ``tp``, ``fn``, and ``fp`` fields.
    """
    result = score_divisions(
        pred_graph,
        gt_graph,
        scale,
        max_distance,
    )
    tp = sum(result.scores.values())
    fn = len(result.scores) - tp
    return DivisionCounts(tp=tp, fn=fn, fp=len(result.fp_forks))
