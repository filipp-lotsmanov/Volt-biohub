"""Post-detection graph repair, ported from the public 0.942-0.948 notebooks.

Why this module exists at all: we reached 0.846 with the *same* pretrained
detector the field uses, against a field median of 0.902. The detector is
therefore not the gap — everything after it is. Reading the actual source of
three notebooks in that band showed they share one ~1500-line post-processing
cell, and their own published ablation attributes most of a heuristic pipeline's
climb to it, not to the model:

    DoG + Hungarian baseline                         0.808
    + prune isolated, retuned threshold              0.827   (+0.019)
    + short-track filter (>= 6 nodes)                0.834   (+0.007)
    + two-pass Hungarian with velocity               0.848   (+0.014)

**The ILP is not implemented here, but it is no longer skipped — see
``run_all``'s *prune_fn*.** This module twice recorded the solver as measured
worthless, and both measurements were of the wrong thing. Running the
reference's own path on the four movies the public notebooks validate on gives
adjusted edge Jaccard **0.9299 with the ILP against 0.8830 without**, spent
almost entirely on precision: edge false positives roughly halve while true
positives move by single digits. What we had measured instead was a solver
handed the *raw* candidate graph, which must then choose the node set too and
deletes anything marginal. ``volt_biohub.ilp_link.prune`` solves the resolved
graph, the way the reference does, and ``run_all`` applies it immediately after
the relink.

Everything operates on the ``(coords_per_t, links_per_t)`` pair the submission
writer already consumes, in *downsampled voxel* coordinates, with distances in
micrometres on the isotropic 1.625 um grid.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

BIG = 1.0e9

# --- motion relink -----------------------------------------------------------
TIGHT_UM = 6.0
RELAXED_UM = 10.0
LAMBDA_V = 0.5
ALPHA_RAW = 0.05
LEARNED_BONUS = 1.0
# --- gap closing -------------------------------------------------------------
GAP_SPAN_UM = 11.6          # total span across one missing frame (5.8 per step)
GAP_REUSE_UM = 3.2
GAP_MAX_FRAC = 0.05
GAP_MAX_ABS = 2000
# --- safe divisions ----------------------------------------------------------
DIV_PARENT_UM = 9.0
DIV_SISTER_UM = 14.0
DIV_EXISTING_UM = 10.0
DIV_DIVERGE_UM = 2.25
DIV_SYMMETRY_TAU = 0.6
DIV_FRAME_FRAC = 0.0076
DIV_GLOBAL_FRAC = 0.00375
# --- filtering and smoothing -------------------------------------------------
MIN_TRACK_LEN = 6
SMOOTH_WINDOW = 2
SMOOTH_WEIGHT = 0.8


def _dist(a, b, iso_um: float) -> float:
    return float(np.linalg.norm(
        (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) * iso_um))


def motion_relink(
    coords_per_t: list[np.ndarray],
    iso_um: float,
    tight_um: float = TIGHT_UM,
    relaxed_um: float = RELAXED_UM,
    lambda_v: float = LAMBDA_V,
    alpha_raw: float = ALPHA_RAW,
    probs_per_pair: list[np.ndarray] | None = None,
    learned_bonus: float = LEARNED_BONUS,
) -> list[list[tuple[int, int, float]]]:
    """Two-pass Hungarian assignment with velocity extrapolation.

    The single largest post-detection item on the public ablation curve (+0.014),
    and it replaces a one-pass fixed-gate assignment — which is what we ship.

    Two things differ from a plain gate. First the cost is measured from where a
    cell is *predicted* to be, not where it was: with a known previous position,
    ``predicted = p + lambda_v * (p - p_prev)``, so a cell moving steadily is
    matched along its own track rather than to whatever happens to be nearest.
    ``alpha_raw`` keeps a small pull toward the raw distance so the extrapolation
    cannot run away on a noisy velocity. Second, the tight pass claims the
    confident pairs first and only the leftovers see the relaxed gate, so a
    genuine long displacement is matched without the loose gate being available
    to everyone — which is the failure a single 8 um gate has.

    **The learned bonus is the third term, and it is how the edge head earns its
    keep here.** The reference's cost is
    ``motion + 0.05*raw - MOTION_RELINK_LEARNED_BONUS * prob``; we shipped only
    the first two. This matters because of what we measured trying the other two
    ways to spend those probabilities: resolved greedily into edges they score
    0.7632 against this linker's 0.7830 (abstention orphans nodes), and handed to
    a global ILP they score 0.8605 against 0.8687 (the solver's decisions land in
    the unannotated 96.5% the metric ignores). As a *bonus on a geometric cost*
    they cannot do either -- geometry still decides the assignment, and the model
    only breaks ties in micrometre-scale units, so a confident pair is pulled
    forward by at most ``learned_bonus`` micrometres of apparent distance.
    """
    out: list[list[tuple[int, int, float]]] = []
    prev_pos: dict[int, np.ndarray] = {}     # index in frame t -> its position at t-1

    for t in range(len(coords_per_t) - 1):
        src, tgt = coords_per_t[t], coords_per_t[t + 1]
        if len(src) == 0 or len(tgt) == 0:
            out.append([])
            prev_pos = {}
            continue

        s = np.asarray(src, dtype=np.float64)
        g = np.asarray(tgt, dtype=np.float64)
        pred = s.copy()
        for i in range(len(s)):
            if i in prev_pos:
                pred[i] = s[i] + lambda_v * (s[i] - prev_pos[i])

        raw = np.linalg.norm((s[:, None, :] - g[None, :, :]) * iso_um, axis=-1)
        motion = np.linalg.norm((pred[:, None, :] - g[None, :, :]) * iso_um, axis=-1)
        cost_base = motion + alpha_raw * raw
        if probs_per_pair is not None and learned_bonus:
            pr = probs_per_pair[t]
            if pr.shape == cost_base.shape:
                cost_base = cost_base - learned_bonus * pr
            elif pr.size:
                # A shape mismatch means detections and probabilities came from
                # different passes, which would silently score the wrong pairs
                # against each other. Louder than a wrong answer.
                raise ValueError(
                    f"probability matrix for frame pair {t} is {pr.shape}, "
                    f"expected {cost_base.shape}"
                )

        links: list[tuple[int, int, float]] = []
        used_s: set[int] = set()
        used_t: set[int] = set()
        for gate in (tight_um, relaxed_um):
            free_s = [i for i in range(len(s)) if i not in used_s]
            free_t = [j for j in range(len(g)) if j not in used_t]
            if not free_s or not free_t:
                break
            sub = cost_base[np.ix_(free_s, free_t)]
            allowed = raw[np.ix_(free_s, free_t)] <= gate
            cost = np.where(allowed, sub, BIG)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] >= BIG:
                    continue
                i, j = free_s[r], free_t[c]
                links.append((i, j, float(raw[i, j])))
                used_s.add(i)
                used_t.add(j)

        out.append(links)
        prev_pos = {j: s[i] for i, j, _ in links}
    return out


def _flatten(coords_per_t, links_per_t):
    """Global-index view: (t array, position array, edge set, per-frame offsets)."""
    offs, pos, ts = [], [], []
    n = 0
    for t, c in enumerate(coords_per_t):
        offs.append(n)
        for p in c:
            pos.append(np.asarray(p, dtype=np.float64))
            ts.append(t)
        n += len(c)
    edges = {(offs[t] + i, offs[t + 1] + j) for t, lk in enumerate(links_per_t)
             for i, j, _ in lk}
    return np.asarray(ts), (np.asarray(pos) if pos else np.zeros((0, 3))), edges, offs


def _unflatten(ts, pos, edges, n_frames):
    """Back to per-frame coords and links, renumbering within each frame."""
    coords_per_t: list[np.ndarray] = []
    local: dict[int, int] = {}
    for t in range(n_frames):
        idx = np.nonzero(ts == t)[0]
        for k, gi in enumerate(idx):
            local[int(gi)] = k
        coords_per_t.append(pos[idx] if len(idx) else np.zeros((0, 3)))
    links_per_t: list[list[tuple[int, int, float]]] = [[] for _ in range(max(n_frames - 1, 0))]
    for a, b in edges:
        t = int(ts[a])
        if t + 1 < n_frames:
            links_per_t[t].append((local[a], local[b], 0.0))
    return coords_per_t, links_per_t


def close_single_frame_gaps(
    coords_per_t: list[np.ndarray],
    links_per_t: list[list[tuple[int, int, float]]],
    iso_um: float,
    span_um: float = GAP_SPAN_UM,
    reuse_um: float = GAP_REUSE_UM,
    max_frac: float = GAP_MAX_FRAC,
    max_abs: int = GAP_MAX_ABS,
):
    """Bridge exactly one missing frame: t -> (t+1 inserted) -> t+2.

    Only one frame, matching the reference — its own ``MAX_GAP=2`` config is
    clamped to 1 in code, and the two-frame case is a separate, far stricter
    pass we are not porting.

    **Reuse an existing detection before inventing one.** An isolated node near
    the midpoint is preferred within *reuse_um*, which repairs the track *and*
    removes a node that isolated-pruning would otherwise delete — a free gain on
    the node-count term. Only when nothing is nearby is a synthetic midpoint
    added, and those are budgeted (5 % of nodes) because every synthetic node
    pays the node-count penalty whether or not it earns an edge.

    Runs *before* the short-track filter on purpose: two 3-node fragments joined
    here survive a length-6 cutoff as one 7-node component, and reversing the
    order would forfeit most of the benefit.
    """
    ts, pos, edges, _ = _flatten(coords_per_t, links_per_t)
    if len(pos) == 0:
        return coords_per_t, links_per_t

    out_deg: dict[int, int] = {}
    in_deg: dict[int, int] = {}
    for a, b in edges:
        out_deg[a] = out_deg.get(a, 0) + 1
        in_deg[b] = in_deg.get(b, 0) + 1

    n_frames = len(coords_per_t)
    budget = min(max_abs, max(1, round(len(pos) * max_frac)))
    added_pos: list[np.ndarray] = []
    added_t: list[int] = []
    new_edges: set[tuple[int, int]] = set()
    used_isolated: set[int] = set()
    n_synth = 0

    for t in range(n_frames - 2):
        ends = [i for i in np.nonzero(ts == t)[0] if out_deg.get(int(i), 0) == 0]
        starts = [j for j in np.nonzero(ts == t + 2)[0] if in_deg.get(int(j), 0) == 0]
        if not ends or not starts:
            continue
        d = np.linalg.norm((pos[ends][:, None, :] - pos[starts][None, :, :]) * iso_um, axis=-1)
        cost = np.where(d <= span_um, d, BIG)
        rows, cols = linear_sum_assignment(cost)

        mid_ids = [int(k) for k in np.nonzero(ts == t + 1)[0]]
        iso_mid = [k for k in mid_ids
                   if out_deg.get(k, 0) == 0 and in_deg.get(k, 0) == 0
                   and k not in used_isolated]
        tree = cKDTree(pos[iso_mid]) if iso_mid else None

        for r, c in zip(rows, cols, strict=True):
            if cost[r, c] >= BIG:
                continue
            a, b = int(ends[r]), int(starts[c])
            mid = 0.5 * (pos[a] + pos[b])

            reused = None
            if tree is not None:
                dd, kk = tree.query(mid)
                if dd * iso_um <= reuse_um and iso_mid[int(kk)] not in used_isolated:
                    reused = iso_mid[int(kk)]
            if reused is not None:
                used_isolated.add(reused)
                new_edges.add((a, reused))
                new_edges.add((reused, b))
            else:
                if n_synth >= budget:
                    continue
                gi = len(pos) + len(added_pos)
                added_pos.append(mid)
                added_t.append(t + 1)
                new_edges.add((a, gi))
                new_edges.add((gi, b))
                n_synth += 1
            out_deg[a] = out_deg.get(a, 0) + 1
            in_deg[b] = in_deg.get(b, 0) + 1

    if added_pos:
        pos = np.concatenate([pos, np.asarray(added_pos)], axis=0)
        ts = np.concatenate([ts, np.asarray(added_t)])
    return _unflatten(ts, pos, edges | new_edges, n_frames)



def _self_rise(node: int, t: int, pos, parents, intensity, lag: int = 2):
    """Peak intensity at *node* relative to its own track *lag* frames back.

    The only fork feature measured to separate. Over all 151 GT divisions
    against 1,729 same-frame controls (``scripts/division_appearance.py``), the
    two-frame rise has **AUC 0.632** -- a dividing parent has *lost* about 10 %
    of its peak (median 0.901) while a non-dividing cell is flat (1.002) --
    against ~0.50 for every geometric feature this project has tried (sister
    separation, symmetry, both daughter distances, all overlapping at the
    quartiles). Absolute brightness does not separate (AUC 0.53-0.59); the
    ratio does, because it cancels the 11x per-sample brightness spread.

    The direction is the opposite of the obvious guess. Condensing chromatin
    ought to brighten a mitotic nucleus; what the measurement shows is the
    parent *dimming*, which is what a mass already splitting into two does to
    the intensity at a single centroid.

    Returns ``None`` when the track does not reach back *lag* frames, which is
    a fifth of candidates -- the caller must treat that as "no evidence",
    never as a score of zero.
    """
    anc = node
    for _ in range(lag):
        ps = parents.get(anc)
        if not ps:
            return None
        anc = ps[0]
    now = intensity(t, pos[node])
    before = intensity(t - lag, pos[anc])
    if before is None or now is None or before <= 0:
        return None
    return float(now / before)


# Diagnostic only: `VOLT_DIV_DUMP=<path>` makes `add_safe_divisions` append one
# row per *admitted candidate* — every fork that survived all four structural
# gates — with its features, its rank under the shipped ordering and whether the
# per-frame cap let it through. This exists because two claims in CLAUDE.md are
# in tension: that the caps bind and out-rank the true forks, and that scaling
# them by 1.25x measured neutral. Both cannot be true, and the pool has never
# been looked at directly. Writing is append-only and costs nothing when unset,
# so it can stay in the tree as the artifact behind whichever claim survives.
_DIV_DUMP_CALL = 0


def _div_reject_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a") as fh:
        if new:
            fh.write("call,t,k_rank,reason,learned_prob,d_par,d_sis,d_exist,"
                     "pz,py,px,cz,cy,cx,hz,hy,hx\n")
        fh.writelines(rows)


def _div_reject_reason(p, ch, cand, pos, iso_um, d_exist, children, claimed,
                       parent_um, sister_um, symmetry_tau, diverge_um,
                       require_grandchildren):
    """The first gate that rejects (p, ch, cand), or "pass".

    Deliberately a re-implementation of the gate order rather than a
    refactor of the live loop: the live loop must stay byte-for-byte the
    shipped behaviour, and a shared helper would put a diagnostic in its
    hot path. The order here mirrors it and the test pins the two agreeing.
    """
    if cand in claimed:
        return "claimed"
    d_par = _dist(pos[p], pos[cand], iso_um)
    d_sis = _dist(pos[ch], pos[cand], iso_um)
    if d_par > parent_um:
        return "d_par"
    if d_sis > sister_um:
        return "d_sis"
    m = 0.5 * (d_exist + d_par)
    if m > 0 and abs(d_exist - d_par) / m > symmetry_tau:
        return "symmetry"
    g1, g2 = children.get(ch, []), children.get(cand, [])
    if len(g1) > 1 or len(g2) > 1:
        return "grandchild_multi"
    if len(g1) == 1 and len(g2) == 1:
        if _dist(pos[g1[0]], pos[g2[0]], iso_um) - d_sis < diverge_um:
            return "diverge"
    elif require_grandchildren:
        return "no_grandchild"
    return "pass"


def _learned_prob(orphan_prob, offs, t, p, cand):
    """P(parent = *p*) for orphan *cand*, or NaN when it was not recorded.

    ``orphan_prob`` is keyed by index *within a frame* -- the space
    ``reference_predict.orphan_parents_per_frame`` hands out -- while this
    module works in ``_flatten``'s global indices, so ``offs`` is the bridge.
    Getting this wrong would read another cell's probability and look entirely
    plausible, which is the failure mode this repo has paid for twice already
    in coordinate code.

    NaN rather than 0.0 when an entry is missing: absent means "outside the
    top-k recorded", which is evidence of a low probability but not a measured
    one, and bucketing unmeasured pairs with confidently-rejected ones would
    bias the very comparison this column exists for.
    """
    if orphan_prob is None or t >= len(orphan_prob):
        return float("nan")
    entries = orphan_prob[t].get(cand - offs[t + 1])
    if not entries:
        return float("nan")
    want = p - offs[t]
    for src, prob in entries:
        if src == want:
            return prob
    return float("nan")


def _div_dump_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a") as fh:
        if new:
            fh.write(
                "call,t,n_props,frame_cap,global_cap,rank,accepted,rank_score,"
                "d_par,d_sis,d_exist,symmetry,diverge,has_grandchildren,cand_k,"
                "pz,py,px,cz,cy,cx,hz,hy,hx\n"
            )
        fh.writelines(rows)


def add_safe_divisions(
    coords_per_t: list[np.ndarray],
    links_per_t: list[list[tuple[int, int, float]]],
    iso_um: float,
    parent_um: float = DIV_PARENT_UM,
    sister_um: float = DIV_SISTER_UM,
    existing_um: float = DIV_EXISTING_UM,
    diverge_um: float = DIV_DIVERGE_UM,
    symmetry_tau: float = DIV_SYMMETRY_TAU,
    frame_frac: float = DIV_FRAME_FRAC,
    global_frac: float = DIV_GLOBAL_FRAC,
    intensity=None,
    rise_weight: float = 0.0,
    require_grandchildren: bool = True,
    n_cand: int = 1,
    orphan_prob: list[dict[int, list[tuple[int, float]]]] | None = None,
):
    """Attach a second daughter, gated structurally rather than by distance alone.

    Our shipped gate was **6.0 um**, below the measured GT median sister
    separation of **10.4 um** (p90 13.0, max 13.7) — so it was rejecting most
    real divisions by construction, and the reference's own history shows
    widening 7->9 um parent and 12->14 um sister taking division TPs from 1 to 4.

    Distance alone is not enough, though: opening a pure distance gate drowns
    real forks in false ones (we measured 6 TP against 390 FP at 10 um). The
    three gates that make a wider window usable are structural rather than
    metric:

    * **mutual nearest orphan** — the candidate must be *the* nearest unclaimed
      orphan to the existing child, which rejects duplicate detections of the
      same cell, the dominant false candidate;
    * **forward divergence** — post-mitotic sisters separate, so both daughters
      must have a successor and the grandchildren must be at least
      *diverge_um* further apart than the sisters are;
    * **symmetry** — a real division splits roughly evenly, so the two
      parent-daughter distances must agree within *symmetry_tau*.

    Caps are per frame and per video because a fork in an annotated region is a
    division false positive, and over-detection is the one place the metric
    taxes it.

    *n_cand* is how many of the nearest orphans are considered, and it is the
    one division knob measured to have headroom. The mutual-nearest rule takes
    the **single** nearest orphan to the existing child, so a spurious
    detection nearer than the true sister makes her unproposable by any gate --
    a selection rule, not a threshold, which is why every gate sweep came back
    empty. Two measurements say widening it is nearly free: the per-frame cap
    holds a median of 1 candidate against a cap of 2 and binds in only 12.8 %
    of frames (`scripts/division_candidate_audit.py`), and only ~1 % of emitted
    forks are ever charged as FP because forks outside annotated regions are
    ignored. So the term is recall-limited, and a candidate that loses is
    usually free while one that wins is a full TP. Each extra candidate still
    has to pass every gate, and `claimed` keeps a daughter from being taken
    twice.
    """
    global _DIV_DUMP_CALL
    dump_path = os.environ.get("VOLT_DIV_DUMP")
    reject_path = os.environ.get("VOLT_DIV_REJECT")
    reject_rows: list[str] = []
    if dump_path or reject_path:
        _DIV_DUMP_CALL += 1

    ts, pos, edges, offs = _flatten(coords_per_t, links_per_t)
    if len(pos) == 0:
        return coords_per_t, links_per_t

    children: dict[int, list[int]] = {}
    parents: dict[int, list[int]] = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
        parents.setdefault(b, []).append(a)

    n_frames = len(coords_per_t)
    global_cap = max(1, round(len(edges) * global_frac))
    added: set[tuple[int, int]] = set()
    claimed: set[int] = set()

    for t in range(n_frames - 1):
        singles = [int(i) for i in np.nonzero(ts == t)[0] if len(children.get(int(i), [])) == 1]
        if not singles:
            continue
        orphans = [int(j) for j in np.nonzero(ts == t + 1)[0]
                   if not parents.get(int(j)) and int(j) not in claimed]
        if not orphans:
            continue
        tree = cKDTree(pos[orphans])
        frame_cap = max(1, round(len(singles) * frame_frac))

        props = []
        dump_feats: dict[tuple[int, int], tuple] = {}
        for p in singles:
            if not parents.get(p):          # a track's first node may not divide
                continue
            ch = children[p][0]
            d_exist = _dist(pos[p], pos[ch], iso_um)
            if d_exist > existing_um:
                continue
            # The orphans nearest the EXISTING child. k=1 is the reference's
            # mutual-nearest rule and stays the default. Above 1 this is a
            # *fallback*, not a multi-propose: the first candidate to clear
            # every gate wins and the rest are dropped, so a parent can still
            # gain at most one second daughter and no node can reach
            # out-degree 3 (which the scorer silently truncates by edge id).
            _dd, kk = tree.query(pos[ch], k=min(n_cand, len(orphans)))
            for cand_k in np.atleast_1d(kk):
                # query pads with len(orphans) once it runs out of neighbours
                if int(cand_k) >= len(orphans):
                    continue
                cand = orphans[int(cand_k)]
                if cand in claimed:
                    continue
                d_par = _dist(pos[p], pos[cand], iso_um)
                d_sis = _dist(pos[ch], pos[cand], iso_um)
                if d_par > parent_um or d_sis > sister_um:
                    continue
                m = 0.5 * (d_exist + d_par)
                if m > 0 and abs(d_exist - d_par) / m > symmetry_tau:
                    continue
                # Forward divergence. The distance half is `diverge_um` and is
                # measured as already optimal (loosening it to 1.0 and -2.0
                # both LOSE, monotonically, divJ 0.0816 -> 0.0691 -> 0.0521 on
                # fold 0). The structural half -- requiring BOTH daughters to
                # have exactly one child -- is the part no knob reached, and it
                # demands two further frames of correct linking on both
                # branches. Absence of a grandchild is not evidence against a
                # division, so `require_grandchildren=False` treats divergence
                # as evidence only when it is available rather than as a
                # precondition. This clause is also what makes the fallback
                # work: a duplicate detection of the existing child sits
                # nearest and has no child of its own, so it fails here and the
                # true sister behind it gets her turn.
                g1, g2 = children.get(ch, []), children.get(cand, [])
                if len(g1) > 1 or len(g2) > 1:
                    continue
                if len(g1) == 1 and len(g2) == 1:
                    if _dist(pos[g1[0]], pos[g2[0]], iso_um) - d_sis < diverge_um:
                        continue
                elif require_grandchildren:
                    continue
                rise_term = 0.0
                if rise_weight and intensity is not None:
                    r = _self_rise(p, t, pos, parents, intensity)
                    if r is not None:
                        rise_term = rise_weight * r
                rank_score = d_par + 0.15 * d_sis + rise_term
                props.append((rank_score, p, cand))
                if dump_path:
                    dz = (_dist(pos[g1[0]], pos[g2[0]], iso_um) - d_sis
                          if len(g1) == 1 and len(g2) == 1 else float("nan"))
                    dump_feats[(p, cand)] = (
                        rank_score, d_par, d_sis, d_exist,
                        abs(d_exist - d_par) / m if m > 0 else 0.0, dz,
                        int(len(g1) == 1 and len(g2) == 1),
                        pos[p], pos[cand], pos[ch], int(cand_k),
                    )
                break

            if reject_path:
                # Every orphan within the sister gate of the existing child,
                # ordered by that distance, with the reason each is refused.
                # `k_rank` is what says whether the true sister was simply
                # further down the queue than `n_cand` reaches.
                near = tree.query_ball_point(pos[ch], sister_um / iso_um)
                near.sort(key=lambda o: _dist(pos[ch], pos[orphans[o]], iso_um))
                for k_rank, o in enumerate(near):
                    c = orphans[o]
                    why = _div_reject_reason(
                        p, ch, c, pos, iso_um, d_exist, children, claimed,
                        parent_um, sister_um, symmetry_tau, diverge_um,
                        require_grandchildren)
                    reject_rows.append(
                        f"{_DIV_DUMP_CALL},{t},{k_rank},{why},"
                        f"{_learned_prob(orphan_prob, offs, t, p, c):.6f},"
                        f"{_dist(pos[p], pos[c], iso_um):.4f},"
                        f"{_dist(pos[ch], pos[c], iso_um):.4f},{d_exist:.4f},"
                        f"{pos[p][0]:.2f},{pos[p][1]:.2f},{pos[p][2]:.2f},"
                        f"{pos[c][0]:.2f},{pos[c][1]:.2f},{pos[c][2]:.2f},"
                        f"{pos[ch][0]:.2f},{pos[ch][1]:.2f},{pos[ch][2]:.2f}\n"
                    )

        ranked = sorted(props)[:frame_cap]
        added_before = set(added)

        for _s, p, cand in ranked:
            if len(added) >= global_cap or cand in claimed:
                continue
            added.add((p, cand))
            claimed.add(cand)
            children[p].append(cand)
            parents.setdefault(cand, []).append(p)

        if dump_path:
            accepted_now = added - added_before
            rank_of = {(q, c): r for r, (_s, q, c) in enumerate(sorted(props))}
            rows = []
            for (q, c), f in dump_feats.items():
                sc, dp, ds, de, sym, dz, hasg, pp, pc, ph, ck = f
                rows.append(
                    f"{_DIV_DUMP_CALL},{t},{len(props)},{frame_cap},{global_cap},"
                    f"{rank_of[(q, c)]},{int((q, c) in accepted_now)},{sc:.4f},"
                    f"{dp:.4f},{ds:.4f},{de:.4f},{sym:.4f},{dz:.4f},{hasg},{ck},"
                    f"{pp[0]:.2f},{pp[1]:.2f},{pp[2]:.2f},"
                    f"{pc[0]:.2f},{pc[1]:.2f},{pc[2]:.2f},"
                    f"{ph[0]:.2f},{ph[1]:.2f},{ph[2]:.2f}\n"
                )
            _div_dump_rows(dump_path, rows)

    if reject_path and reject_rows:
        _div_reject_rows(reject_path, reject_rows)

    return _unflatten(ts, pos, edges | added, n_frames)


def filter_short_tracks(
    coords_per_t: list[np.ndarray],
    links_per_t: list[list[tuple[int, int, float]]],
    min_len: int = MIN_TRACK_LEN,
    keep_divisions: bool = True,
):
    """Drop weakly connected components smaller than *min_len* NODES.

    Nodes, not frames: a fork makes parent and both daughters one component, so
    a division inflates the count — which is deliberate, since components
    containing a fork are exempt at any size. A division true positive requires
    the whole matched lineage to sit in **one** component, so deleting a short
    fragment that happens to hold a fork would cost the 0.1 division term.

    No volume-boundary exemption, matching the reference. Safety valve: if the
    keep set comes back empty, nothing is removed — better a bad graph than an
    absent dataset.
    """
    ts, pos, edges, _ = _flatten(coords_per_t, links_per_t)
    n = len(pos)
    if n == 0:
        return coords_per_t, links_per_t

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comp: dict[int, list[int]] = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)

    out_deg: dict[int, int] = {}
    for a, _b in edges:
        out_deg[a] = out_deg.get(a, 0) + 1

    keep: set[int] = set()
    for members in comp.values():
        has_fork = keep_divisions and any(out_deg.get(m, 0) >= 2 for m in members)
        if len(members) >= min_len or has_fork:
            keep.update(members)
    if not keep:
        return coords_per_t, links_per_t

    order = sorted(keep)
    remap = {g: k for k, g in enumerate(order)}
    ts2 = ts[order]
    pos2 = pos[order]
    edges2 = {(remap[a], remap[b]) for a, b in edges if a in keep and b in keep}
    return _unflatten(ts2, pos2, edges2, len(coords_per_t))


def linefit_smooth(
    coords_per_t: list[np.ndarray],
    links_per_t: list[list[tuple[int, int, float]]],
    window: int = SMOOTH_WINDOW,
    weight: float = SMOOTH_WEIGHT,
):
    """Blend each interior node toward a linear fit of its own neighbourhood.

    Runs last, after nodes have been removed, so the neighbourhoods reflect the
    final topology. Only nodes with an unambiguous chain on both sides move: the
    walk stops the moment a node does not have exactly one predecessor or
    successor, which leaves track ends, fork parents and daughters untouched —
    exactly the places a straight-line prior would be wrong.

    All updates are computed from the original positions and applied in bulk, so
    a smoothed neighbour cannot feed into its neighbour's fit.
    """
    ts, pos, edges, _ = _flatten(coords_per_t, links_per_t)
    if len(pos) == 0:
        return coords_per_t, links_per_t

    succ: dict[int, list[int]] = {}
    pred: dict[int, list[int]] = {}
    for a, b in edges:
        succ.setdefault(a, []).append(b)
        pred.setdefault(b, []).append(a)

    new = pos.copy()
    for i in range(len(pos)):
        pts = [(0, i)]
        cur = i
        for k in range(1, window + 1):
            p = pred.get(cur, [])
            if len(p) != 1:
                break
            cur = p[0]
            pts.append((-k, cur))
        cur = i
        for k in range(1, window + 1):
            s = succ.get(cur, [])
            if len(s) != 1:
                break
            cur = s[0]
            pts.append((k, cur))
        if len(pts) < 3:
            continue
        dts = np.asarray([d for d, _ in pts], dtype=np.float64)
        coords = pos[[j for _, j in pts]]
        fitted = np.asarray([np.polyval(np.polyfit(dts, coords[:, a], 1), 0.0)
                             for a in range(3)])
        new[i] = (1.0 - weight) * pos[i] + weight * fitted

    return _unflatten(ts, new, edges, len(coords_per_t))


def _enforce_structure(
    links_per_t: list[list[tuple[int, int, float]]],
) -> list[list[tuple[int, int, float]]]:
    """Cap in-degree at 1 and out-degree at 2, dropping the longest links first.

    Gap closing and the division pass each add edges under their own local
    guard, so neither can see that the *other* has already claimed a node. The
    scorer does not error on a merge: it silently collapses the duplicates and
    keeps the lowest edge id, so a merge is a free FP that never shows up as a
    failure. Enforce it here instead of trusting insertion order.
    """
    out: list[list[tuple[int, int, float]]] = []
    for links in links_per_t:
        by_target: dict[int, tuple[int, int, float]] = {}
        for a, b, w in sorted(links, key=lambda e: e[2]):
            if b not in by_target:
                by_target[b] = (a, b, w)
        kept: list[tuple[int, int, float]] = []
        out_deg: dict[int, int] = {}
        for a, b, w in sorted(by_target.values(), key=lambda e: e[2]):
            if out_deg.get(a, 0) >= 2:
                continue
            out_deg[a] = out_deg.get(a, 0) + 1
            kept.append((a, b, w))
        out.append(kept)
    return out


def _clamp_to_grid(
    coords_per_t: list[np.ndarray], shape: tuple[int, int, int]
) -> list[np.ndarray]:
    """Keep every position inside the volume it was detected in.

    Smoothing fits a line through a node's neighbourhood, so a node on the
    border can be pulled marginally outside the grid -- a z/y/x of -0.3 becomes
    -1 once the submission writer multiplies by the downsample factor and
    rounds, and the validator rejects the file. Measured: 11 such nodes in two
    videos. Clamping rather than dropping, because the node itself is real and
    its edges are almost certainly right; only the sub-voxel overhang is wrong,
    and 7 um matching makes that overhang worthless anyway.
    """
    hi = np.asarray(shape, dtype=np.float64) - 1.0
    return [np.clip(c, 0.0, hi) if len(c) else c for c in coords_per_t]


def run_all(
    coords_per_t: list[np.ndarray],
    iso_um: float,
    *,
    divisions: bool = True,
    min_track_len: int = MIN_TRACK_LEN,
    smooth: bool = True,
    shape: tuple[int, int, int] | None = None,
    div_kwargs: dict | None = None,
    links_per_t: list[list[tuple[int, int, float]]] | None = None,
    gaps: bool = True,
    relink_kwargs: dict | None = None,
    gap_kwargs: dict | None = None,
    smooth_kwargs: dict | None = None,
    prune_fn=None,
):
    """The reference's ordering, which is load-bearing — see each step's docstring.

    relink -> gap close -> divisions -> short-track filter -> smooth. Gap closing
    must precede the length filter or repaired fragments are deleted before they
    can be joined; divisions must precede it too so the fork exemption protects
    the component they create; smoothing runs last so its neighbourhoods reflect
    the final graph.

    **Pass *links_per_t* to skip the relink entirely**, using the topology a
    better linker already produced. This is not a convenience: the reference's
    own clean-split ablation prices its two-pass Hungarian at **-0.011**, because
    once a global solver has chosen the edges, re-deriving them from geometry
    overwrites good ones (and, in their measurement, steals division daughters —
    the division term rose 0.059 -> 0.114 when it was removed). The relink earned
    its +0.076 here only because it replaced a naive one-pass fixed-gate linker;
    against the ILP it is expected to be the harmful pass, and this parameter is
    how that gets measured rather than assumed.

    *gaps* likewise switches off single-frame gap closing, which the same
    ablation prices at +0.0004 — i.e. nothing — once the solver is choosing
    topology.
    """
    links = (motion_relink(coords_per_t, iso_um, **(relink_kwargs or {}))
             if links_per_t is None else links_per_t)
    coords = coords_per_t
    if prune_fn is not None:
        # An integer program over the *resolved* link set, before anything adds
        # edges back. Measured on the reference's own path: solving the graph
        # its greedy edge head already resolved lifts adjusted edge Jaccard
        # 0.8830 -> 0.9299 by roughly halving edge false positives, while
        # solving the raw candidate graph instead costs 0.05 (see
        # volt_biohub.ilp_link). It runs here rather than at the end because
        # gap closing and the division pass add edges under their own local
        # guards, and a solver that saw those would be asked to re-judge
        # decisions made on evidence it does not have.
        coords, links = prune_fn(coords, links)
    if gaps:
        coords, links = close_single_frame_gaps(coords_per_t, links, iso_um,
                                               **(gap_kwargs or {}))
    if divisions:
        coords, links = add_safe_divisions(coords, links, iso_um,
                                           **(div_kwargs or {}))
    coords, links = filter_short_tracks(coords, links, min_len=min_track_len)
    if smooth:
        coords, links = linefit_smooth(coords, links, **(smooth_kwargs or {}))
    links = _enforce_structure(links)
    if shape is not None:
        coords = _clamp_to_grid(coords, shape)
    return coords, links
