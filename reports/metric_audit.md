# Scorer audit — what the metric rewards that we are not doing, and what it punishes silently

Line-by-line audit of `src/volt_biohub/metrics.py`, `src/volt_biohub/division_metrics.py`,
`metrics.md`, and the `tracksdata` matching they delegate to. Every claim carries a
`file:line`; every claim marked **measured** was settled by running the real scorer on a
purpose-built fixture (probe scripts under `/tmp/audit/` and, for the `r*` probes added in
review, `/tmp/review/` — neither committed; they belong in `tests/` if any of this becomes
a regression guard).

Independently re-verified in adversarial review: every `file:line`, the §2 closed form
(exact to 7e-18), and the measured results of §0, §1, §2b, §3, §3a–§3d, §4, §4a, §6 and §7
all reproduce. Four claims did not survive and are corrected in place: §1's monotonicity
*proof* (a counterexample now sits in that section), §2a's cost term (it charged FPs that
the metric never charges), §3b's cost table (it assumed `Σ FP = 0`), and §6's "harmless"
(the arithmetic is harmless, the sweep consequence is not).

Classification: **(a)** exploit we are not using, **(b)** trap that silently costs points,
**(c)** already in CLAUDE.md.

Nothing in `metrics.py` / `division_metrics.py` was modified.

---

## 0. First: the claim we just built a decision on

> *an edge is scored only when an endpoint matched a GT node WITH GT edges (so
> unmatched-region FPs are free); this makes node recall a GATE on FP exposure.*

**The first half is exactly right. The second half is right in mechanism but the word
"gate" is misleading, and there is one refinement CLAUDE.md does not state.**

Code path: `metrics.py:157-167` builds per-GT-node `out_valid = out_degree > 0` and
`in_valid = in_degree > 0`; `:170-178` joins them onto the *matched* GT node of each
predicted node, filling `False` for unmatched (`MATCHED_NODE_ID == -1` joins to nothing);
`:181-195` attaches `out_valid` **by edge source** and `in_valid` **by edge target** and
sets `pred_valid = out_valid(source) | in_valid(target)`. Only `pred_valid` edges enter
the denominator (`:209`, `:315-316`). So:

```
charged(edge)  <=>  out_degree_GT(match(source)) > 0   OR   in_degree_GT(match(target)) > 0
```

**Measured** (probe `p1b`, GT chain t=1..3, four probe edges):

| predicted edge | out_valid(src) | in_valid(tgt) | charged |
|---|---|---|---|
| unmatched → matched to GT track **start** | false | false | **no** |
| matched to GT track **end** → unmatched | false | false | **no** |
| matched to GT mid → unmatched | true | false | yes |
| unmatched → matched to GT mid | false | true | yes |

**(b) Refinement CLAUDE.md is missing:** the *temporal endpoints* of every annotated
track are free attachment points. An edge leaving the last GT node of a track, or
entering the first, is discarded even though it lands squarely on a matched, annotated
cell. Measured over all 199 GT files: **4,586 / 133,318 GT nodes (3.4 %) have
`out_degree == 0`** and 4,435 (3.3 %) have `in_degree == 0` (per-sample out-degree-0
fraction p5 1.1 %, p50 2.8 %, p95 5.8 %, max 8.4 %; probe `p6`). So the free zone is real
but small — the simplification in CLAUDE.md is safe to keep reasoning with, and this is a
correctness footnote, not a lever.

**On "node recall is a gate":** matching a GT node is what *creates* FP exposure, so
higher node recall means strictly more of our edges become chargeable. That is not a gate
in the sense of a precondition we can turn off — we need the match to earn the TP in the
first place. The accurate statement is: **FP exposure is created by the same matches that
create TP opportunity, one endpoint at a time, and the exchange rate between them is
computed in §2.** Do not read "recall is a gate" as "raise recall and FPs stay free".

**(b) And node recall is a broken diagnostic for this.** Measured (probe `p3`): a
prediction with `node_recall = 1.00` scored **edge Jaccard 0.000**. `summarise`
additionally reports node recall as an *unweighted* mean over samples (`metrics.py:532`)
while the score is edge-weighted — the two numbers are not on the same footing. Do not
use node recall to sanity-check a run.

---

## 1. (a) EXPLOIT — a predicted node that touches no edge pays the node penalty and can steal a match

`num_pred_nodes` is `graph.num_nodes()` (`metrics.py:330`), taken from the **raw**
predicted graph: before matching, before the `dt == 1` filter, before merge dedup, before
the out-degree cap. It feeds `total_node_ratio` at `metrics.py:440`. A node that appears
in no edge therefore contributes to the node penalty and **cannot** contribute to any
numerator — `pred_valid` is a property of edges (`metrics.py:193-195`).

Worse, it actively destroys score, because it competes in the node assignment.

**Measured** (probe `p3`; GT = 3-node chain; prediction = a correctly linked chain
1 µm off the GT centroids, plus one edge-less detection sitting exactly on each GT
centroid):

| prediction | TP/FP/FN | N_pred | J | J_adj | node_recall |
|---|---|---|---|---|---|
| with the 3 edge-less detections | **0/0/2** | 6 | 0.000 | 0.000 | 1.00 |
| edge-less detections pruned | **2/0/0** | 3 | 1.000 | 1.050 | 1.00 |

The isolated node wins the bipartite assignment (it is closer), so the *linked* node is
left unmatched and its correct edge is not merely un-credited — it is not even chargeable.
Score goes from 1.05 to 0.

**This is live in our pipeline.** `build_graph` in
`scripts/predict_unet_transformer.py:132-141` adds **every** detection coordinate as a
node and only then adds whatever edges the linker produced; `csv_to_geffs.py:25-32` does
the same from the CSV. With `--det-threshold 0.99` leaving ~6,000 voxels/frame above
threshold (CLAUDE.md, Status) and a `softmax`-over-sources linker keeping only pairs above
0.5, a large fraction of emitted nodes are certainly edge-less.

**Fix:** drop every node with in-degree + out-degree == 0 before writing the geff/CSV.
One filter in `build_graph`. The multiplier half is monotone — dropping nodes can only
raise `m_i`. **The assignment half is not, and an earlier draft of this section wrongly
claimed it was.** Freeing a GT node lets a *linked* detection match it, which makes that
detection's edges chargeable — a gain only if those edges are right. When they are wrong,
the edge-less node was shielding us. **Measured** counterexample (probe `r2`, two annotated
2-node tracks, one linked correctly and one linked to a target 50 µm off, `N_true` 100):

| prediction | TP/FP/FN | N_pred | J_adj |
|---|---|---|---|
| edge-less node pruned | 1/1/1 | 4 | 0.3653 |
| edge-less node kept, shielding the bad link | 1/0/1 | 5 | **0.5475** |

That is §0's own rule read forwards: matching a GT node creates FP exposure. So the filter
is a *near-certainly* positive change, not a provable one, and its size depends on the
linker's precision **on detections that match GT** — which at ~6,000 voxels/frame over
threshold is unmeasured. It needs a CV number before it is believed, not a proof.

**Expected gain:** two components. (i) the node-count multiplier —
`1 + 0.1·(1 − N_pred/N_true)`, so removing 10 % of `N_true` worth of nodes buys **+1 %
relative on the entire edge term**. This part is guaranteed. (ii) the recovered
assignments, potentially much larger since each stolen match costs two edges, but signed
by the linker's precision on matched detections (see the counterexample above) and
therefore an empirical question.

### 1b. (a) The same arithmetic, generalised: the node budget's real exchange rate

From the closed form in §2, deleting one unmatchable detection from sample *i* is worth
`0.1 · TP_i / N_true_i` edge-TPs. **Measured** over the 199 train samples (probe `p17`,
using `n_edges` as the TP ceiling): per 1,000 unmatchable detections removed you buy
p5 0.2, **p50 3.5**, p95 16.1 TP-equivalents. Meanwhile a detection that *does* match is
worth ~2 TPs, i.e. **573× more than an unmatchable one at the median**.

That reconciles the two directions CLAUDE.md argues at once: keep every detection that
might match a GT cell (each is worth ~573 junk detections), delete every detection that
cannot. It is not "the asymmetry favours over-detection" — it is "the asymmetry favours
*placement*, and is indifferent-to-negative on *count*".

### 1c. (a) The metric does not ask us to detect all the cells

Nothing in the numerator or the FP rule references an unannotated cell
(`metrics.py:193-195`, `:315-317`). The *only* term tying `N_pred` to the true cell count
is the multiplier — and it pays a **bonus** for under-predicting, capped at +10 %
(`metrics.py:446-449`, ratio bottoms out at −1). GT annotates ~3.6 % of cells at the
median. So the score-maximal prediction is **the annotated subset alone**, at ~4 % of
`N_true`, earning multiplier ≈ 1.096 *and* full recall.

We cannot identify that subset, which is why this is a reframing rather than a recipe.
But it has a testable consequence and a warning:

- The optimal `N_pred` is **not** `N_true`; it is as far below `N_true` as we can go
  without losing GT-node recall. At our current over-detected operating point that
  direction is down, and the first ~90 % of the available reduction is free (§1b).
- CLAUDE.md records a possible handle on which cells are annotated: GT centroids have
  normalised intensity p50 0.479 while non-GT **local maxima** have p50 1.000. Note what
  is and is not measured there — the comparison set is local maxima, which are bright by
  construction, so the finding supports "GT centroids are dimmer than non-GT local maxima",
  not "annotated cells are dimmer than unannotated cells". Preferring dimmer peaks would
  raise TP-per-node *if* the stronger reading holds. **Treat this as doubly suspect**: it
  is an artifact of the annotation *process*, not of biology, and
  betting on it means betting that the third embryo was annotated the same way. If tried
  at all, it must be validated on LOEO and never on the leaderboard.

---

## 2. (a) The weighted aggregation collapses to a global micro-average — and it hands us the optimal edge threshold in closed form

`summarise` weight-averages the per-sample adjusted Jaccard by `w_i = TP_i + FP_i + FN_i`
(`metrics.py:498-506`). Since `J_i = TP_i / w_i` by construction (`:444-445`), the weight
cancels:

```
w_i · J_adj_i  =  TP_i · max(0, 1 − 0.1·r_i)

adj_edge_jaccard  =  Σ_i TP_i · m_i   /   Σ_i (E_i + FP_i)          m_i = max(0, 1 − 0.1·r_i)
```

where `E_i` is sample *i*'s GT edge count (`FN_i = E_i − TP_i`, `metrics.py:317`).
**Measured exact** to 1e-12 over 40 randomised synthetic samples (probe `p5`).

This is the single most useful thing in this audit, because it kills a whole class of
wrong intuition:

- **A TP's value does not depend on how big its sample is.** Marginal value `= m_i / Σw`,
  independent of `w_i`. It does still depend on `m_i`, and §2b shows `m_i` reaches **0** on
  18–47 samples at a dense operating point — a TP in a clamped sample is worth exactly
  nothing. "44b6 carries only 15 % of the metric weight" is true as a *ceiling* on
  what that embryo can contribute and as a statement about fold variance — it is **false**
  if read as "an edge fixed in 44b6 counts less". It does not.
- **A charged FP costs the same wherever it is:** `adj_J / Σw`. So one TP is worth
  exactly `m_i / adj_J` charged FPs — **invariant to how many FPs we already have**
  (probe `p14`): 5.0 FPs at adj_J 0.20, **3.3 at 0.30**, 2.0 at 0.50.
- **The per-sample objective is separable:** `TP_i·m_i − adj_J·FP_i`. A per-sample
  detection threshold can therefore be tuned against a per-sample objective instead of
  against the global score, which is a much cheaper sweep.
- It puts `scripts/check_t_true.py:104-118`'s `n_edges` weighting on a firm footing: the
  exact numerator weight is `TP_i`, so `E_i` is the right shape and a well-motivated upper
  bound on it (the script's own comment already calls `n_edges` a lower bound on `w_i`).
  It remains a proxy, not an identity — samples we score badly on matter *less* for the
  node budget than `E_i`-weighting implies.

### 2a. (a) The metric-optimal edge threshold is ≈ 0.2, not the hardcoded 0.5

Keeping one extra edge that has probability *p* of being correct: gain `p·m_i/Σw`, cost
`(1−p)·adj_J/Σw`. Keep it iff

```
p  >  adj_J / (m_i + adj_J)
```

At `m_i ≈ 1`: **0.17 at adj_J 0.20, 0.23 at 0.30, 0.33 at 0.50.** The hardcoded
`threshold: float = 0.5` (`predict_unet_transformer.py:76`, applied at `:467`) is roughly
2× too strict at any plausible operating point. Note the rule is self-consistent — the
threshold is a function of the score it produces, so it is a fixed point to iterate, not
a constant. Note also the coherence: on a sample where we blew the node budget (`m_i`
small) the threshold *rises*, because edges there are worth less.

**Three corrections apply before acting on it, and the first one moves the number a
long way.**

1. **The cost term above charges *every* wrong edge, which contradicts §0.** A wrong edge
   costs `adj_J/Σw` only when it is `pred_valid`. Writing `q = P(charged | wrong)` the rule
   is really `p > q·adj_J/(m_i + q·adj_J)`, and GT annotates ~3.6 % of cells, so most wrong
   edges have neither endpoint matched and `q ≪ 1`. **Measured** (probe `r3`, one annotated
   3-node track): adding **200** wrong edges in the unannotated region left `TP/FP/FN` at
   `2/0/0` and `w` at 2 — literally zero cost — while wrong edges leaving the *annotated*
   cell were charged. So ≈0.2 is itself an upper bound on the optimum; for the edge term
   alone the threshold wants to go as low as the next two corrections allow.
2. The ≤2-children cap (§4). Note this is also what bounds the damage: the same probe put
   200 edges on one matched source and the cap admitted exactly **one** extra FP.
3. The division-FP interaction (§3b) — the real binding constraint, since forks are read
   off the raw graph.

And one blocker that is not a correction but a prerequisite: **the quantity we threshold is
not a calibrated probability.** `predict` takes `softmax(logits, dim=0)` over sources,
whose denominator scales with the detection count (CLAUDE.md, Known defects: 753 → 34
candidate sources across `--det-threshold` 0.3 → 0.999). The formula gives the right target
and the right direction; applying a literal 0.2 to an uncalibrated softmax measures the
softmax artifact, exactly as CLAUDE.md warns. There is also **no measured `adj_J` for this
project yet**, so every number in this subsection is conditional on an operating point we
have not established.

### 2b. (b) TRAP — the per-sample multiplier clamps to zero at `N_pred ≥ 11·N_true`

`max(0, ...)` at `metrics.py:447-449` is a **per-sample** clamp. A clamped sample
contributes 0 to the numerator but still contributes `E_i + FP_i` to the global
denominator, so it is worse than a sample we simply did not predict.

`N_true` spans 3,783 → 78,644, so the clamp sits at **416 detections/frame on the sparsest
sample** and 8,651/frame on the densest — a 21× spread. **Measured** (probe `p7`, edge-weighted):

| single global det/frame | edge-weighted multiplier | samples zeroed | share of GT edges zeroed |
|---|---|---|---|
| 100 | 1.008 | 0 | 0 % |
| 179 (median true density) | 0.936 | 0 | 0 % |
| 400 | 0.734 | 0 | 0 % |
| **600** | 0.564 | **18** | **8.5 %** |
| 800 | 0.454 | 47 | 26.7 % |

The densest sample genuinely holds 786 cells/frame. **A detector tuned to cover the dense
samples annihilates the sparse ones.** This is an independent argument for the per-sample
`N_true` prediction CLAUDE.md already plans, and it sets a hard floor: whatever the sweep
decides, per-sample `N_pred` must stay under `11·N̂_true` with margin, because the
prediction itself carries 25 % median error.

---

## 3. (b) TRAP — the division metric reads the RAW predicted graph, so every edge the edge metric silently deletes manufactures a division false positive

`score_divisions` takes forks straight off `pred_graph` (`division_metrics.py:354-357`,
`out_degree(node_id) >= 2`), and `_pred_division_fork_sets` off the matched copy
(`:465-468`). **Neither applies the `dt == 1` filter (`metrics.py:71-87`), the duplicate-edge
dedup (`:56-66`), the merge dedup (`:89-135`) or the out-degree cap (`:140-153`).**

**Measured** (probe `p2`; GT = a straight annotated 4-node chain, zero GT divisions):

| prediction | edge TP/FP/FN | division TP/FP/FN |
|---|---|---|
| clean chain | 3/0/0 | 0/0/0 |
| chain + one `t → t+2` "gap closing" edge | 3/0/0 | **0/1/0** |
| chain + one backward `t → t−1` edge | 3/0/0 | **0/1/0** |
| chain + the *same* edge emitted twice | 3/0/0 | **0/1/0** |

CLAUDE.md says these edges are "deleted" and "dropped". They are — from the edge term
only. On the division term each one is a fork, and a fork matched to an annotated GT node
is an FP (`division_metrics.py:469-472`). This is a genuinely new finding and it changes
the guidance: a `t → t+2` edge is not harmlessly ignored, it is **worse than omitting the
edge**, and a duplicated edge row in the CSV is not a no-op.

With 151 GT divisions in the entire training set, the division denominator is tiny and
every such FP is expensive. **Add all four to `validate_submission()`**: no `dt != 1`
edges, no duplicate `(source, target)` pairs, no backward edges, out-degree ≤ 2.

### 3a. (b) TRAP — a fork in a completely unannotated region is *not* always free

`_pred_division_fork_sets:474-490` marks a fork FP when two of its **distinct direct-child
branches** have matched evidence in two different GT *weakly connected components*
(`_gt_weak_component_ids:393-407`), and `_branch_component_evidence:424,430` marks it FP
("malformed") when a child or grandchild used by the fork has any parent other than that
fork. Neither rule requires the fork itself to match anything.

**Measured** (probe `p10`, two annotated lineages 12 µm apart, zero GT divisions):

| prediction | division TP/FP/FN |
|---|---|
| fork **unmatched**, its two children land on two different GT lineages | **0/1/0** |
| spurious fork 100 µm from all GT | 0/0/0 |
| spurious fork whose own node matches an annotated GT node | 0/1/0 |

Since GT is sparse, it has many weak components, and the median cell pitch is 18.5 µm
(min 11.3) — well inside the reach of a fork with two children. So **an identity swap or a
spurious fork that bridges two annotated tracks is taxed even though it is "in the
unannotated region"**. CLAUDE.md's "forks in unannotated regions are ignored" is true only
for forks with no matched descendants.

### 3b. (b) TRAP — hedging extra parents wrecks divisions, and one wrecked division costs 20–150 edge TPs

`_branch_component_evidence:424` requires `set(predecessors(child)) == {fork}`. Give any
daughter of a predicted fork a second parent and the fork becomes malformed: it loses its
TP **and** becomes an FP.

**Measured** (probe `p9`, one GT division): perfect prediction scores division 1/0/0;
adding one hedged second parent onto one daughter scores **0/1/1** — a two-step swing in
the denominator plus the lost TP, for +1 edge FP.

Cost, computed from the closed form with 151 GT divisions, **division TP 75 / FN 76**
(i.e. an assumed 50 % division recall — every row needs it to reproduce) and
**Σw = 128,883** (probe `p14`):

| division FP total | division TP / FN | cost of wrecking one division, in edge TPs |
|---|---|---|
| 0 | 75 / 76 | **127** |
| 50 | 75 / 76 | 88 |
| 150 | 75 / 76 | 53 |
| 500 | 75 / 76 | 22 |

**These are a lower bound, and by a factor of ~2 at §2a's own operating point.** `Σw =
Σ E_i + Σ FP_i` and `Σ E_i = 128,883` exactly, so setting `Σw = 128,883` asserts
`Σ FP_i = 0` — a perfect-precision linker, which back-solves to `adj_J ≈ 0.47` at 50 %
edge recall. §2 and §2a instead assume `adj_J` 0.20–0.30, and §2a names *this* subsection
as the correction to apply, so the two are on inconsistent footings. Rescaling (probe
`r4`): the cost of wrecking one division at division FP 0 is **297** edge TPs at
`adj_J` 0.20 and **198** at 0.30, not 127. The direction only strengthens the conclusion —
protecting divisions is worth more than the table says — but do not mix the two numbers
when doing the trade-off arithmetic.

So the §2a hedging rule must carry an exception: **never create in-degree > 1 on a node
within one frame of a predicted fork.** Away from forks, hedging costs nothing on the
division term (measured: in-degree 2 and in-degree 4 both gave `divFP = 0`, probe `p8`).

### 3c. (b) TRAP — hedging the division *timepoint* is strictly negative

`considered` (`division_metrics.py:376`) collects every local fork near a GT division, the
bipartite pairing (`:383-385`) awards at most one, and `fp_forks` is
`(considered | evaluable | invalid) − tp_forks` (`:389`). So a second, redundant fork one
frame away becomes an FP. **Measured** (probe `p15`): a single correct fork scores 1/0/0;
the same plus a plausible extra fork one frame earlier scores **1/1/0**. Emit exactly one
fork per suspected division.

### 3d. (a) The division window's parent anchor is more forgiving than metrics.md reads

`_matched_division_nodes:216-224` builds the parent side as `{divider} ∪
predecessors(divider)`, and `_is_strongly_connected_division:260-261` accepts the fork or
any of its immediate predecessors as the anchor. **Measured** (probe `p9`): displacing the
fork node 30 µm so it matches nothing **still scores a division TP** (1/0/0), because the
*grandparent* matched. Move both off and it becomes a clean FN with no FP (0/0/1).

So a division survives a badly localised or missing parent detection provided the
grandparent is matched and the local topology holds. Combined with the ±1-frame tolerance
(measured: fork one frame late 1/0/0, one frame early 1/0/0), the division rule is
generous about *when* and *where* the fork is, and strict about the two daughter branches
being clean and unmerged. Effort on divisions belongs in daughter-branch hygiene, not in
pinpointing the split frame.

---

## 4. (b) TRAP — the out-degree cap is decided by CSV row order

`metrics.py:140-153` ranks outgoing edges by `EDGE_ID` and keeps the two lowest;
`:89-135` keeps the lowest `EDGE_ID` among merge duplicates. Edge IDs are assigned by
insertion order — and for a submission that order is **the order of edge rows in
`submission.csv`**, via `csv_to_geffs.py:36-43` (`bulk_add_edges` over the CSV rows in
file order).

**Measured** (probe `p8`, a node with three outgoing edges, one correct):

| CSV order | edge TP/FP/FN | division FP |
|---|---|---|
| correct edge last | **3/2/1** | 1 |
| correct edge first | **4/1/0** | 1 |

Two consequences. (i) It is a **free, one-line exploit**: emit each source's outgoing
edges in descending confidence order, so that if anything slips past our own cap the
scorer keeps the best two. `predict_unet_transformer.py:462-470` already sorts candidates
by probability descending, so preserving that order through the writer is enough — but
nothing currently *guarantees* it, and a `sort` by dataset/t in the writer would silently
break it. (ii) Note the division FP is 1 in **both** rows: out-degree 3 is a fork on the
raw graph regardless of truncation (§3), so enforcing ≤2 ourselves is not optional. The
greedy linker already does (`PredictConfig.__post_init__` sets `max_children_per_node = 2`),
which makes (i) genuinely just insurance — **but `--use-ilp` sets both caps to `None` on
purpose**, so the ILP path is the one place where whatever out-degree the solver emits
reaches the scorer, and every out-degree-3 node there is a free division FP.

### 4a. (a) In-degree is not capped by the scorer at all — and our linker caps it at 1

The scorer constrains out-degree (≤2, `:140-153`) and nothing else. In-degree is
unconstrained. **Measured** (probe `p8`): four parent edges onto one target gave
TP 3 / FP 3 — each hedge charged exactly one FP, no truncation, no division penalty.

Meanwhile `PredictConfig.__post_init__` sets `max_parents_per_node = 1`
(`predict_unet_transformer.py:92-94`) and the training loss normalises `softmax(dim=0)`
over sources, i.e. one parent per child (CLAUDE.md, "The pipeline as it stands"). **Our
pipeline enforces exactly the constraint the metric does not impose, and the metric's
own constraint — ≤2 children — is the one we should be enforcing hard.** Combined with
§2a: for an ambiguous target, emit every parent candidate above `adj_J/(m+adj_J) ≈ 0.2`,
subject to out-degree ≤2 per source (a small confidence-ordered constrained assignment,
not a free-for-all) and to the §3b no-hedging-near-forks rule.

---

## 5. Node matching — tie-breaks, duplicates, and what the assignment actually optimises

The assignment is per-timepoint (`tracksdata/metrics/_ctc_metrics.py:174-177` groups by
`T`; `_match_single_frame:66-71` returns nothing when a timepoint is missing on either
side), so **a node at the wrong `t` never matches**. Weights are `1/(1+d)`
(`_matching.py:296`) over pairs within `max_distance` only (`:282`), solved by
`min_weight_full_bipartite_matching(..., maximize=True)` with a `-1`-fill fallback and a
final dense `linear_sum_assignment` (`_ctc_metrics.py:84-103`). Net effect: **maximum
cardinality first, then maximum Σ1/(1+d)** — a convex, not linear, function of distance,
so it prefers "one exact hit plus one marginal" over "two mediocre" pairings.

**(b) Duplicate predicted nodes at identical coordinates are a coin flip you do not
control.** Measured (probe `p4`, GT chain, exact-duplicate detection at every GT
centroid, only one copy linked):

| geometry | TP/FP/FN |
|---|---|
| linked copy inserted **first** (lower node id) | 2/0/0 |
| linked copy inserted **second** | **0/0/2** |
| unlinked copy 0.1 µm **closer** than the linked one | **0/0/2** |

On an exact tie the lower node id won, reproducibly across three runs — but that is
scipy's column order, an implementation detail, not a contract. The third row is the one
that matters: the assignment is **purely geometric**; whether a detection is well linked
is invisible to it, and 0.1 µm decides who inherits the GT identity. This sharpens
CLAUDE.md's "sub-voxel localization is worthless": worthless for *whether* you match,
decisive for *which* of two competing detections matches. The actionable form is §1 —
suppress duplicates and drop edge-less nodes — plus: `validate_submission()` must reject
duplicate `(dataset, t, z, y, x)` rows, which it already plans to.

---

## 6. (b) Empty and degenerate paths

- `_evaluate` returns 0.0 **before matching** when the prediction has no edges or no nodes
  (`metrics.py:251-253`). `evaluate()` then takes its own zero-edge branch (`:308-311`) and
  returns correct counts — **measured**: nodes-but-no-edges gives
  `TP/FP/FN = 0/0/E`, `num_pred_nodes` intact, `J_adj = 0`, and the sample keeps its full
  weight `E_i` in the denominator. That is the right behaviour.
- **But `node_recall()` raises `KeyError: match_node_id`** on such a graph
  (`metrics.py:403` reading an attribute `match` never created). **Measured.**
  `scripts/evaluate.py:88-92` happens to guard it; **any new sweep harness that calls
  `node_recall` without that guard will throw**, and a zero-edge prediction is exactly
  what a too-strict edge threshold produces.
- **`metrics.py:197` is a bare `assert` inside the scorer**
  (`edge_attrs.filter(MATCHED_EDGE_MASK)["pred_valid"].all()`). It should never fire, but
  it compounds with the next item: an `AssertionError` there is swallowed and the sample is
  dropped, so a scorer invariant breaking looks like a *higher* CV score.
- **`scripts/evaluate.py:94-97` catches every exception and *skips* the sample.** A
  skipped sample is dropped from `summarise`, not scored 0 — so a partially failing
  prediction run reports a **higher** CV score than the truth, over a smaller `n`. It is
  printed, so not invisible, but the summary line does not say how many were skipped.
  Any comparison of runs must compare `n` first.
- `_jaccard` returns NaN on a zero denominator (`metrics.py:46-48`); `summarise` drops the
  division term only when `TP+FP+FN == 0` (`:508-517`). Note the refinement: a split with
  no GT divisions but with FP forks does **not** drop the term — it scores `0.1 × 0`, which
  is numerically identical to dropping it. **That is harmless for the number and dangerous
  for the sweep**, because it makes CV blind to precisely the damage §3 is about: with zero
  GT divisions, `division_jaccard = 0/(0+FP+0) = 0` for *any* FP count. **Measured** (probe
  `r5`): on a division-free split, **10,000 FP forks cost 0.000000 score**; the same forks
  on a split containing divisions cost 0.048. 112 of 199 samples contain no divisions
  (CLAUDE.md), so the within-embryo FOV split recommended for fast iteration can easily be
  division-free — and a change that manufactures forks (§3, §3a, §3c) will look free on it
  and cost real points on the leaderboard. Score the division-sensitive changes on a split
  that actually contains divisions. The other documented hazard still stands: never compare
  a division-dropped number against a division-included one.
- `evaluate_datasets` (`:334-383`) returns a micro-averaged Jaccard with **no** node
  adjustment. Already documented; restated because the closed form in §2 makes the two
  look deceptively similar.

---

## 7. (c) Confirmed as documented — no action

- Edges filtered to `dt == 1` (`metrics.py:71-87`); out-degree truncated to the two lowest
  edge ids (`:140-153`); merge duplicates dropped keeping the lowest edge id (`:89-135`);
  duplicate `(source,target)` pairs deduped (`:56-66`). All confirmed. **But see §3** —
  "deleted" is true only of the edge term.
- Over-detection taxed on divisions via forks matched to annotated GT nodes
  (`division_metrics.py:469-472`). Confirmed; §3a extends it to unmatched forks.
- Node-count penalty non-monotonic, bonus capped at +10 %, scores can exceed 1.0
  (`metrics.py:446-449`). Confirmed; §1c and §2b extend it.
- `summarise` drops the division term with a warning when a split has none (`:508-517`).
  Confirmed.
- All the above are `UserWarning`s. Confirmed.
- Division scoring requires a pre-split anchor, both daughter lineages touched, one
  connected component, a predicted fork (`metrics.md:70-93`, implemented at
  `division_metrics.py:216-271`). Confirmed, with §3d's caveat that the anchor may be the
  grandparent.

### One documented item that is *less* dangerous than CLAUDE.md implies

> *`evaluate()` mutates the predicted graph in place. Re-evaluating the same object warns
> and resets. Build a fresh graph per scoring run.*

**Measured** (probe `p11`): three consecutive `evaluate()` calls on the same object return
**byte-identical `EvaluationResult`s**. The reset at `metrics.py:230-246` and
`division_metrics.py:38-55` is complete, and `evaluate_divisions` works on copies
(`:133`, `:158`). The GT graph is **not** mutated at all; the predicted graph gains
`match_node_id`, `match_score` and `matched_edge_mask` (measured). So "build a fresh
graph" is good hygiene, but the `UserWarning` is **not** evidence that a number is wrong —
worth knowing before someone chases it. The one real consequence: if a scored graph is
then re-exported, those three attributes travel with it.

---

## Ranked action list

| # | Change | Where | Expected effect |
|---|---|---|---|
Items 1–2 are unconditional: the mechanism is measured and the sign is not in doubt.
Items 3–5 are **conditional on an operating point we have not measured** — they need a CV
number first, and item 3 needs a calibrated edge probability before its number means
anything.

| # | Change | Where | Expected effect | Conditional? |
|---|---|---|---|---|
| 1 | Reject `dt != 1`, backward, and duplicate edges in the validator | validator | removes an unaccounted division-FP source; each FP ≈ 20–150 edge TPs (≈300 at `adj_J` 0.2, §3b) | no |
| 2 | Emit exactly one fork per suspected division; never in-degree > 1 within ±1 frame of a fork | linker | avoids two strictly-negative hedges (§3b, §3c) | no |
| 3 | Drop nodes with degree 0 before writing | `predict_unet_transformer.py:132` | +up to 10 % relative on the edge term (guaranteed) plus recovered matches (signed by linker precision — §1) | **partly** |
| 4 | Per-sample `N_pred` cap well under `11·N̂_true` | node-budget sweep | avoids zeroing 18 samples / 8.5 % of edge weight | no |
| 5 | Expose the edge threshold and **sweep** it downward from 0.5 | `PredictConfig.threshold` | 0.5 is too strict; `adj_J/(m+adj_J)` ≈ 0.2 is an upper bound on the optimum, and uncharged FPs push it lower (§2a) | **yes — needs calibration first** |
| 6 | Raise `max_parents_per_node` above 1, keep children ≤2 | `predict_unet_transformer.py:92-94` | hedge is +EV above the item-5 threshold; suppress near forks | **yes** |
| 7 | Preserve descending-confidence edge order into the CSV | writer | insurance only; the greedy path already caps children at 2, the ILP path does not | no |
| 8 | Guard/fix `node_recall` on zero-edge graphs; report skipped `n`; score division changes on a split that has divisions | `metrics.py:394`, `scripts/evaluate.py` | stops a crash, a silently inflated CV, and a division-blind sweep (§6) | no |

Probe scripts live in `/tmp/audit/` (`p1b`, `p2`–`p17`) and `/tmp/review/` (`r2`–`r5`,
the review's counterexamples). **`/tmp` is ephemeral and nothing here is pinned by a
committed test.** Items 1, 3, 6, the §1 counterexample, the §6 division-blindness case and
the §0 free-endpoint refinement are the ones worth turning into committed CPU-only tests —
each pins a behaviour that would otherwise be rediscovered the hard way.
