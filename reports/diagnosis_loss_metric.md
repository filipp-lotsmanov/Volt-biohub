# Why model A's detection loss falls 4.4× while its score falls

Diagnosis of the item CLAUDE.md Status (b) calls "the single most important open
item". Analysis only — no code changed, no GPU used. Every number below was
measured on CPU against the one checkpoint that exists,
`weights/modelA_corrected/split_0/edge_predictor_best.pth` (epoch 0), on real
fold-0 data. Reproduction script in the appendix.

## 1. The observation, stated precisely

| | epoch 0 | epoch 11 |
|---|---|---|
| train detection loss (mean over 200 iters) | 0.0416 | 0.0094 |
| selection score | **0.5461** | 0.5248 |

Three qualifications that change how much this observation can carry, and none
of them were recorded with it:

1. **The selection score is not the fold score.** `epoch_selection_score`
   (`scoring.py:162`) scores whatever names it is handed; the subsample is taken
   by its caller, `test_files[:select_videos]`
   (`train_unet_transformer.py:1274`, `select_videos` defaulting to 8 and
   settable with `--select-videos`) — an 8-video subsample of the 128-video
   fold. The published fold number, 0.5886, is epoch 0 scored on all
   128. Epoch 11 has **never** been scored on the full fold.
2. **The yardstick was mis-set, in the direction that matters.** The run is
   `git_sha 9bafdf5`, where `linking.MOTION_GATE_UM = 12.0` and
   `DIVISION_GATE_UM = 10.0`. The gate sweep run *after* it measured those two
   settings as the **worst of the four** linker configurations tried
   (0.5849) against **0.6925** for gate 8 + no division pass. So every epoch of
   model A was ranked by the one linker setting that is now known to convert
   detector improvements into score losses (§4, H1).
3. **`best_score` is a running maximum over a noisy signal.** "Best epoch is 0"
   means only that no later epoch beat 0.5461 on 8 videos. Model B, the same
   architecture with pseudo-labels, moved 0.541–0.549 across all 12 epochs — so
   ~0.008 is the visible epoch-to-epoch spread of this measurement. A's
   0.5461 → 0.5248 is ~2.6× that spread: real, but not enormous.

**The gating gap.** Only the best epoch was ever saved — the run wrote
`edge_predictor_best.pth` and nothing else — so the loss/score curve cannot be
re-scored under a corrected yardstick, and **no hypothesis below can be
*confirmed* on existing artifacts, only supported or refuted.** A `--save-every`
flag landed in the working tree while this analysis was running (§6, E0), so
closing the gap now costs one 12-epoch retrain and nothing more.

## 2. What was measured here

A set of CPU probes on the epoch-0 checkpoint, over fold-0 videos (train embryo
`44b6`, held-out `6bba`). Detection maps come from `model.detect`, peaks from
`detectors._peak_window(5.86, 1.625) = 7` voxels — i.e. **NMS is a ±3-voxel
cube, whose corner is 8.44 µm away**. Budget `k = budget_for(t_true, 100, 1.0)`,
exactly as `scoring.score_video` runs it.

## 3. The mechanism, in one paragraph

`detection_loss` assigns weight **exactly zero** to every voxel that is bright
and more than 7 µm from an annotated centroid (`detection_targets.py:327-331`
builds `keep_neg`, and `343-345` puts a literal `zeros(())` in that branch).
That region is 1.7 %–52 % of the volume — **51.9 % on the fold-0 *training*
embryo**. Node survival, meanwhile, is decided by a *local* winner-take-all
contest: a voxel is a detection only if it is the maximum of its ±3-voxel cube.
Measured over 126 annotated centroids across 8 videos: **73 % of them have
zero-weight voxels inside the very cube that decides their survival**, the
current logit margin between a cell's own best voxel and the best unconstrained
voxel competing with it is only **+4.0 logits at the median (p5 = +0.13; 12 %
below 1 logit)**, and **4 % are already outranked** — their peak is being
deleted by a voxel the loss cannot see. So the loss is *literally flat* in the
direction that decides most of the metric's inputs, and gradient descent moves
that direction as a side effect. The direction is not unbiased: the maximum
logit in the zero-weight region is **+18.4 to +45.3** while the maximum at any
annotated centroid in the same frame is **+8.0 to +21.2**.

**Reviewer correction (2026-09-07):** that last comparison carries no weight and
the "outruns the supervised region by 2–3×" reading it was given here has been
removed. It is a maximum over 1.7–52 % of the volume — 136,076 voxels on
`44b6_0b24845f` t=20 — against a maximum over `n_gt` voxels, which is **1** on
that same frame. With 3.5 % of cells annotated the brightest cell in a frame is
unannotated with probability ~0.96, so a working detector *must* put its global
maximum in the ignored region. Re-measured over 12 frames of 4 videos, one frame
(`6bba_062c8d37` t=10) has the annotated maximum **above** the ignored maximum
(+32.8 vs +32.0), which a genuine 2–3× bias could not produce.

Confirmation by direct perturbation. Adding δ to logits **only inside the
zero-weight region** changes the loss by exactly 0 (it is multiplied by a zero
weight) and moves node recall:

| video | t | δ = −5 | δ = 0 | δ = +5 | loss change |
|---|---|---|---|---|---|
| `6bba_05b6850b` | 20 | 1.000 | 1.000 | 0.625 | exactly 0 |
| `6bba_05db0fb1` | 30 | 0.938 | 0.812 | 0.250 | exactly 0 |
| `6bba_07477033` | 20 | 1.000 | 1.000 | 0.571 | exactly 0 |

Twelve frames measured; the pattern holds on all of them. Note the δ = −5
column: **suppressing the unconstrained region raises node recall above where it
is now** (0.812 → 0.938, 0.889 → 1.000 on the dense video). The mechanism is
already costing us score at epoch 0, before any question of training longer.

**The channel is NMS, not top-k, and this refutes the hypothesis as originally
framed.** Splitting the perturbation into its two effects — re-ranking the same
peak set, versus changing which voxels survive NMS — the ranking channel is
completely inert:

| | ceiling (all maxima) | as-is | ±5 on **ranking** only | ±5 through **NMS** |
|---|---|---|---|---|
| `6bba_05b6850b` t=20 | 1.000 | 1.000 | 1.000 / 1.000 | 0.625 / 1.000 |
| `6bba_05db0fb1` t=30 | 0.812 | 0.812 | 0.812 / 0.812 | 0.250 / 0.938 |
| `6bba_062c8d37` t=20 | 1.000 | 1.000 | 1.000 / 1.000 | 0.778 / 1.000 |

So "the loss rewards confidence at annotated centroids while the score depends
on the *global ranking* of peaks" is **wrong as stated**: annotated centroids
sit comfortably inside the budget (ranks 10–36 of ~350 maxima where k = 64), and
on dense samples the map yields **fewer maxima than the budget asks for** (500
maxima against k = 698), so *every* peak is kept and the ranking is irrelevant.
The competition the loss fails to express is **local** — within one 8.44 µm cube
— not global. That relocation matters, because it changes the fix from "calibrate
the map" to "add a term that orders a centroid above its own neighbourhood".

**Reviewer correction (2026-09-07) — the "REFUTED" verdict above does not hold;
it is an artifact of the three videos it was tested on.** The three rows in that
table are the only fold-0 videos where the global budget provably *cannot* bind:
on `6bba_05b6850b` every annotated cell's matched peak ranks ≤ 32 with k = 64, on
`6bba_062c8d37` ≤ 51 with k = 60, and on `6bba_05db0fb1` the peak count (504) is
below the budget (698) so nothing is cut at all. Re-running the *identical*
ranking-only perturbation on three other fold-0 videos, peak set held fixed:

| video | t | k | as-is | ranking-only δ = −5 | ranking-only δ = +5 |
|---|---|---|---|---|---|
| `6bba_0c7fa718` | 10 | 136 | 0.750 | 0.833 | **0.583** |
| `6bba_0c7fa718` | 20 | 136 | 0.500 | 0.667 | **0.167** |
| `6bba_085bf656` | 10 | 85 | 0.889 | 1.000 | **0.667** |
| `6bba_085bf656` | 20 | 85 | 0.929 | 1.000 | **0.714** |
| `6bba_07e24132` | 10 | 215 | 0.800 | 1.000 | 0.800 |

So the ranking channel moves node recall by up to **−0.33** with the peak set
unchanged, and δ = −5 raises it on 4 of these 8 frames. Measured across the first
24 held-out videos of fold 0 (2 frames each, 381 matched cells): **8 of 23
videos (35 %) have at least one annotated cell whose peak is cut by the global
top-k**, 3.4 % of cells overall. The global-ranking hypothesis is therefore *not*
refuted — it is live on about a third of videos, fed by the same zero-weight
region. Both channels are real; §3's local mechanism is confirmed (see the
independent reproduction note in the appendix), the global one is not excluded,
and the fix wants to address the ranking as well as the local ordering.

## 4. Hypotheses, ranked by cost to test

### H1 — the yardstick was bent, and there may be no loss problem at all
**Cheapest to test, and it must be excluded before anything else.**

Claim: the score fell because it was measured with `gate_um = 12.0` and the
division pass on, a configuration in which a *better* detector scores *worse*.
Mechanism is metric fact 3, already written up: node recall gates how much
linking gets graded, so raising node recall exposes previously-free bad links as
scored FPs. Existing evidence, no new run needed — swapping DoG for the U-Net
(node recall 0.796 → 0.860) is worth **−0.059 at gate 12** and **+0.006 at gate
8** on the same 30 videos. If training raises node recall epoch over epoch (it
does — 0.85 for A against DoG's 0.78), the gate-12 score must fall while the
detector improves. That is the observation, with no loss pathology required.

*Test:* score per-epoch checkpoints at gate 12 **and** gate 8, no divisions.
*Predicted:* the gate-12 curve falls, the gate-8 curve is flat or rises. If so,
H1 is the whole story and CLAUDE.md's item (b) should be rewritten as "the
selection metric was mis-configured", not "the loss is anti-correlated".
Needs E0 (per-epoch checkpoints). Cost: one 12-epoch retrain, ~13 min/epoch.

### H2 — the loss's null space is exactly the quantity NMS uses
Measured in §3 and the strongest *loss-side* mechanism: 73 % exposure, median
margin +4.0 logits, loss-invariant ±5 perturbations moving node recall 1.00 →
0.25. This does not by itself predict a *monotone* decline (a null-space walk is
a random walk), but it does predict that (i) the score is high-variance across
epochs for reasons the loss cannot report, and (ii) the walk is biased, because
the negative term keeps sharpening the intensity boundary the mask is drawn on
while the interior of the mask is free to run away — and it already has, by 2–3×.

*Test A (no training, ~10 min CPU):* raise `bright_threshold`, which is what
sets the mask's extent, and re-measure the exposure. Already run:

| `bright_threshold` | masked volume | GT centroids with an unconstrained voxel in their NMS cube | median margin |
|---|---|---|---|
| **0.3 (default)** | 16.5 % | **79 %** | +4.39 |
| 0.5 | 5.8 % | 30 % | +3.38 |
| 0.7 | 0.8 % | **4 %** | +3.41 |

This is an independent, mechanistic argument for the same direction CLAUDE.md
already derived from occupancy ("over-masks by 4.5× at the median"). Two
separate derivations agreeing on one scalar is the cheapest open move in the
project.

*Test B (one training run):* add a hinge that makes the loss see the contest —
for each annotated centroid, penalise `max_pool3d(logits, 7)` over its own cube
exceeding its own logit by more than a margin. One `max_pool3d`, ~5 lines.
*Predicted:* node recall rises most on the dense/high-mask samples (the δ = −5
column bounds the headroom at +0.06 to +0.13 node recall there), and the
epoch-to-epoch score variance drops.

### H3 — the loss number is dominated by a handful of hard annotated centroids, so it is measuring memorisation
The positive term at a *typical* centroid is already numerically dead: median
sigmoid-equivalent loss ~1e-6 at epoch 0. Decomposition over 72 frames:

| | detection loss mean | p50 | p90 | positive-term share, p50 |
|---|---|---|---|---|
| train embryo `44b6` | 0.0126 | 0.0072 | 0.0196 | 6.7 % |
| held-out `6bba` | **0.0164** | **0.0101** | **0.0355** | 15.6 % |

So (i) the loss is ~93 % negative term on a typical training frame — further
reduction is background suppression, which the score does not reward, and (ii)
the reducible *tail* is a small number of low-logit annotated centroids (one
measured frame: positive term 0.2397 against a negative term of 0.0166, i.e.
94 % of the loss coming from 14 voxels). Driving the printed number from 0.0416
to 0.0094 is therefore consistent with memorising a tail of hard exemplars in
the training embryo — plain overfitting, which would show as a rising *held-out
detection loss*. **The training loop never logs that number**, only train-side
loss and the CV score, and the gap is already 30–40 % at epoch 0.

*Test:* log detection loss on held-out windows each epoch. Two lines in
`train_epoch`'s caller; no new compute beyond one pass over the test loader.
*Predicted:* if held-out detection loss rises while train falls, this is
overfitting and the fix is regularisation/early stopping, not loss design.

### H4 — on fold 0 the loss gets two positive voxels per frame
Measured from the geffs, annotated nodes per usable frame:

| embryo | min | p10 | p50 | p90 | max | ≤2 nodes |
|---|---|---|---|---|---|---|
| `44b6` (fold-0 **train**) | 1 | 1 | **2** | 7 | 16 | **56 %** |
| `6bba` (fold-0 test) | 1 | 4 | 8 | 15 | 33 | 4 % |

The positive term on fold 0 is estimated from a median of **2 voxels per
frame**, against 126,067 negatives and 136,076 zero-weight voxels (51.9 % of the
volume) on the frame measured. Fold 1 trains on `6bba` and gets 4× more. This
does not explain the anti-correlation on its own, but it caps how much the
positive term can ever constrain the model and it makes fold 0 the wrong fold to
debug detection on.

*Test:* run the same 12 epochs on split 1 (train `6bba`, 4× the positives) and
on split 2 (within-`6bba`). *Predicted:* if the decline is milder or absent
there, positive-sample starvation is a contributing factor. Cheap, no code
change: `--split 1`.

## 5. Refuted, with the evidence — so they don't get re-proposed

- ~~**Global top-k ranking collapse.**~~ **Reviewer correction (2026-09-07):
  withdrawn — this refutation does not survive a wider sample.** The 12/12
  "exactly 0.000" frames all came from the three videos where the budget cannot
  bind; on `6bba_0c7fa718` and `6bba_085bf656` the same ranking-only
  perturbation moves node recall by −0.17 to −0.33, and 8 of 23 held-out fold-0
  videos have at least one annotated cell cut by the global top-k. See the
  correction at the end of §3.
- **Saturation destroying the ranking by ties.** In *logit* space the map has a
  wide dynamic range and only 1–17 maxima sit within 0.1 of the k-th value, so
  `_top_peaks`' tie-break jitter is never load-bearing. CLAUDE.md's "median
  sigmoid 1.0000 / ~6000 voxels above 0.99" is a *sigmoid* artifact, not a
  ranking problem — which is exactly why `scoring.detect_video` peaks the logits.
  The line "the detection map carries almost no usable ranking information"
  overstates it and should be corrected: at window 7 the map yields only 332–517
  local maxima per frame, well ordered. (**Reviewer correction:** the span
  originally quoted here, "−4.7 to +52", did not reproduce — over 18 frames of 6
  videos spanning both embryos the logits run −8.4…−6.4 at the minimum and
  +16.2…+42.2 at the maximum. The tie count is 1–17 on 12 frames, not 1–13. The
  conclusion — ties are never load-bearing — survives both corrections.)
- **BatchNorm statistics drifting onto the source embryo.** Re-estimating all 10
  `BatchNorm3d` running stats on the held-out video itself (AdaBN, no labels)
  changes node recall by 0.000 on 9 of 11 frames and **−0.06/−0.10** on the
  other two. Refuted, and it points the wrong way.
- **The fake-pair mismatch at inference.** `model.detect` duplicates one frame
  into a `(frame, frame)` pair while training always saw two different
  consecutive frames. Logit correlation between that path and the real-pair
  `encode` path is **0.9999–1.0000**, node recall identical. Harmless — *and*
  a collateral finding worth its own line: the temporal attention contributes
  nothing measurable to the detection head, so the "temporal" U-Net is paying
  temporal-attention cost for a per-frame detector.

## 6. What to run, in order

**E0 — per-epoch checkpoints (blocker for everything else).** As measured, the
run that produced the observation saved only the running best, which is why this
diagnosis is inferential rather than conclusive. **This landed concurrently**
while the analysis was running: `train()` now takes `--save-every N` and writes
`edge_predictor_epoch{NNN}.pth` (`train_unet_transformer.py:1077,1335`, tests in
`tests/test_train_checkpoints.py`) — uncommitted in the working tree at the time
of writing. So E0 is done; E1 can run as soon as a GPU is free.

**E1 — settle H1 before touching the loss.** With E0 in place:

```bash
CUDA_VISIBLE_DEVICES=$(volt_gpu) uv run python scripts/train_unet_transformer.py \
    --split 0 --epochs 12 --max-iters 200 --batch-size 16 --lr 1e-3 --seed 0 \
    --single-gpu --det-neg-weight 1e-1 --save-every 1 \
    --method modelA_percheckpoint     # output dir is weights/<method>/split_<n>
# then, per epoch, both yardsticks (same 30 videos, one variable changed):
for f in weights/modelA_percheckpoint/split_0/edge_predictor_epoch*.pth; do
  e=$(basename "$f" .pth); e=${e##*epoch}
  uv run python scripts/score_pipeline.py --detector unet --weights "$f" \
      --split 0 --limit 30 --gate-um 8 --no-divisions --run "A_ep${e}_gate8"
  uv run python scripts/score_pipeline.py --detector unet --weights "$f" \
      --split 0 --limit 30 --gate-um 12 --run "A_ep${e}_gate12"
done
```

The scoring half is CPU-only if a GPU is unavailable (`score_pipeline.py`
resolves `cuda if available else cpu`). Two curves from one training run
separates "the loss is broken" from "the ruler was bent" in a single experiment.

**E2 — `bright_threshold` sweep, 0.3 / 0.5 / 0.7.** Predicted direction is known
(up) from two independent derivations, and §4 H2 Test A shows it removes 79 % →
4 % of the exposure. It is one scalar, but **it is not reachable from the CLI**:
`build_detection_loss` never forwards it, so it is pinned at
`DEFAULT_BRIGHT_THRESHOLD = 0.3` (`detection_targets.py:58`, threaded through
`detection_loss`'s signature but not `train_epoch`'s call site at
`train_unet_transformer.py:892`). Exposing it as `--det-bright-threshold` is a
three-line change and is a prerequisite for the sweep — the same class of defect
CLAUDE.md records for `pool_kernel_um` and the edge threshold.

**E3 — the margin hinge.** Only if E1 shows a real loss-side problem. For each
annotated centroid, add `relu(max_pool3d(logits, 7)[centroid] - logits[centroid]
+ m)`. This is the smallest change that makes the loss express the contest the
metric actually runs, and it is a *local ordering* term, which is what §3 says
is missing — not a calibration term.

**E4 — log held-out detection loss per epoch** (H3). Currently invisible.

Do **not** re-ablate Gaussian-vs-voxel targets or PU-masking-on-vs-off until E1
and E2 land: both arms share the zero-weight region, so the ablation would be
measuring the same null space twice.

## 7. Incidental findings

- **`6bba` contains repeated timepoints.** Byte-identical consecutive frames:
  `6bba_05db0fb1` has 10 of 100, `6bba_20852818` has 11; **21** of the first 24
  `6bba` videos have at least one (19 was reported here first; 21 is the
  re-measured count, md5 per timepoint). **The first 24 `44b6` videos have none.**
  Those pairs carry exactly zero real motion, which inflates the "median |dz| =
  0.00 µm" statistic in CLAUDE.md and gives the linker free 0 µm assignments on
  the embryo that carries 85 % of the metric weight.
- **The duplicate positions coincide across FOVs.** `6bba_05db0fb1` and
  `6bba_09961292` duplicate at *identical* timepoints
  (t = 27, 35, 50, 65, 75, 80, 82, 85, 89, 99) though the videos are not
  byte-identical; likewise `6bba_20852818`/`6bba_1f58c2f6`. That is consistent
  with FOVs cropped from one acquisition, which would make the within-embryo
  split (split 2) more optimistic than assumed. Worth a dedicated check before
  any decision rests on split 2.
- **The U-Net map yields only 332–517 local maxima per frame** at the 7-voxel
  window. On dense samples that is *below* the node budget (500 against
  k = 698 on `6bba_05db0fb1`), so the U-Net cannot fill its quota there either —
  the same failure CLAUDE.md attributes to DoG alone. The aggregate node ratio
  of −0.016 hides it.

## Appendix — reproducing the numbers

CPU only, ~4 min. Save as `/tmp/probe.py` and run it with `uv run python`; it
reproduces two of the three §3 perturbation rows (its `sp["test"][:3]` covers
`05b6850b`, `05db0fb1` and `062c8d37`, so the `6bba_07477033` row needs the slice
widened), plus the per-frame exposure count behind H2.

**Reviewer reproduction note (2026-09-07), independent run of this script and of
four further probes.** What reproduced *exactly*: the two covered perturbation
rows (1.000/1.000/0.625 and 0.938/0.812/0.250); the loss invariance — asserted
numerically, `detection_loss` returns a bit-identical value after a +5 shift over
the whole ignored region, on 42 frames; `n_pos=1 / n_neg=126067 / n_ignored=136076
/ 51.91 %` on `44b6_0b24845f` t=20 (and they sum to 64³ exactly); the H4 table to
the digit (`44b6` min 1 / p10 1 / p50 2 / p90 7 / max 16, 56.44 % ≤ 2 over 6,393
frames; `6bba` 1 / 4 / 8 / 15 / 33, 4.21 % over 12,540); 500 maxima against
k = 698 on `6bba_05db0fb1`; and every duplicate-timepoint list in §7, including
the cross-FOV coincidence.

What moved on a wider sample: the margin distribution (217 exposed centroids over
8 videos gives p5 −0.48, p50 +5.70, 11.1 % under 1 logit, **5.5 %** already
outranked, against p5 +0.13 / p50 +4.00 / 12 % / 4 % here); the train-vs-held-out
detection loss gap, which on 42 frames is 8× rather than 30–40 % (mean 0.0081 vs
0.0643) — and note that gap is **between two embryos**, so at epoch 0 it is a
domain difference, not yet evidence of overfitting, since fold 0's train and test
sets differ 3.4× in cell density and 2.5× in ignored-volume fraction.

**One measurement that changes the E2/E3 ordering.** The exposed voxels are not
scattered: because `DEFAULT_IGNORE_RADIUS_UM = 7.0` while the NMS cube's corner
is 8.4437 µm, the ignorable voxels inside a cell's deciding cube are *exactly*
the 7–8.44 µm corner shell. Measured: of 110 annotated centroids, the cube argmax
is an ignored voxel for 3.6 %, and in **100 %** of those cases it lies beyond
7 µm (p50 8.03 µm). Setting `ignore_radius_um = 8.45` — one constant, ≥ the cube
corner — closes that shell while changing the masked volume by less than a point
(36.8 % → 36.2 % on `6bba_05db0fb1` t=20; 1.9 % → 1.7 % on `6bba_05b6850b`):

| video | t | δ = +5 at r = 7.0 | δ = +5 at r = 8.45 | δ = −5 at r = 8.45 |
|---|---|---|---|---|
| `6bba_05b6850b` | 30 | 0.778 | 0.889 (= as-is) | 0.889 (= as-is) |
| `6bba_05db0fb1` | 30 | 0.250 | 0.500 | 0.812 (= as-is) |
| `6bba_062c8d37` | 20 | 0.778 | **1.000** | 1.000 |
| `6bba_062c8d37` | 30 | 0.875 | **1.000** | 1.000 |

The δ = −5 column collapsing onto the as-is value is the null space closing: at
r = 8.45 there is no longer free score to be had by suppressing the shell. That
matters against E2 as written, because raising `bright_threshold` 0.3 → 0.7 buys
the same exposure reduction by taking the masked volume from 16.5 % to 0.8 %,
which discards the density-adaptive PU masking CLAUDE.md defends at ρ +0.87.
`ignore_radius_um` reaches the null space without touching that, and it is the
same three-line CLI-plumbing change (`build_detection_loss` forwards neither
constant).

Verified against the numbers above on 2026-09-07.

```python
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/y2a/Volt-biohub/scripts")
sys.path.insert(0, "/home/y2a/Volt-biohub/src")
torch.set_num_threads(24)
from scipy.ndimage import maximum_filter
from volt_biohub.scoring import load_volumes, read_t_true, load_gt, ISO_UM, DOWNSAMPLE
from volt_biohub.detectors import _peak_window, budget_for
from volt_biohub.detection_targets import ignore_mask
from predict_unet_transformer import load_model

DATA = Path("/home/y2a/data/biohub/train")
sp = json.loads((DATA / "dataset_splits.json").read_text())[0]
CK = Path("/home/y2a/Volt-biohub/weights/modelA_corrected/split_0/edge_predictor_best.pth")
model, _, _ = load_model(CK, torch.device("cpu")); model.eval()
R = _peak_window(5.86, ISO_UM) // 2            # 3 voxels; cube corner 8.44 um
VOX = (ISO_UM,) * 3

def gt_frame(gt, t):
    df = gt.node_attrs(); m = np.asarray(df["t"]) == t
    return np.stack([np.asarray(df["z"])[m] / DOWNSAMPLE[0],
                     np.asarray(df["y"])[m] / DOWNSAMPLE[1],
                     np.asarray(df["x"])[m] / DOWNSAMPLE[2]], 1).astype(np.float32)

def recall(idx, c):
    if len(idx) == 0: return 0.0
    d = np.linalg.norm((idx[:, None, :] - c[None, :, :]) * ISO_UM, axis=-1)
    return float((d.min(0) <= 7.0).mean())

for name in sp["test"][:3]:
    vols = load_volumes(DATA, name); gt = load_gt(DATA, name)
    k = budget_for(read_t_true(DATA, name), len(vols), 1.0)
    for t in (10, 20, 30):
        c = gt_frame(gt, t)
        if len(c) == 0: continue
        img = torch.from_numpy(vols[t]).unsqueeze(0)
        coords = torch.from_numpy(c).unsqueeze(0)
        mask = torch.ones(coords.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            m = model.detect(torch.from_numpy(vols[t])).numpy()
        ign = ignore_mask(img, coords, mask, VOX)[0].numpy()   # zero-weight region

        out = []
        for delta in (-5.0, 0.0, 5.0):                          # loss change: exactly 0
            mm = m.copy(); mm[ign] += delta
            p = maximum_filter(mm, size=2 * R + 1, mode="nearest")
            idx = np.argwhere(mm >= p); v = mm[idx[:, 0], idx[:, 1], idx[:, 2]]
            out.append(recall(idx[np.argsort(-v)][:min(k, len(v))], c))
        # exposure: unconstrained voxels inside the deciding NMS cube
        ci = np.clip(np.round(c).astype(int), 0, 63)
        reach = sum(ign[max(0,z-R):z+R+1, max(0,y-R):y+R+1, max(0,x-R):x+R+1].any()
                    for z, y, x in ci)
        print(f"{name} t={t:2d} recall(-5/0/+5)={out[0]:.3f}/{out[1]:.3f}/{out[2]:.3f} "
              f"exposed={reach}/{len(ci)}")
```
