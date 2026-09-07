# Volt-biohub

Kaggle **Biohub – Cell Tracking During Development**: zebrafish embryo, 3D+time
light-sheet microscopy. Detect cells per timepoint, link them across time,
reconstruct lineages including divisions.

Two goals, score first where they conflict: **top-7 leaderboard placement**, and a
repo that shows engineering judgement. The write-up is assembled in the final week
from artifacts accumulated along the way — don't stop to write essays mid-competition.

**Deadlines** (all 23:59 UTC). Started 2026-06-29. **2026-09-22: entry deadline *and*
team-merger deadline** — the team must exist before it, and nothing recovers a miss.
**2026-09-29: final submission.** Features freeze ~Sept 22.

**Prizes**, top 7: $18k / $12k / $8k / $6k / $6k / $5k / $5k.

**Code-competition rules that constrain design:** submission is a **Notebook**, ≤12 h
run-time (CPU *and* GPU limits both 12 h), **internet disabled**, output named exactly
`submission.csv`. Freely and publicly available **external data and pre-trained models
are explicitly allowed** — so bundling wheels/weights as Kaggle Datasets is sanctioned,
and a pretrained detector (e.g. Spotiflow) is a legal fallback.

**When unsure what to work on:** does this change the CV score, or how fast we find
the next thing that changes the CV score? If neither, it is not a priority.

## This code is ours

Seeded from `royerlab/kaggle-cell-tracking-competition` (BSD-3-Clause), the
organizers' baseline. A starting point, not a dependency: rewrite, restructure and
**delete superseded code rather than leaving it dead**. No obligation to stay
mergeable with `upstream`. Two exceptions:

1. **Do not change the semantics of `metrics.py` / `division_metrics.py`** — that is
   the official scorer. Refactor and add diagnostics freely, but if its numbers stop
   matching Kaggle's, our CV stops predicting the leaderboard and every later
   decision is noise. Changes here need a test proving identical outputs.
2. **Keep `LICENSE` and the README provenance section.** BSD-3 requires the notice,
   and "built on the baseline, then replaced X, Y, Z" is a stronger, checkable claim
   than implying we wrote a 3D UNet from nothing.

Repo stays **private until 2026-09-29** — Kaggle forbids sharing code outside the
team. Flip it public immediately after.

## THE METRIC — read before proposing any method

`metrics.py` + `division_metrics.py`; prose version in `metrics.md`.

```
score = weighted_adjusted_edge_jaccard + 0.1 * division_jaccard
```

Per-sample adjusted edge Jaccard weight-averaged by `w_i = TP+FP+FN`; division
Jaccard micro-averaged. Node matching is optimal bipartite assignment on centroid
distance, **max 7.0 µm**, scale `(z, y, x) = (1.625, 0.40625, 0.40625)` µm/voxel.
**Ground truth is sparse** — only a subset of cells is annotated.

### Five facts that determine the method

1. **Divisions are worth 10%.** Don't optimize mitosis until edges plateau.
2. **False positives outside annotated regions are free.** A predicted edge counts
   at all (TP *or* FP) only when `out_valid(matched source) OR in_valid(matched
   target)` — an endpoint matched a GT node that itself has GT edges. Everything
   else is discarded, not penalized. **Dense over-detection is nearly cost-free**
   *for the edge term* — but see the caveat below: it is **taxed on the division
   term**, which is the one place over-detection bites.
3. `edge_FN = gt_num_edges - TP`. With (2): **recall dominates, global precision
   barely matters** — *for nodes*. **For edges this is wrong, and believing it cost us a
   day.** Fact (2) says an edge is scored only when an endpoint matched a GT node that
   itself has GT edges, which makes **node recall a gate on how much of your linking gets
   graded**. Raise node recall and previously-invisible links become *scored* links.
   Measured on 20 samples of fold 0, same linker, same budget:

   | | DoG | U-Net |
   |---|---|---|
   | node recall | 0.792 | **0.860** |
   | edge TP | 11,520 | **11,774** |
   | edge FN | 4,576 | **4,322** |
   | edge **FP** | **1,674** | 3,830 |
   | edge precision | **0.873** | 0.755 |
   | edge Jaccard | **0.648** | 0.591 |

   The U-Net is strictly better on edge *recall* — more TPs, fewer FNs — and loses on a
   **2.3× FP excess**. It earned 254 extra TPs and paid 2,156 extra FPs for them. DoG's
   bad links near the 21 % of GT nodes it misses are **free**; the U-Net matches those
   nodes, so the same bad links now cost. **Detector and linker therefore cannot be swept
   independently** — a detector improvement changes what the linker is graded on, so
   sweeping a detector against CV with a fixed linker systematically undervalues the
   better detector.
4. **The node-count penalty is soft and non-monotonic:**
   `J * (1 - 0.1 * (N_pred - N_true)/N_true)`, clamped at 0, `N_true =
   estimated_number_of_nodes` from the GEFF metadata `extra`. Doubling node count
   costs ~10% relative; predicting *fewer* nodes multiplies J *above* 1.0 (hence
   scores > 1.0). Total node count is a scalar hyperparameter with an interior
   optimum — tune it last, it interacts with detection and linking both.
   **But `N_true` is unreadable at inference**: it lives only in the GT geff, and the
   test set ships images alone (see Data). It also varies 20× across samples
   (3,783–78,644). **It is, however, predictable from metadata the test set *does*
   carry** — the stored intensity quantiles give Spearman **0.856** against `q0.9` and a
   log-linear fit on `p90` alone reaches R² 0.72 / 25 % median error (`check_t_true.py`).
   That matters because a single global `N_pred` is expensive: the best constant scores a
   weighted multiplier of **0.936** (min 0.627, 63 samples below 0.9) while a per-sample
   predicted target reaches **0.993**. So this knob is *not* condemned to be global.
5. **7 µm is generous** — ~4 z-voxels, ~17 xy-voxels. Sub-voxel localization is
   worthless here.

Consequences:

- Tune the detector for **recall at a fixed node budget**, not F1: sweep (detection
  threshold, suppression radius) against CV score, never against detection P/R.
- **The suppression radius is squeezed from both sides, and the upper bound is the one
  that binds.** Cell pitch argues for a large radius — it is a *distribution*,
  **11.3–31.0 µm, median 18.5 µm**, so no single value is derivable and it wants
  sweeping — and GT nearest-neighbour spacing (~23 µm) must *not* be used, being a
  sparse-sampling artifact. But **sister separation one frame after a division caps it
  hard: min 2.87 µm, p1 4.11, p5 5.41, median 10.57** over all 151 events. Any radius
  above the separation merges the two daughters into one detection and makes the division
  unrecoverable. At the current 3.0 µm exactly one event is already lost; 4 µm risks one
  or two more; 5.4 µm risks about seven (~0.005 of score at the 0.1 division weight).
  **So the pitch argument is bounded at ~4 µm, not 18 µm** — the two constraints
  disagree by 5× and the division term is what breaks the tie. The one real cost of
  over-detection is a spurious point winning the assignment the true node should have
  taken.
- The node penalty is mild in both directions: predicting 2× `N_true` costs ~10%
  relative, and the bonus for under-predicting is capped at +10% (the ratio bottoms out
  at −1). Since a missed node costs two edges outright, **the asymmetry favours
  over-detection** — but the budget is worth a deliberate sweep, not a shrug.
- Never spend effort on centroid refinement, sub-voxel peak fitting, or segmentation
  masks. None of it is measured.
- A missed detection costs two edges, not one.
- **Over-detection is free on edges but taxed on divisions, and this is measured, not
  theorised.** The overfit test (see Status) memorised a 5-frame clip to a perfect
  **edge Jaccard of 1.000** with node recall 1.0 — and still scored **division
  Jaccard 0.5**. With exactly one GT division in the fixture that arithmetic pins
  down to TP=1, FN=0, **FP=1**: it found the real division and invented a second one,
  while predicting 28 nodes against 22 GT. The mechanism is structural — a spurious
  extra detection near a real cell lets the ≤2-children rule attach a second child,
  and a fork in an *annotated* region is a division FP (only forks in unannotated
  regions are ignored). So the node budget is not a pure edge-recall trade: pushing
  detection density up buys edge recall almost free and pays for it in division
  precision. Sweep the two terms together, never the edge term alone.
- Aggressive detector + *selective* linker beats conservative detector + permissive
  linker.
- **Linking is close to solved by geometry, so the learned linker is not where the score
  is.** At *true* inference density — pitch from `estimated_number_of_nodes`, not from
  sparse GT spacing — the half-distance to a cell's nearest neighbour beats its own
  per-frame motion by a wide margin:

  | | p10 | p50 | p90 |
  |---|---|---|---|
  | true pitch (µm) | 12.5 | 18.5 | 27.3 |
  | motion p50 (µm) | 1.22 | 1.82 | 2.96 |
  | **margin** (half-pitch / motion) | 3.05 | **4.86** | 10.03 |

  The typical case is unambiguous in **199 of 199** samples and the p99 tail in 157; the
  samples that are genuinely ambiguous carry only **~10 %** of edge weight. Confirmed
  end to end: Hungarian assignment on µm distance, with no network at all, scores
  **0.6424 on LOEO fold 0** (`src/volt_biohub/linking.py`). So spend learned capacity on
  *detection* and keep the geometric linker — which also removes the transformer's
  zero-edge failure from the critical path.

### Silent scorer behaviours — these lose points without raising anything

- Edges are filtered to `t_target - t_source == 1`; **any longer edge is deleted**.
  Express gap closing by *inserting an interpolated node*, never as a t→t+2 edge.
- Out-degree > 2 is truncated to the two lowest edge IDs (insertion-order
  dependent). Enforce ≤2 yourself.
- Merge duplicates — several predicted edges collapsing onto one GT edge — are
  dropped, keeping the lowest edge ID.
- `evaluate()` **mutates the predicted graph in place**. Re-evaluating the same
  object warns and resets. Build a fresh graph per scoring run.
- **Two score paths, only one is the leaderboard's.** `evaluate_datasets()` returns a
  micro-averaged edge Jaccard with **no node-count adjustment**; `summarise()` (what
  `scripts/evaluate.py` prints) returns the weighted *adjusted* Jaccard. Never
  compare numbers across the two.
- `summarise()` silently **drops the division term** (with a warning) if a split
  contains no divisions at all. Empty denominators give NaN, not 0.
- All of the above are `UserWarning`s. `filterwarnings = ["error::UserWarning"]` in
  pytest would make our own regressions loud, but several tests deliberately trigger
  them — scope it per-test.

**Division scoring** (±1 timepoint tolerance) counts a GT division as TP only if:
the prediction covers a pre-split timepoint; both daughter lineages are touched
(possibly at different timepoints); all matched nodes lie in one connected
component; and that component contains a predicted fork. Forks in unannotated
regions are ignored.

## Submission format

`scripts/geffs_to_csv.py` is the reference implementation — and it validates nothing.

```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

Node rows: `row_type=node`, integer `t,z,y,x`, `source_id=target_id=-1`. Edge rows:
`row_type=edge`, `node_id=t=z=y=x=-1`. `id` is a throwaway consecutive index;
`dataset` matches test folder names without `.zarr` (i.e. `{embryo}_{fov}`); file must
be named exactly `submission.csv`. **`sample_submission.csv` ships with the data and is
the format oracle** — diff our writer's output against it rather than trusting the spec.

Write `validate_submission()` early and run it before every save — a silent format
error costs a submission slot, and near the deadline the competition. Invariants:

- integer coordinates in **original anisotropic voxel space**, not isotropic
  resampled space (`io.rescale_graph_to_original` inverts it)
- node IDs unique **within each dataset** — `sample_submission.csv` restarts them at 1
  for every dataset, and `csv_to_geffs.py` groups by dataset before resolving edges, so
  global uniqueness is *not* required (an earlier note here claimed otherwise)
- every edge strictly t → t+1; ≤2 outgoing edges per node
- no duplicate nodes (same dataset+t+z+y+x), no duplicate edges
- every `source_id`/`target_id` resolves to a node row
- **every** test dataset present, even one yielding zero nodes

## Environments

| Role | Machine | Purpose |
|---|---|---|
| Dev | Windows 11, RTX 5070 (12 GB) | iteration, debugging, visualization |
| Train | University Linux box, 8× 48 GB (A6000 / RTX 6000 Ada), shared | all real training |
| Inference | Kaggle notebook, T4/P100 (16 GB) | submission only |

Code must run unmodified on all three. The binding constraint is asymmetric:
**we train on 48–96 GB and must infer on 16 GB within 12 h.** Large training patches
are a free luxury; a large *model* is not. Measure full test-set timing on Kaggle in
week 2, not from one video in week 4.

**Windows dev box (this machine).** PowerShell 7 — *assume PowerShell syntax*. `uv`
for Python, `.python-version` pinned to **3.11** to match the Kaggle image; don't
bump it. The 5070 is Blackwell (`sm_120`) and needs CUDA ≥ 12.8, but PyPI's Windows
torch wheels are CPU-only, so `pyproject.toml` routes torch to the **cu130** index
on `sys_platform == 'win32'` (cu128 wheels stop at torch 2.11; cu126 has no
`sm_120`). After any dependency change:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

`sm_120` must appear or kernels fall back to PTX JIT. `VOLT_DATA_DIR` is set (User
scope) to `C:\data\biohub\train`; `LongPathsEnabled` is already 1. Only one drive,
~100 GB free, so keep only a working subset of the data here. Given 64³ training
windows the 12 GB card is genuinely usable for real training at this scale, not just
for the `division_clip` fixture — the server is for sweeps and long runs, not for
making training possible at all.

**Linux server** — measured 2026-09-02, all the open questions are now answered.
It is a **Docker container on a shared 8-GPU box** (`coder.ai` workspace), not a
SLURM cluster: **no scheduler**, bare shell, 256 cores, 1 TB RAM.

- **GPUs: 4× RTX A6000 (`sm_86`) + 4× RTX 6000 Ada (`sm_89`), 48 GB each.**
  Not Blackwell, so the cu130 index marker **stays Windows-only**. Linux's default
  PyPI wheel is now `torch 2.13.0+cu130` anyway, whose arch list is
  `sm_75/80/86/90/100/120` — **`sm_89` is absent**, but CUDA guarantees binary
  compatibility upward across minor revisions, so the `sm_86` cubins run on Ada.
  Verified by executing `conv3d` on an Ada device, not assumed.
- **The GPUs are shared with other tenants**, and *which* cards are busy changes day
  to day, so a hardcoded index is wrong by tomorrow. `nvidia-smi` sees their memory
  but **not their processes** from inside the container — never infer "idle" from an
  empty process table. **Always pin `CUDA_VISIBLE_DEVICES`**, chosen at run time by
  `src/volt_biohub/gpu.py`: `CUDA_VISIBLE_DEVICES=$(volt_gpu)`, or `volt_gpu --list`
  to see the box and `volt_gpu --interactive` for a table plus a prompt. It samples
  each card several times (one instantaneous reading reports a busy card as idle
  often enough to matter) and ranks on free memory, then a 10 %-bucketed
  utilisation, then compute capability. `/home/set_gpu.py` (shipped on the box) does
  **not** work for this: it sets `os.environ` in a process that immediately exits.
- **256 cores and 1 TB RAM make CPU-side work essentially free here** — registration,
  classical detection, zarr decode and large `num_workers` all cost nothing on this
  box. **This does not transfer to Kaggle**, whose notebooks have a small
  single-digit vCPU count (verify it on the box in week 2 rather than trusting this
  line). So CPU-heavy *preprocessing* is a fine way to explore ideas cheaply, but any
  CPU cost that lands in the *inference* path must be budgeted against the 12 h cap
  on Kaggle's hardware, not this machine's.
- **Keep the whole project under 100 GB** — the box admins object above that. It sits
  at **94 GB** today (83 GB data + 5.7 GB repo/venv + 5.6 GB uv cache), so there is
  little headroom: weights and predictions need a budget, and the 81 GB archive was
  deleted after extraction for exactly this reason. Beware that a file-manager delete
  moves files to `~/.Trash-0` and **frees nothing** — check `du -sh ~/.Trash-0` after
  deleting anything large. If space gets tight the prunable items are the uv cache
  (5.6 GB, costs a re-download on rebuild) and `test/` (1.8 GB of byte-identical
  copies of train samples).
- **Only `/home/y2a` is persistent** (an ext4 volume, 6.6 TB free). Everything else,
  `/root` included, is the container's ephemeral overlay. So `uv` lives in
  `/home/y2a/.local/bin`, `UV_CACHE_DIR` in `/home/y2a/.cache/uv` (5.6 GB — a rebuild
  would otherwise re-download it), and the venv is already under the repo. A rebuild
  needs only `bash startup.sh`, which restores the SSH keys and re-sources
  `~/.volt-env.sh` into `.bashrc`/`.profile`.
- **Outbound internet works** (pypi, github, kaggle all 200), so the `tracksdata` git
  dependency resolves. Training must still checkpoint-and-resume — not for preemption
  but because the container can be rebuilt out from under a run.
- **The filesystem handles zarr's many small files comfortably**: 5.5 k 64 KiB
  writes/s (348 MiB/s), 21 k reads/s. 24,886 files is a non-issue.

`uv sync` provides the Kaggle CLI. Auth is the new-style `KGAT_…` token — **not** the
retired `kaggle.json` pair — and `KAGGLE_CONFIG_DIR` does **not** apply to it:
`kagglesdk` hardcodes `~/.kaggle/access_token`, which is ephemeral here. Since
`KAGGLE_API_TOKEN` accepts a *path* (`kagglesdk/kaggle_env.py:99`), it points at
`/home/y2a/.kaggle/access_token` on the persistent volume. The competition rules must
be accepted in the browser or the API returns 403. Data is CC0, so university storage
is fine.

**Kaggle notebook.** Internet **disabled**, ≤12 h, output `submission.csv`.
`tracksdata` is **not on PyPI**, so **keep the inference path free of tracksdata** —
numpy/scipy/torch/zarr plus our own CSV writer — and use tracksdata only locally for
GT parsing and scoring. Since `.geff` is plain Zarr v3 (see Data), this costs little.
That removes the largest single submission-failure risk. If ILP is needed at inference,
`ilpy` + `pyscipopt` ship manylinux wheels and bundle fine. Offline install:
`pip download <pkgs> -d wheels --only-binary=:all: --platform manylinux2014_x86_64
--python-version 3.11`, then `pip install --no-index --find-links=/kaggle/input/<ds>`.
Three Kaggle Datasets — **wheels**, **code** (this repo, versioned per push with the git
SHA in the message so an LB entry traces to a commit), **weights**; the notebook is ~30
lines of glue, all logic lives in the repo. 20 GB output storage fills faster than
expected.

**Cross-platform rules** (each of these has caused real bugs):

- **`pathlib.Path` everywhere** — no path concatenation, no `\` or `/` literals, no
  drive letters outside a shell profile. Paths enter via `VOLT_DATA_DIR`, CLI args,
  or `dataspec.py`. `os.environ`, never `$env:`, inside Python.
- Linux is case-sensitive, Windows isn't; if it broke only on Linux, check casing.
  `.gitattributes` pins `eol=lf` so CRLF churn can't make phantom diffs.
- **DataLoader workers differ**: Windows spawns (workers re-import the module,
  everything must be picklable, `if __name__ == "__main__":` mandatory), Linux forks.
  Make `num_workers` a CLI arg defaulting to 0 on Windows.
- **Checkpoints must be portable**: `state_dict` + a plain-dict config, never a
  pickled model object; load with `weights_only=True`. It has to load on Kaggle.
- **Never assume a GPU exists** — one device resolver (`cuda` if available else
  `cpu`). `open_dataset(device="cuda")` still violates this.
- `.venv/` is per-machine; `uv.lock` is committed and is the cross-platform contract.
- Set seeds explicitly and record them in the run config.

## Data

87.61 GB, 24,886 files, CC0. Never in git — `.gitignore` covers `data/`, `tests/data/`,
`weights/`, `predictions/`, `results/`, `*.zarr`, `*.geff`, `submission*.csv`,
`dataset_splits.json`. `scripts/dataspec.py` resolves the data dir: `$VOLT_DATA_DIR` →
Kaggle mount → `<repo>/data/train`.

**Shape and size.** Each sample is `{embryo}_{fov}.zarr` (Zarr **v3**) with the array at
`0/`, shape `(T, Z, Y, X)` — **typically `(100, 64, 256, 256)` uint16**, one timepoint per
chunk (`0/c/{t}/0/0/0`, blosc/zstd). That is **~839 MB uncompressed per video**, and at
scale `(1.625, 0.40625, 0.40625)` µm/voxel a physical extent of roughly **104 µm cubed**.
Two consequences:

- **These are field-of-view crops, not whole embryos.** Whatever bulk drift exists is
  drift within a small box; a whole-video-in-RAM load is a nuisance, not a blocker.
- The default `--downsample 1,4,4` is not arbitrary: `0.40625 × 4 = 1.625`, exactly the
  z spacing, so it yields a **64³ isotropic grid at 1.625 µm with no interpolation**.
  Everything trains on 64³ volumes, which is small.

Distances are still computed in µm, never voxels — the raw grid is 4× anisotropic in z.

**Ground truth** is `{name}.geff` (also Zarr v3, zstd): `nodes/ids`,
`nodes/props/{t,z,y,x}/values` (**integer** voxel centroids), `edges/ids` of shape
`(N, 2)` = `(source_id, target_id)`, and `estimated_number_of_nodes` in the metadata
`zarr.json` — the `N_true` of the node-count penalty. **All of it is readable with plain
zarr**, which is what makes a tracksdata-free inference path straightforward rather than
speculative.

**Normalization is safe** — settled, not assumed: the *group*-level `zarr.json` of both
train and test samples carries `multiscales` (scale `(1.0, 1.625, 0.40625, 0.40625)`)
**and** `image_statistics.quantiles`. The array-level `0/zarr.json` has empty attributes;
`open_dataset` reads the group, so it works. Quantile *values* differ substantially per
sample (e.g. 0.001/0.999 of 44.9/1577 vs 26.2/2145), so per-sample normalization is doing
real domain-shift work across embryos — keep it.

**Which quantiles — `DATASET.md` is right about `io.py` and wrong about the pipeline, so
don't act on it.** It reports that the loader normalizes by true **min/max**
(`_process_on_gpu` defaults `q_min=0.000, q_max=1.000`) and that this compresses real
signal, spreading the normalized p90 42× across samples. True of
`io.open_dataset(normalize=True)` — but **neither `train` nor `predict` goes through it**.
Both call `open_dataset(normalize=False)` and then normalize themselves from
`quantiles["0.001"]`/`["0.999"]` with `clamp(min=0)` and no upper clamp
(`train_unet_transformer.py:335`, `predict_unet_transformer.py:369`). The two paths agree
exactly, so there is **no train/inference normalization mismatch** and no min/max problem
to fix in the training path. `_process_on_gpu`'s min/max default only reaches
visualization and other `open_dataset` callers — worth fixing there, but it is not a
scoring bug. Residual spread under the *actual* 0.001/0.999 normalization, measured over
all 199 samples: normalized p90 ranges **0.066 → 0.726, an 11× spread**.

**Uniformity — measured across all 203 downloaded samples, not spot-checked.** Every
sample is *exactly* `(100, 64, 256, 256)` uint16, chunked `(1, 64, 256, 256)` (one
timepoint per chunk), scale `(1.0, 1.625, 0.40625, 0.40625)`, and **all 203 carry
`image_statistics.quantiles`** — so the hard-crash-on-missing-quantiles path in
`train`/`predict` is latent, not active. Quantile *values* vary far more than the earlier
two-sample check suggested (q0.001 spans 0.0–465, q0.999 spans 615–5398), so per-sample
normalization is doing more domain-shift work than assumed, not less.

**The underlabelling is SPATIAL, not temporal — and this decides where relabelling
effort is worth spending.** Two different things get called an annotation "gap" and only
one of them is large. Measured over all 199 GT files:

*Dataset-level holes — small.* **140 of 199 samples are annotated at every frame
t=0..99.** Interior holes (a frame with no annotation inside the annotated span) exist in
only **25** samples and are a **median 1 frame** long, worst case 24. What does happen is
truncation at the ends: 29 samples start after t=0 (median 12 frames missing) and 25 stop
before t=99 (median 19). The extreme is `44b6_0b24845f`, annotated only t=11..50.

*Track-level abandonment — common, but cheap to fix.* **41 % of tracklets (1,828 of
4,435) end mid-movie while their dataset carries on being annotated, and away from the
volume border** — so not a cell leaving the field of view. Orphaned *starts* number
1,741, suspiciously close, which suggests one lineage cut in two. Testing that directly —
for each orphaned end, is there an orphaned start within `gap × 6.84 µm` (the motion p99)
in the next k frames?

| gap | joinable ends | cumulative | % of orphans |
|---|---|---|---|
| 1–3 frames | 233 | 233 | 12.7 % |
| 4–7 frames | 271 | 504 | 27.6 % |
| 8–10 frames | 259 | 763 | 41.7 % |
| unjoinable within 10 | — | 1,065 | **58.3 %** |

**Closing every small and medium gap would add roughly 1,821 edges to a corpus of
128,883 — about +1.4 %.** So gap-closed labels are worth having but cannot move the
score much, and 58 % of abandoned tracks have no partner at all within 10 frames.

The reason is arithmetic: gap closing repairs temporal holes *inside the annotated 3.5 %*.
It does not touch the **96.5 % of cells that were never labelled at all**. Any experiment
framed as "how much do the labels matter" has to attack the spatial axis to see a real
effect — which is what `--det-pseudo-weight` does (see Known defects → addressed).

**How sparse "sparse" actually is** — the earlier figures came from two unusually sparse
samples and understated density by ~3×. Measured over all 199 GT files: **133,318 nodes /
128,883 edges / 151 divisions**, all 199 readable with plain zarr, **zero integrity
problems** (no duplicate ids, every edge endpoint resolves, all centroids integer and
inside the volume, no out-degree > 2, and **no GT edge with `dt != 1`** so the scorer's
edge filter never bites on GT).

| per sample | min | p10 | p50 | p90 | max |
|---|---|---|---|---|---|
| GT nodes | 50 | 150 | **659** | 1230 | 1950 |
| GT edges | 49 | 147 | **639** | 1187 | 1879 |
| divisions | 0 | 0 | **0** | 2 | 5 |
| `estimated_number_of_nodes` | 3783 | 5553 | **17909** | 57347 | 78644 |
| annotated fraction | 0.13 % | 0.41 % | **3.56 %** | 15.1 % | 20.2 % |
| annotated nodes/frame | 0.7 | 1.7 | **6.6** | 12.7 | 20.1 |

- The median sample is **3.56 %** annotated, not 0.2–1.6 %; only **7 of 199** are
  single-lineage like `44b6_0113de3b`. The linker is still data-starved in absolute terms
  (~7 annotated nodes per frame), but "one row of the transition matrix" was wrong.
- Annotation starts at **t=0 in the median sample** (max t=46), not t=4.
- **112 of 199 samples contain zero divisions** and there are only 151 divisions in the
  entire training set, so per-video division Jaccard is pure noise — and `summarise()`
  silently drops the division term for a split with none, which a within-embryo split can
  easily produce.
- The weighted score is **not** dominated by a few videos: the top-20 samples by
  `w_i` carry only **22 %** of all GT edges. It is spread across ~199 samples, so a
  per-sample regression hides easily and the aggregate moves slowly.

**The two embryos are wildly asymmetric in metric weight, and this is the CV constraint
that matters most.** `44b6` is 71 samples but only **0.77 %** annotated (median 214
nodes); `6bba` is 128 samples at **5.37 %** (median 826) — that 5.37 % is the *ratio of
totals*; the median of per-sample ratios is **9.71 %**, and the two definitions must not
be mixed (`44b6` gives 0.77 % under both, which is why the inconsistency was invisible).
So `44b6` carries **just 15 %
of the total GT edges** despite being 36 % of the samples. The two LOEO folds are
therefore *not* two comparable estimates: the fold scored on `44b6` rests on a sixth of
the evidence. Weight them accordingly and expect the `44b6` fold to be the noisier one.

**Nucleus size, measured — this is what sets the detection target width.** Radial
intensity profiles around 208 GT centroids (12 samples spanning both embryos), computed
in µm so the 4× z anisotropy cannot distort them: intensity peaks at 0.571 normalised on
the centroid and falls to a flat 0.304 floor by ~7 µm, giving **nucleus FWHM = 6.9 µm**
and hence a Gaussian sigma of **2.93 µm (1.81 voxels** on the isotropic 1.625 µm grid).
Two cells at the *minimum* observed pitch of 11.3 µm still produce a clearly bimodal
target, so peaks stay separable.

**The unannotated cells are brighter than the annotated ones** — the sparse-negative
problem in one line. Normalised intensity, p5/p50: GT centroids 0.239/0.479; random
voxels 0.008/0.110; **local maxima carrying no GT annotation 0.406/1.000**. The old loss
was therefore teaching the network that the brightest blobs in the image are background.
A bright-voxel threshold of **0.3** sits below 90.9 % of GT centroids and above ~95 % of
non-GT peaks while masking 17.5 % of voxels — comfortably more than the 3–12 % that
cells actually occupy at the observed densities.

**But 17.5 % is a mean over a wildly skewed distribution, and the level is wrong.**
Measured over all 199 samples under the *training* normalisation (0.001/0.999, three
frames each): the fraction of voxels the PU mask can drop is **median 11.1 %, p5 1.9 %,
p95 49.0 %** — a **25× spread**, and systematically per embryo (`44b6` median 25.5 %,
`6bba` 7.5 %). On **23 samples it drops >40 % of voxels** from the negative term, which
is most of the background holding the logits down. Two consequences, and they point in
opposite directions:

- **The fixed threshold is not arbitrary — it is accidentally density-adaptive.** The
  masked fraction tracks the cell occupancy implied by `estimated_number_of_nodes` at
  **Spearman +0.87** (nucleus volume ≈ 40 voxels on the 1.625 µm grid). So it is already
  doing the per-sample job a global constant has no right to do, which is a real defence
  of the design and an argument *against* replacing it with a fixed masked fraction —
  true occupancy varies 13× across samples, so a constant fraction would be worse.
- **The level over-masks by 4.5× at the median** (p5 1.7×, p95 9.0×) against implied
  occupancy of median 2.7 %. So `bright_threshold` wants raising, and it is a **single
  scalar sweep on CV** — the cheapest open knob in the detector. Sweep it together with
  `--det-threshold`, since both move the same precision/recall trade.

**Measured motion — pooled over all 128,883 GT edges**, which is the distribution a
linking gate must actually cover:

| p50 | p75 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| 1.82 µm | 2.73 | 4.14 | 5.34 | 8.38 | 13.8 | 60.8 |

- The 7 µm matching tolerance is indeed **larger than the median inter-frame motion**
  (1.82 µm), confirming sub-voxel localization is worthless.
- **A 5 µm gate forfeits 6.30 % of real edges outright** (8,119 of 128,883) — the low end
  of the old "~5–10 µm" advice is not safe. Gate at **≥10 µm** (misses 0.30 %) or 12 µm
  (0.18 %). Since recall dominates and false positives outside annotated regions are
  free, the gate should err generous.
- Motion is anisotropic in a way that matters: **`|dz|` has median 0.00 µm** (cells move
  under one 1.625 µm z-voxel per frame) while `|dxy|` median is 1.22 µm. A spherical gate
  in µm is fine; a gate in *voxels* would be wrong by 4×.
- **`6bba_f20478e9` contains a single 60.8 µm displacement, 60.1 µm of it in z** — 37
  z-voxels in one frame. That is almost certainly an annotation artifact or an id swap,
  not biology. Treat it as suspect rather than as evidence that the gate must be 60 µm.

**Cell pitch is not one number.** `estimated_number_of_nodes` ranges **3,783 to 78,644**
(38 to 786 cells per frame in the same ~104 µm cube), giving a per-sample pitch of
**11.3 µm (min) to 31.0 µm (max), median 18.5 µm**. The earlier "~15 µm" sits at the 24th
percentile — it is not a global constant. Do still refuse to read the suppression radius
off GT nearest-neighbour spacing (~23 µm, an artifact of sparse annotation), but the
replacement is a *distribution*, so a fixed radius is a compromise on most samples and
the knob wants sweeping rather than deriving. The current defaults (5 µm training, 3 µm
inference) sit below even the densest sample's pitch, so the detector over-detects
everywhere — nearly free per metric fact 2, but it spends the node budget.

**`estimated_number_of_nodes` is NOT available at inference — this constrains the
method.** It lives only in the GT `.geff` metadata (`attributes.geff.extra`). The test
folder contains **`.zarr` images and nothing else**: no `.geff`, and the image group
attrs carry only `image_statistics` and `multiscales`. Verified by exhaustive search of
the test tree. Consequences:

- The node-count penalty's `N_true` is knowable in CV and **not directly readable on the
  hidden test set** — but it is **well predicted by metadata the test set does carry**.
  The stored `image_statistics.quantiles` give Spearman **0.856** against `N_true`
  (`q0.9`; `q0.1` 0.838, `p90−p10` 0.82), and a log-linear fit on `p90` alone reaches
  **R² 0.72, 25 % median relative error**, against 54 % for a per-embryo constant. So
  metric fact 4's sweep is *not* restricted to one global setting: the per-sample
  predicted target is worth **0.993 vs 0.936** weighted multiplier. Earlier notes here
  said this had to come from the image; it does not.
- `N_true` and annotated edge count are mildly **negatively** correlated (Spearman
  −0.28), so the samples most exposed to the node penalty carry **28.8 %** of total edge
  weight against a 25 % population share. Edge weighting does not shield them.
- Any per-sample adaptation of the *suppression radius* still has to come from the image
  or from a fixed sweep, since pitch spans 11–31 µm — but see the division cap above:
  the usable range is much narrower than the pitch spread suggests.

**Only two embryos exist in train** (measured from the file manifest): **199 samples,
79.8 GiB, `44b6` (71 samples) and `6bba` (128)**, ~411 MiB each. Folders are
`{embryo_id}_{field_of_view}`; train and test share no embryo, so the hidden test set is
a **third, entirely unseen embryo**.

Consequence for CV, and it is a hard constraint rather than a preference:
leave-one-embryo-out is **exactly two folds** (train 44b6 → test 6bba, and the reverse).
That is the only honest estimate of what the leaderboard measures, and it is two numbers
with no error bars. So:

- Report the 2-fold LOEO score as *the* number, and never tune a hyperparameter on a
  delta smaller than the gap between the two folds.
- Use a grouped-by-FOV split *within* an embryo for fast iteration, knowing it is
  optimistic — it answers "did this change break something", not "will this generalize".
- Unusually for a Kaggle competition, the **leaderboard is a genuine third embryo** and
  therefore real information. Still don't search on it, but weigh it more than you
  normally would against a 2-fold CV.

**The local `test/` folder is fake** — now proven, not inferred. It holds exactly **4
samples, byte-identical to their train counterparts** (`44b6_0113de3b`, `44b6_0b24845f`,
`6bba_05b6850b`, `6bba_05db0fb1` — md5 of the concatenated chunks matches train), and
they are images only, no `.geff`. At notebook rerun a hidden test set is swapped in,
**approximately the size of the training set**. So the 12 h budget must be measured
against roughly train-sized throughput, and a local `test/` score means nothing.

**`sample_submission.csv` names exactly those 4 placeholder datasets** (20 data rows: 3
nodes and 2 edges each). It is the format oracle for *column layout and conventions* —
node rows carry `source_id=target_id=-1`, edge rows carry `node_id=t=z=y=x=-1`, and
`node_id` restarts at 1 in every dataset, confirming node ids need only be unique
per-dataset. It is **not** a source of dataset names: at rerun those change. **The
submission writer must enumerate the test directory**, never read the dataset list out of
`sample_submission.csv`.

`tests/conftest.py` auto-builds `tests/data/division_clip` from frames 26–31 of
`6bba_c328f2fd` (real division at original t=28 → fixture t=2) with recomputed quantile
attrs. **The dev-loop asset**: 5 frames, known division, seconds on the 5070 or on CPU.

**Data placement.** The Windows box has ~100 GB free, so the full 88 GB does not belong
here — pull a handful of samples (including `6bba_c328f2fd` for the fixture) locally and
put the complete set on the Linux server.

**Download it as the bulk archive, not file by file.** `fetch_data.py zip` issues one
`competition_download_files` request for the whole competition and extracts it, filtered
by the same `--samples`/`--embryos`/`--per-embryo`/`--all` flags. That is not a
preference: Kaggle rate-limits the *per-file* endpoint hard, rejected requests still
count against the bucket, and 25k of them collapses to a crawl — it fails on subsets, not
just on `--all`, which is why `get --all` is now refused outright and `get` is only for
surgical top-ups. The archive path is strictly better in three ways beyond throughput:
the transfer resumes across runs via HTTP Range (ETag-validated against a
`.kaggle-partial` marker, so a stale partial is re-fetched rather than concatenated), the
archive's own central directory *is* the manifest so no throttled `competition_list_files`
paging is needed at all, and extraction is idempotent on a size check. The cost is
all-or-nothing transfer: `--samples` narrows what gets *extracted*, never what gets
*transferred*, and peak disk is 2× (archive + extracted) unless `--rm-zip` is passed.

One trap worth knowing, since it is invisible: `select()` keys off the sample stem, so
top-level files have no sample and fall out of every selection. `sample_submission.csv`
is one of them, and it is the submission-format oracle — the `zip` command therefore
re-adds root files by hand. `tests/test_fetch_data.py` pins that, along with the layout
guard that fails loudly if Kaggle ever wraps the archive in a top-level folder rather
than silently extracting nothing.

## Codebase map

| Path | Lines | Role | Expectation |
|---|---|---|---|
| `src/volt_biohub/metrics.py` | 536 | official scorer | semantics frozen |
| `src/volt_biohub/division_metrics.py` | 574 | division TP/FP/FN | semantics frozen |
| `src/volt_biohub/io.py` | 396 | zarr/geff I/O, resample, normalize | heavy rework |
| `src/volt_biohub/img_proc.py` | 187 | normalization, coord rescaling, `nms_3d` | keep, extend |
| `src/volt_biohub/gpu.py` | 218 | shared-box GPU scan/rank, device resolver | ours; keep |
| `src/volt_biohub/detection_targets.py` | 452 | Gaussian targets, PU mask, pseudo-labels, detection loss | ours; keep |
| `src/volt_biohub/linking.py` | 195 | Hungarian linking in µm, ≤2 children, graph build | ours; keep |
| `src/volt_biohub/detectors.py` | 132 | DoG and score-map front-ends behind one signature | ours; keep |
| `src/volt_biohub/scoring.py` | 165 | video-level detect→link→score, shared by train and CLI | ours; keep |
| `src/volt_biohub/models/temporal_unet.py` | 141 | `TemporalUNet3D` | main detector |
| `src/volt_biohub/models/simple_node_transformer.py` | 210 | pairwise edge scorer | late-stage option |
| `scripts/dataspec.py` | 40 | path resolution | keep |
| `scripts/fetch_data.py` | 757 | Kaggle downloader: bulk archive + per-file, resumable | ours; keep |
| `scripts/train_unet_transformer.py` | 1294 | joint training | rework or replace |
| `scripts/predict_unet_transformer.py` | 677 | inference; `--use-ilp`, `--det-threshold` (0.99) | rework |
| `scripts/geffs_to_csv.py` / `csv_to_geffs.py` | 85 / 89 | submission round-trip | wrap in a validator |
| `scripts/evaluate.py` | 142 | score pred vs GT geffs | keep, add per-video diagnostics |
| `scripts/score_pipeline.py` | 172 | score a pipeline end to end, `--detector dog\|unet` | ours; keep |
| `scripts/augmentations.py` | 63 | brightness shift + axis flips only | extend |
| `scripts/explore_data.py` | 1319 | full-dataset measurement → `reports/eda/` | B's; keep |
| `scripts/check_integrity.py` | 603 | asserts pipeline assumptions, non-zero exit | B's; keep |
| `scripts/check_t_true.py` | 149 | can `N_true` be predicted per sample | B's; keep |
| `scripts/visualize_data.py` | 704 | matplotlib figures → `reports/figures/` | B's; keep |
| `DATASET.md` | 554 | B's measured dataset write-up | see the normalization caveat |
| `visualize/` | 512 | napari GT/prediction viewers, TP/FP/FN colours | keep — debugger *and* figure source |

`scripts/` is not a package: `dataspec`, `evaluate` and `train_unet_transformer` are
imported via `sys.path` insertion (`tests/conftest.py`, `predict_unet_transformer.py`).
Fragile — anything reused beyond one script belongs in `src/volt_biohub/`.

## The pipeline as it stands

**Train.** Metadata-only load per video (no images in RAM) → one window per pair of
consecutive frames that both carry GT nodes → strided zarr read of W frames,
quantile-normalised from the zarr attrs, fp16 → `TemporalUNet3D` gives 32-channel
features plus a 1-channel detection map per frame → detection loss → local-max peaks on
the detection logits, greedily matched to GT nodes within 5 µm → U-Net features indexed
at the *detected* coordinates (gradients reach the features, not the coordinates) plus
32 dims of sinusoidal position → `SimpleNodeTransformer` bidirectional cross-attention →
an `N_t × N_{t+1}` logit matrix → focal-weighted BCE over `softmax(dim=0)`, i.e.
normalised over *sources* per target, so **one parent per child but two children per
parent: divisions allowed, merges structurally forbidden** — masked to the rows/columns
that have GT, which is where sparse supervision actually lives.

**Predict.** Sliding windows, stride `W−1` → detection with 4× flip TTA over Y/X only (Z
excluded as out-of-distribution given the anisotropy) → local-max peaks, 3 µm suppression
kernel, `sigmoid > 0.99` → edges by softmax over sources, threshold 0.5, greedy with ≤1
parent / ≤2 children → optionally replaced by the tracksdata ILP solver → coordinates
scaled back by the downsample factor, one `.geff` per video.

**Round trip.** `predict` → `.geff` → `geffs_to_csv` → `submission.csv` → `csv_to_geffs`
→ `evaluate`. That the loop closes is the main reason to start from this repo.

**Not implemented at all:** gap closing, any global registration or drift correction, any
explicit division objective, submission validation, offline Kaggle packaging.

## Known defects in the inherited code

Modelling and scoring:

- ~~`compute_detection_loss` marks a single voxel per GT cell positive and penalizes
  every other voxel at `neg_weight`.~~ **Addressed** by
  `src/volt_biohub/detection_targets.py`, now the default (`--det-target gaussian`
  with `--det-pu-mask`). The legacy behaviour is kept as `--det-target voxel` /
  `--no-det-pu-mask` purely so the change can be ablated on CV rather than believed.
  **Still open, and newly suspicious:** `--det-threshold 0.99` at inference was a
  workaround for the miscalibration this fixes, so it must be re-swept once a CV
  harness exists — and note `detect_and_match` thresholds raw **logits** at 0.3 during
  training while `predict` thresholds a **sigmoid** at 0.99. Those are not the same
  quantity and the mismatch has never been checked.
- ~~**Checkpoint selection does not use the competition metric.**~~ **Fixed 2026-09-07.**
  Kept here because the failure is instructive. `train()` selected on
  `test_acc × test_recall`, per-cell agreement on the edge matrix — which is
  overwhelmingly zeros, so **predicting no edges at all scores 0.989–0.999**, and the
  figure *rises* with detection density (750×760 detections against 7 GT edges gives
  0.9993). It read `acc=0.9943, recall=0.9408` after **20 iterations** and froze at
  epoch 2 of 8 while the detection loss halved, discarding six epochs of real gain.
  Selection now runs the competition scorer over a subsample of held-out *videos* via
  `volt_biohub.scoring.epoch_selection_score`; after five iterations that reads **0.1453**,
  i.e. it has dynamic range. Tunables: `--select-videos` (default 8) and
  `--select-budget`. The old edge-matrix `evaluate()` survives for display only and is
  capped by `--legacy-eval-batches` (default 20) because walking the full test split cost
  ~13 of every 30 minutes per epoch on the heavy fold.
  **Still true and still important: do not read the training console's `acc` column as
  evidence about anything.**
- **The two knobs the metric says matter most are unreachable from the CLI.**
  `PredictConfig` holds `pool_kernel_um` (suppression radius), `threshold` (edge
  probability), `edge_activation` and `det_tta`, but `main()` forwards only
  `--det-threshold` and the ILP weights. Sweeping today means editing the dataclass.
- **The edge threshold is not scale-free in detection count, and this blocks the
  detection sweep.** `predict` takes `softmax(logits, dim=0)` — normalised over
  *sources* for each target — and keeps pairs above a **hardcoded 0.5**. So emitting any
  edge at all requires one source to hold over half the column mass, and the number of
  competitors it must beat is the detection count. Measured on one real frame pair,
  varying `--det-threshold` alone: **753 → 117 → 48 → 42 → 34 candidate sources** for
  thresholds 0.3 / 0.5 / 0.9 / 0.99 / 0.999. A 20× swing in the denominator with the
  numerator's threshold fixed means **`--det-threshold` and the edge threshold are one
  coupled knob, not two** — and since metric fact 2 pushes detection density *up*, the
  coupling works against the score. `--det-threshold 0.99` is therefore load-bearing for
  the **linker**, not only a calibration workaround for the detection loss; the two
  defects are one defect. Fix by exposing `threshold` and making it scale-aware (or
  scoring per-target top-1 with a margin instead of an absolute probability) *before*
  sweeping detection, or the sweep measures the threshold artifact rather than the
  detector.
- **`--det-neg-weight` defaults to 1e-2 on the CLI but 0.1 everywhere else** —
  `detection_loss` and `compute_detection_loss` both document and default to 0.1, and
  `train_epoch` signs 0.1, while `train()` signs 1e-2 and the CLI passes it. So the
  as-run negative term is **10× weaker than the loss was designed for**, giving a 100:1
  positive:negative weight ratio before the PU mask removes a further 1.9–49 % of the
  negatives. A detector that is barely penalised for firing saturates high, which is a
  second and independent reason inference needed `sigmoid > 0.99`. Pass it explicitly
  and sweep it; do not trust either default.
- **`predict()` cannot enumerate a directory** — it needs a `dataset_splits.json` with a
  `"test"` list, which the competition test folder has none of. Blocks full test-set
  inference until fixed.
- Isotropic ↔ original-anisotropic round-trips are the highest-risk code here: wrong by
  4× in z, no exception raised, score near zero. Anything crossing that boundary needs a
  round-trip test.

Hygiene, each capable of wasting a day:

- **Three disagreeing defaults per knob.** `det_loss_weight`: `train_epoch` signature
  0.1, `train()` signature 1e1, CLI 1e0 with help text claiming 1e1. Same shape for `lr`
  (1e-3 vs 1e-4) and `num_workers` (4 vs 8 vs help saying 4). Always pass explicitly.
- ~~**No `--seed`.**~~ **Fixed 2026-09-07** — `--seed` now reaches `train()` *and* seeds
  torch, numpy and `random`, not only the DataLoader generator as before. That omission
  is why the gaussian-vs-voxel ablation was uncontrolled: model init, dropout and the
  augmentation draws all came from the global RNGs, so two "identical" runs differed.
  **Pass it on every run**; a config without a seed is not an experiment.
- `--method baseline` in the README/commands matches nothing — `DEFAULT_METHOD` is
  `unet_transformer`, so it looks for `weights/baseline/split_0/edge_predictor_best.pth`,
  which nothing writes.
- ~~`dataset_splits.json` is silently auto-generated as a seed-0 90/10 split when
  absent, and it is a *video* split, not embryo-grouped.~~ **A deliberate one now
  exists** at `$VOLT_DATA_DIR/dataset_splits.json` (not in git — it lives beside the
  data): split **0** = train `44b6` → test `6bba` (the fold carrying 85 % of metric
  weight), **1** = the reverse (noisier, 15 % of weight), **2** = within-`6bba` FOV
  split for fast iteration, **3** = a 4-train/2-test smoke fold for checking a command
  before committing hours. Still open: it is a throwaway script, not the
  `scripts/make_splits.py` that step 2 of the Plan asks for, so it is not reproducible
  from the repo.
- **`fetch_data.extract()` swallows batch failures, and there is an intermittent
  extraction bug behind it.** A batch that throws is appended to `failures`, *printed*,
  and the function returns normally — so a partial extraction of the 24,886-file dataset
  looks like a success. Mitigated only by the size-based idempotency check on a re-run
  and by `fetch_data.py status`. Separately,
  `test_fetch_data.py::test_zip64_archive_reports_real_sizes` fails intermittently at its
  last assertion — one member (the 55-byte train chunk) still reports `_needs_fetch`
  after a 2-worker extract. **Measured rate: 1 of 6 full-suite runs, 0 of 12 runs of that
  file alone**, so it needs the rest of the suite in the process first. Not disk space
  (6.6 TB free). Not diagnosed further; `extract` returning success while files are
  missing is the part worth fixing first, because it is what makes the failure silent.
- Both `train` and `predict` **raise** if a zarr lacks `image_statistics.quantiles`. The
  real data does carry it on both train and test (see Data), so this is latent rather
  than active — but it is a hard crash, not a fallback, if the hidden test set ever
  differs.
- `open_dataset(device="cuda")` is the **default** → crashes CPU-only. Needs a device
  resolver. Its eager `da.from_zarr(...).compute()` loads a whole video, which at ~839 MB
  is survivable; `load_image=False` gives metadata only.
- Detections are stored as `int16` (`predict_unet_transformer.py:293,496`). Fine for
  256³-scale volumes, silently wrong if a future sample exceeds 32767 in any axis.
- **`train()` wraps the UNet in `nn.DataParallel` across every visible GPU**, which on
  the shared training box means all 8 — including cards other tenants have filled. The
  slow test dies in `torch/nn/parallel/comm.py` with `CUDA error: out of memory` while
  scattering, even with 47 GB free on device 0, and the message names memory rather than
  the real cause. It passes with a single pinned card. **`--single-gpu` already turns it
  off** (the flag exists; the earlier note here claiming no such argument was wrong), but
  it defaults **on**, so **every run on that box still needs both
  `CUDA_VISIBLE_DEVICES=$(volt_gpu)` and `--single-gpu`**. DataParallel is the wrong tool
  at this model size (1.5 M params, 3.4 GiB per training step), so the fix is to drop it
  and flip the default rather than keep parameterising it.
- **`_stamp` dropped GT nodes at the far edge of the volume** — fixed 2026-09-02, kept
  here as the shape of bug to watch for. `get_window_data` divides raw coords by the
  downsample factor as floats, so at `1,4,4` original `y=255` becomes `63.75` on a grid
  of extent 64; `.round()` sent it to 64, outside the volume, so no positive target was
  stamped and `n_pos` silently undercounted. **225 of 133,318 real GT nodes, up to 7.2 %
  on `44b6_c96cfa10`** — and since a missed node costs two edges, ~450 edges. The legacy
  loss floored and clamped, so it kept them: the bug made our own ablation unfair
  against us. `_snap_centres` now snaps the sub-voxel overhang and still drops centroids
  genuinely outside the window (the legacy clamp-anything behaviour was its own bug,
  stamping false positives on the border). Two regression tests pin both halves.

## Commands

```powershell
uv sync                                    # add --extra viz for napari
uv run pytest -m "not slow"
uv run ruff check .
uv run python scripts/fetch_data.py zip --split both --all      # bulk; see Data placement
uv run python scripts/fetch_data.py status                      # progress, any terminal
uv run python scripts/predict_unet_transformer.py --split 0
uv run python scripts/evaluate.py --pred-dir <dir> --gt-dir $env:VOLT_DATA_DIR
uv run python scripts/geffs_to_csv.py --in-dir <dir> --csv submission.csv
uv run python visualize/visualize_predictions.py --split 0
uv run python scripts/explore_data.py --data-root <parent of train/> --out reports/eda
uv run python scripts/check_integrity.py --data-root <parent of train/>
```

Note `--method` is omitted above: `DEFAULT_METHOD` is `unet_transformer` and
`--method baseline` matches nothing that writes weights. The EDA scripts want the
*parent* of `train/`, not `$VOLT_DATA_DIR` itself, and do not read `VOLT_DATA_DIR`.

Same on Linux with `$VOLT_DATA_DIR` and forward slashes. On the server the environment
comes from `. /home/y2a/.volt-env.sh` (already sourced by `.bashrc`/`.profile`, and
re-wired by `startup.sh` after a container rebuild). **Every GPU run needs a pinned card
*and* `--single-gpu`**, or DataParallel scatters onto cards other tenants have filled:

```bash
CUDA_VISIBLE_DEVICES=$(volt_gpu) uv run python scripts/train_unet_transformer.py \
    --split 3 --epochs 4 --single-gpu --batch-size 8 --num-workers 8 --max-iters 40
```

Split 3 is the smoke fold (~70 s/epoch); use split 2 for real iteration and 0/1 for the
LOEO number. `volt_gpu` is a console script declared in `pyproject.toml` — it needs a
`uv sync` after a fresh clone, and `python -m volt_biohub.gpu` is the fallback.

## Status

Renamed `tracking_cellmot` → `volt_biohub`, `CELLMOT_DATA_DIR` → `VOLT_DATA_DIR`.
Windows torch on the cu130 index, CUDA verified with `sm_120` present. Previously
undeclared runtime deps (numpy, polars, dask, geff) now declared; ruff in the dev
group. `.gitignore` / `.gitattributes` in place.

Kaggle auth works and `scripts/fetch_data.py` has cached the full 24,886-file manifest to
`C:\data\biohub\manifest.csv` (the `zip` path no longer needs that step — the archive
carries its own manifest). A 7-sample dev subset (`6bba_c328f2fd` for the fixture plus
3 per embryo) is on the dev box. **`pytest -m "not slow"` is fully green: 122 passed** —
the scorer is verified against real data, which closes the "do we trust the scorer"
question. The whole inherited pipeline has been read end to end (see "The pipeline as it
stands" and "Known defects").

**Linux server provisioned and loaded 2026-09-02** (see Environments for the measured
facts). `uv sync` clean, `torch 2.13.0+cu130` sees all 8 GPUs, every script imports, and
`TemporalUNet3D` trains: 1.5 M params, **3.4 GiB peak for a `(1, 5, 1, 64³)` step**, so a
48 GB card is nowhere near the constraint and neither is Kaggle's 16 GB.

**The full dataset is on the server: 203/203 samples, 81.59 GiB, 24,885 files, verified
complete against the archive manifest.** Downloaded via the new `fetch_data.py zip` path
(the per-file endpoint was rate-limiting even on subsets, so the downloader was reworked
around one bulk request — see Data placement). `pytest -m "not slow"` is **131 passed, 0
failed** there, which closes "do we trust the scorer" on this machine too. The 9 new
`tests/test_fetch_data.py` cases cover archive selection, root-file handling, ZIP64 size
resolution and idempotent extraction against a synthetic zip — no network, no competition
data.

**The data has been swept end to end and it moved several numbers in Data**, so re-read
that section rather than trusting older recollections. Headlines: all 203 samples are
dimensionally identical and all carry quantiles; GT is ~3.5 % annotated at the median
rather than 0.2–1.6 %; `44b6` carries only 15 % of the metric weight so the two LOEO
folds are not comparable; a 5 µm linking gate would forfeit 6.3 % of real edges; cell
pitch is a 11–31 µm distribution rather than ~15 µm; and **`estimated_number_of_nodes`
does not exist at inference**, which bounds how the node-count knob can ever be tuned.
GT integrity is clean across all 199 files (one suspect 60 µm z-jump in `6bba_f20478e9`).

`pytest -m "slow"` — **run to completion for the first time (47 min, one pinned card)
and it FAILS, on real behaviour rather than environment.** Two separate issues came out
of it. First the environment one: it DataParallels across all 8 visible GPUs and OOMs on
cards other tenants have filled, which is why pinning is mandatory (Known defects).
Pinned, it then trains to a genuine overfit — `acc=1.0000, det=0.0001, edge=0.0000` —
and passes `edge Jaccard >= 0.95` (it gets **1.0000**) and `node_recall == 1.0`, but
**fails `division_jaccard == 1.0` with 0.5**, having predicted 28 nodes against 22 GT.
That is a spurious fork on an over-detection, and it is the evidence behind the
over-detection/division trade-off now recorded under metric fact 2. Note the documented
command is `pytest -m "not slow"`, which deselects this test — so its status had simply
never been established, and this is a pre-existing failure rather than a regression.
**The baseline cannot reliably get divisions right even while memorising a single clip
that contains one**, which is a sharper argument for fixing detection first than any of
the code review.

`uv run ruff check .` reports **69 pre-existing errors** (mostly `visualize/` and
`train_unet_transformer.py`: unsorted imports, `zip` without `strict=`, trailing
whitespace). Inherited, not ours, and concentrated in code slated for rework — but the
lint gate is not green today, so don't read a clean run as a regression signal yet.

**THE FIRST REAL CV NUMBER IN THE PROJECT, 2026-09-07 — and it comes from no learning at
all.** `scripts/score_pipeline.py --detector dog`: Difference-of-Gaussians detection at
the measured σ, Hungarian assignment on µm distance, scored with `summarise`. Untuned —
every parameter read off a measurement, nothing fitted to the fold.

| LOEO fold | test embryo | n | score | edge J | div J | node recall | node ratio |
|---|---|---|---|---|---|---|---|
| 0 | `6bba` | 128 | **0.6424** | 0.6371 | 0.0149 | 0.7775 | −0.096 |
| 1 | `44b6` | 71 | **0.4030** | 0.3873 | 0.0118 | 0.5927 | **−0.371** |

Edge-weight-weighted: **0.6056**. This is the floor. Any learned component that does not
beat it is not earning its complexity.

**The fold gap is a detector ceiling, and it says exactly what to fix.** The two embryos
are opposites: `6bba` is 97 cells/frame with 9.7 % annotated, `44b6` is **327
cells/frame** with 0.77 % annotated. On the crowded one the classical detector emits
**37 % fewer nodes than the budget asks for** — it *cannot fill its quota*, because DoG
maxima at a 5.86 µm separation merge before the budget is reached at a ~15.9 µm true
pitch. Node recall collapses to 0.59. So:

- the node-budget knob is not even *reachable* on `44b6`, which makes budget sweeping
  pointless there until detection improves;
- this is the one place a learned detector has an obvious, measurable job — separating
  crowded nuclei that a fixed-σ blob filter merges;
- and it re-reads the 15 %/85 % weight asymmetry as *good* news: the fold we are worst on
  is the one that counts least.

**Two full-scale learned models, 2026-09-07 — and the classical floor beats both.** Both
12 epochs × 200 iters on LOEO fold 0, batch 16, lr 1e-3, seed 0, `det_neg_weight 1e-1`
(the designed value, not the CLI's 1e-2), scored through the *same* harness as the
classical run. `experiments.csv` carries every row.

| run | score | edge J | div J | node recall | node ratio |
|---|---|---|---|---|---|
| **classical DoG** | **0.6424** | 0.6371 | 0.0149 | 0.7775 | −0.096 |
| model A — Gaussian + PU | 0.5886 | 0.5871 | 0.0069 | **0.8505** | −0.016 |
| model B — + pseudo-labels | 0.5923 | 0.5910 | 0.0057 | 0.8497 | −0.015 |

Three things came out of this, in ascending order of importance.

**(a) The label-quantity question is answered, and the answer is "not the bottleneck".**
Model B supervises the unannotated 96.5 % with classical pseudo-labels instead of merely
ignoring it; it bought **+0.004**. It did stabilise training — A peaks at epoch 0 and
decays to 0.525, B holds 0.541–0.549 across all 12 epochs — but the achievable level did
not move. Put beside gap closing being worth ~+1.4 % more edges, **both label levers are
weak**. Re-run B's arm when the gap-closed labels land and compare to its
`experiments.csv` row; that is a clean one-variable test with the counterfactual already
banked.

**(b) The detection loss is anti-correlated with the score on this architecture.** Model
A's detection loss fell **4.4×** (0.0416 → 0.0094) while its CV score fell 0.5461 →
0.5248. Best epoch is **0**. That is a loss-specification problem, not a data or capacity
problem, and it is the single most important open item — training longer currently makes
the model worse.

**(c) THE FIX: tighten the linking gate, and the U-Net overtakes the classical floor.**
Since better nodes expose more links to scoring (metric fact 3), a detector with higher
node recall wants a *stricter* linker. Swept on 30 samples of fold 0, model A, refined
peaks, everything else held:

| config | score | edge J | div J | node recall |
|---|---|---|---|---|
| U-Net, gate 12.0 µm (the motion-derived default) | 0.5849 | 0.5834 | 0.0071 | 0.8601 |
| U-Net, gate **8.0 µm** | 0.6672 | 0.6656 | 0.0058 | 0.8601 |
| U-Net, gate 5.0 µm | 0.6143 | 0.6132 | 0.0020 | 0.8601 |
| **U-Net, gate 8.0 µm, division pass OFF** | **0.6925** | 0.6914 | 0.0000 | 0.8601 |
| classical DoG, gate 8.0 µm | 0.6614 | 0.6572 | 0.0076 | 0.7959 |
| classical DoG, gate 12.0 µm | 0.6435 | 0.6391 | — | 0.7959 |

Node recall is identical down the U-Net rows — every one of these gains is the linker
discarding links it should never have made. Three separate results:

- **The 12 µm gate was too generous for *everyone*.** DoG gains +0.018 at 8 µm too. It
  was derived from ordinary motion (p99.9 = 13.8 µm) without accounting for the FP
  exposure of fact 3, which is a reasoning error, not a tuning miss.
- **But the coupling is real: the U-Net gains +0.082 against DoG's +0.018.** The stronger
  detector is the one punished by a loose gate, exactly as fact 3 predicts. **Sweep the
  gate per detector and never carry one detector's gate to another.** The optimum is
  interior — 5 µm starts severing real edges, and division Jaccard collapses to 0.0020
  because the parent-to-far-daughter median is 7.13 µm.
- **The division second pass is a net loss of ~0.025 and should be off by default.**
  Turning it off raises edge Jaccard 0.6656 → 0.6914 while giving up 0.0058 of division
  Jaccard — which at the 0.1 weight is worth **0.00058 of score**. It pays 0.026 to buy
  0.0006. That is metric fact 2's over-detection/division trade running in reverse:
  forcing a second child manufactures edge FPs far faster than it finds real forks.

**Best configuration measured to date: U-Net + 8 µm gate + no division pass = 0.6925**,
against the classical floor's 0.6424. The learned detector does win — but only once the
linker is matched to it, and none of that was visible before a CV harness existed.

**(d) The U-Net detects better and tracks worse, for a reason that is in the metric.**
See metric fact 3: its 2.3× edge-FP excess is the price of its higher node recall
exposing more links to scoring. **Two plausible explanations were tested and refuted**,
which is why the FP decomposition was worth doing rather than guessing a third time:

- *Plateau jitter.* The saturated map puts each cell on a plateau whose winning voxel
  could move between frames. Response-weighted centroid refinement
  (`detectors.refine_centres`) moved the median link 2.30 → 2.10 µm but left p90 at 8.75
  µm untouched — so the long links join *different cells*, not the same cell wobbling.
- *Marginal over-detection.* Dropping the budget makes it monotonically **worse** (1.0× →
  0.585, 0.8× → 0.582, 0.6× → 0.525, 0.4× → 0.401), so the extra detections earn their
  keep. Note DoG cannot even fill the budget (ratio −0.096) while the U-Net can (−0.016).

**Gaussian vs legacy-voxel, first head-to-head on real held-out data — and it does not
settle the question, but it relocates it.** Identical budget (split 3, 8 epochs, 60
iters, batch 8, lr 1e-3, one pinned card, ~70 s/epoch), then detector calibration
measured on the two held-out videos at 5 timepoints each (62 GT centroids, paired
random-voxel sample):

| cut | gaussian+PU: GT kept / random voxels | legacy voxel: GT kept / random voxels |
|---|---|---|
| 0.3 | 100.0 % / **20.4 %** | 100.0 % / **27.9 %** |
| 0.5 | 98.4 % / **13.7 %** | 100.0 % / **20.4 %** |
| 0.9 | 98.4 % / 5.1 % | 98.4 % / 5.2 % |
| 0.99 | 95.2 % / 2.3 % | 96.8 % / 2.1 % |

- The precision edge to the Gaussian loss at low cuts is on a large paired background
  sample and is probably real; the GT-recall column is 62 points and is noise. **But
  there is one run per arm and `--seed` does not reach the CLI**, so init and data order
  are uncontrolled — per our own experiment discipline this is one sample of a noisy
  process, not a result. Do not cite it as the ablation.
- **What it does settle is that neither target shape nor PU masking is the dominant
  problem.** Both detectors are *saturated*: median sigmoid at GT centroids is 1.0000
  and the random-voxel p99 is ~0.999. At cut 0.99 — the inference default — **~6,000
  voxels per frame are still above threshold**, against a median true density of 179
  cells/frame and 786 in the densest sample. The detection map carries almost no usable
  ranking information, which is why NMS radius and threshold are doing all the work.
- The cause is not mysterious: `--det-neg-weight 1e-2` (see Known defects) makes the
  negative term 100× weaker than the positive before the PU mask removes a further
  1.9–49 % of it. **So the order is: fix checkpoint selection, sweep `det_neg_weight`,
  then sweep `bright_threshold`, and only then ask whether the target shape matters.**
  Ablating target shape underneath a saturated detector measures nothing.
- `acc × test_recall` ranked *legacy* higher (0.9451 vs 0.9377), and in the Gaussian run
  it froze at epoch 2 while `det` loss improved for six more epochs. Further evidence
  that number cannot be used for selection or comparison.

**PR #2 (B's EDA) is merged into `main` as of 2026-09-02** and supersedes the
`inspect_data.py` item above: `explore_data.py`, `check_integrity.py`, `check_t_true.py`
and `visualize_data.py` make the whole sweep reproducible, plus `DATASET.md`. Verified by
re-running all three measurement scripts against the full 199 samples on the server —
every headline number reproduces exactly (11 s for `explore_data.py`, 3.7 s and
0 FAIL / 7 WARN for `check_integrity.py`). Three of its findings changed this file:
`N_true` *is* predictable from test-time metadata (metric fact 4), sister separation caps
the suppression radius at ~4 µm rather than the ~18 µm cell pitch (metric fact 5
consequences), and the median parent→far-daughter displacement of 7.13 µm exceeds the p99
of ordinary motion, so a learned motion prior scores the second division edge as
implausible by construction. Still to fix in that PR: the four EDA-only packages
(`kagglehub`, `python-dotenv`, `pandas`, `matplotlib`) landed in runtime `dependencies`
rather than an extra — that list is what gets bundled as offline wheels for Kaggle — and
`kagglehub`/`python-dotenv` are used by nothing. B is fixing the `--data-root` defaults.

**Identity trap, worth knowing before the next PR:** GitHub attributes commits by *email*,
and `240247@buas.nl` / `240843@buas.nl` are verified on the university accounts
(`OleksiiKrasnoshtanov240247`, `FilippLotsmanov240843`), not on `Gfgf96` /
`filipp-lotsmanov`. PR #3's four commits were rewritten to
`145783662+Gfgf96@users.noreply.github.com` and `.git/config` now pins it — local scope
deliberately, since `/root` is the container's ephemeral overlay.

Not done: `scripts/make_splits.py` as a committed script (a deliberate
`dataset_splits.json` now exists beside the data, but from a throwaway script), any
modelling with a CV number attached, Kaggle team.

## Plan

**Gate 0 — the team is the only thing left that needs a human.** Dev subset downloaded,
`pytest -m "not slow"` green at 122 passed on the dev box; the Linux server is
provisioned, its environment questions are all answered, and the full dataset is
downloading. Remaining: **form the Kaggle team before Sept 22** — 20 days out, and
nothing recovers a miss.

1. **Measure before designing — substantially done, needs committing.** The sweep over
   all 199 samples is complete and its findings are folded into Data (counts vs
   `estimated_number_of_nodes`, pooled displacement percentiles, single-lineage count,
   sparsity by embryo, per-sample metric weight). What remains is turning it into
   `scripts/inspect_data.py` so it is reproducible and re-runnable, emitting a per-sample
   CSV — that table is the input to the split design in step 2 and to the node-budget
   sweep in step 7, and it is also a portfolio artifact. Still genuinely open from this
   step: **residual displacement after global registration** (the number that decides
   whether step 6a is worth anything), and the nucleus radius σ that step 5's Gaussian
   heatmap targets need.
2. **CV harness** — `scripts/make_splits.py` emitting **both** splits described in Data:
   the 2-fold leave-one-embryo-out (the honest number) and a within-embryo FOV split (the
   fast one). Every experiment scored through it, one row per run in `experiments.csv`,
   both numbers recorded.
3. **First submission, classical** — DoG local maxima + Hungarian on µm distance,
   gated at ~2× median displacement, no divisions. The point is a scored end-to-end
   chain and a floor. Teams that skip this find their CSV bugs on Sept 28.
4. **Offline submission chain** (parallel with 3) — a directory-mode `predict`,
   `validate_submission()` diffed against `sample_submission.csv`, wheels/code/weights
   datasets, thin notebook. Confirm the LB score matches local CV. **A CV/LB
   disagreement here is the most important bug in the project;** chase it down before any
   modelling. Also time one video end to end and extrapolate to a train-sized hidden test
   set against the 12 h cap.
5. **Detection (week 2) — REWRITTEN 2026-09-07 now that CV numbers exist.** The
   priority order below changed because measurement contradicted the previous ordering;
   see Status for the runs. In order:
   **(a) sweep the linking gate per detector.** Worth **+0.082** on model A from one
   scalar, and it is what takes the learned detector past the classical floor. Nothing
   else measured today comes close to that return. Then re-sweep the budget *jointly*,
   since detector and linker are coupled through the FP-exposure mechanism in metric
   fact 3.
   **(b) find out why the detection loss is anti-correlated with the score.** Model A's
   loss fell 4.4x while its CV fell; best epoch is 0. Until that is understood, longer
   training makes the model worse and every detector experiment is fighting it. Score
   per-epoch checkpoints, not just the best one — the harness makes this cheap.
   **(c) sweep `det_neg_weight` and `bright_threshold`**, still unswept and still the
   cheapest remaining detector knobs.
   **(d) drop or fix the learned linker.** Deprioritised on geometry, then partly
   rehabilitated: precision 0.755 says the *atypical* cases are where the score now
   leaks, which is what a learned edge cost is for. But it emits zero edges today, so
   fix the gate sweep first and revisit only if (a)–(c) plateau.
   **Not a priority: more labels.** Pseudo-supervising the unannotated 96.5 % bought
   +0.004, and closing every joinable annotation gap is worth ~+1.4 % more edges.

   *Previous framing, kept because the reasoning is still sound:* the loss is written,
   the evidence is not.
   `src/volt_biohub/detection_targets.py` implements Gaussian targets at the measured
   sigma plus positive-unlabeled masking, defaulted on, with `--det-target voxel` /
   `--no-det-pu-mask` retained so each half can be ablated. All three paths train and
   detect on the fixture. **What is missing is the only thing that matters: a CV number.**
   The fixture is a 5-frame overfit and proves nothing about generalisation, so this
   stays open until step 2 exists and the ablation runs through it. Still to do here,
   in the order that unblocks the most:
   **(a) switch checkpoint selection to the real scorer** — this is now the gating item,
   not a tidy-up: `test_acc × test_recall` reads 0.9943/0.9408 after *20 iterations*
   (see Known defects), so nothing printed during training can rank a loss variant, and
   until it is replaced the ablation cannot be run at all;
   **(b) sweep `bright_threshold`** — measured to over-mask by 4.5× at the median while
   correctly tracking density at ρ +0.87, so it is one scalar with a known direction
   (raise it), the cheapest open knob here;
   **(c) re-sweep `--det-threshold`** now that calibration has changed, reconciling the
   logit-vs-sigmoid mismatch noted under Known defects.
   Fallback if the UNet
   will not converge in budget: Spotiflow 3D, which trains natively from point labels
   and is legal here (pre-trained models are explicitly allowed).
6. **Linking (week 3)** — note (a) below is now partly superseded: the 12 µm gate was
   derived from motion percentiles and turned out to be 4 µm too generous once the
   detector improved, so *gate sweeping* moved into step 5a. The rest stands, in
   priority order: (a) **global frame-to-frame registration
   before association** — bulk motion removed makes a dense assignment easy, and most
   teams skip it; temper the expectation, though, since these are ~104 µm crops rather
   than whole embryos, so measure the residual displacement in step 1 before investing.
   (b) candidate graph + ILP (`tracksdata.solvers.ILPSolver`, already wired as
   `--use-ilp`), weights tuned on CV and never on the leaderboard. (c) gap handling by
   interpolated node insertion. (d) learned edge costs (`SimpleNodeTransformer`) last,
   only if a–c plateau.
7. **Freeze week (Sept 22–29)** — tune total node count (metric fact 4), full
   test-set timing on Kaggle hardware, ≥2 submissions banked before the final day,
   nothing new on Sept 29.

Rejected, so they don't get re-proposed: **Trackastra** needs instance masks we don't
have, and synthesizing spheres from centroids discards the appearance features that
make it strong. **Ultrack** duplicates in-repo tracksdata functionality and adds
offline-install burden.

**Division of labour.** A: data inspection, CV harness, detection. B: CSV writer +
validator + offline chain, linking + ILP tuning, LB probes. Both: local scorer from
day one, branch per experiment, `main` always able to produce a valid submission.

## Engineering standards

- **Simple, direct, coherent code over clever code.** Extensible, but no speculative
  abstraction — build for the requirement in front of us in a way that can grow.
- No ideological FP/OOP preference; say why when the choice isn't obvious.
- Comment the **design rationale**, not the mechanics. `# 7 µm tolerance means
  sub-voxel accuracy is worthless here` earns its line; `# loop over frames` doesn't.
- **Verify library APIs against current docs before writing against them** —
  `tracksdata` is pre-1.0 and moving, zarr 3.x differs from 2.x.
- **Read the relevant files before proposing changes**; never infer structure from
  filenames. For changes touching more than two files, propose the plan first.
- Don't create README or summary documents unless asked. This file is the exception
  and should be kept current — especially **Status** and **Plan**, which a fresh
  session reads to know where we are. Delete fixed items from "Known defects"; if a
  metric claim turns out wrong, correct it here first.
- **Flag likely bug sources as part of the answer**: shared mutable state, unhandled
  error paths, silent failures, encoding assumptions. Here specifically: coordinate
  spaces, in-place graph mutation, CV/LB disagreement.
- Say why an approach was chosen and what the realistic alternatives were, including
  rejected ones. Say plainly when something is a genuine trade-off.
- Direct tone. No preamble, no flattery, no emojis. Correct me plainly and early.

**Experiment discipline.** Every run takes a config (dict or dataclass), saves it
next to the checkpoint, and appends one row to `experiments.csv`: timestamp, git SHA,
machine, config hash, CV score, edge Jaccard, division Jaccard, node count ratio,
notes. One variable at a time, or say so in the notes. A CV improvement below
run-to-run noise is not an improvement — establish the noise level early by running
one config twice with different seeds. Keep the LB for confirmation, not search.

**Portfolio artifacts, gathered as byproducts** (cheap continuously, expensive
reconstructed): `experiments.csv` itself; the metric analysis above, written up as "how
we reverse-engineered the objective and let it drive the design"; TP/FP/FN frames or GIFs
from `visualize_predictions.py` at each milestone; a CV-vs-LB scatter; an ablation table
that falls out of `experiments.csv`. README last. After Sept 29, flip the repo public and
publish the metric analysis as a Kaggle notebook or discussion post.

---

*Baseline code derived from `royerlab/kaggle-cell-tracking-competition`
(BSD-3-Clause). See `LICENSE`.*
