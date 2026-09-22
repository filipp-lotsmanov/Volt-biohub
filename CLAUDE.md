# Volt-biohub

Kaggle **Biohub – Cell Tracking During Development**: zebrafish embryo, 3D+time
light-sheet microscopy. Detect cells per timepoint, link them across time,
reconstruct lineages including divisions.

Two goals, score first where they conflict: **top-7 leaderboard placement**, and a
repo that shows engineering judgement. The write-up is assembled in the final week
from artifacts accumulated along the way — don't stop to write essays mid-competition.

**Deadlines** (all 23:59 UTC). Started 2026-06-29. 2026-09-22 was the entry and
team-merger deadline; **the team ("Volt") exists, so that gate is closed** — stop
raising it. **2026-09-29: final submission.** Features freeze ~Sept 22.

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

1. **Divisions are worth 10%, and we were scoring ZERO of it.** The old advice here
   — "don't optimize mitosis until edges plateau" — cost us the entire term: our first
   leaderboard submission contained **0 forks in 73,130 edges**, because Hungarian
   assignment is one-to-one and `division_gate_um` was None. The arithmetic that
   matters: our measured edge ceiling at node recall ≥0.98 is adjJ **0.848**, so
   `0.848 + 0.1·divJ` gives 0.848 at divJ 0, **0.898 at divJ 0.5** (the leaderboard
   median is 0.902) and **0.948 at divJ 1.0** (the top-10% line is 0.946). The field's
   score distribution is *explained* by the division term. It is not a tiebreaker.
   **The gate alone is not the lever, but it WAS blocking the candidates** — opening
   it from 6 µm to 10 µm in `linking.py` doubles true divisions found (3→6) and *lowers*
   the score, drowned by 390 false forks, so the binding constraint is the **ranking**
   of fork candidates. That conclusion stands and is incomplete: at a 6 µm gate against
   a measured GT sister separation of **median 10.4 µm**, most real divisions were never
   proposed at all, so the 1:556 base rate measured on 2026-09-12 was partly an artefact
   of the gate. `postprocess.add_safe_divisions` opens the gates (parent 9.0, sister
   14.0) *and* replaces the geometric ranker with structural admission — mutual-nearest
   orphans, forward divergence, a symmetry test, hard per-frame and global caps — and
   reaches **divJ 0.1077** on the full fold against 0.0110. Both halves were needed.
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
  resampled space. **Do NOT reach for `io.rescale_graph_to_original` here** — an
  earlier version of this line recommended it and that was wrong, in the highest-risk
  direction. It inverts an *interpolating resample* (`open_dataset(target_scale=…)`,
  `img_proc.rescale_coords_to_isotropic`) by dividing out that resample's zoom. Our
  `1,4,4` path never resamples — it is a **strided read**, so the exact inverse is
  `coord_original = coord_grid * downsample` and there is no scale arithmetic to
  undo. Using the scale-based function on the strided grid divides z by 4 *and*
  leaves y/x at a quarter of their true value: wrong on both axes, silently.
  `src/volt_biohub/submission.py` validates against this, and
  `tests/test_coordinate_roundtrip.py` is the reference for which inverse pairs with
  which forward transform.
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
for Python, `.python-version` pinned to **3.11**. That pin used to be justified as
"matches the Kaggle image" and **that is no longer true — the image is Python
3.12.13** (measured 2026-09-11 by probe kernel). Our source is clean on both, so the
pin itself is harmless; the offline wheels are what must track the image. The 5070 is Blackwell (`sm_120`) and needs CUDA ≥ 12.8, but PyPI's Windows
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

**How a code competition is actually submitted** (researched 2026-09-11; we had been
reasoning from the competition page alone and were wrong three ways). You submit a
**notebook version**, and Kaggle re-executes it privately with the competition data
**replaced by the hidden set**, then extracts and scores the file it wrote.
`competitions submit -f submission.csv` names *the kernel's output file* — the local
CSV is never uploaded, though the flag is spelled exactly like the upload flag for
ordinary competitions. The run you watch after `kernels push` uses the four visible
placeholders, so **timing must always be extrapolated**, and **the graded rerun's log is
not visible at all**. Crucially, **errored submissions still count against the daily
limit** (five/day) — Kaggle does this so a failure cannot be used to probe the hidden
data, which makes a one-minute probe kernel much cheaper than a guess. Full procedure,
the four failure modes and the checklist: `kaggle/README.md` → *How code competitions
actually work*.

**Kaggle notebook — four facts measured on the real image 2026-09-11, three of which
contradicted what this file stated.** Python is **3.12.13**, not 3.11. The competition
mounts at **`/kaggle/input/competitions/<slug>/test`**, not `/kaggle/input/<slug>/test`
— the one-level glob matched nothing and killed the first kernel run on
`StopIteration` in its opening seconds. Preinstalled: torch 2.10.0+cu128, numpy 2.0.2,
scipy 1.16.3, polars 1.35.2, and **no zarr**, which is what the wheels dataset is for.

**And the one that would have cost a whole submission cycle: `enable_gpu: true` hands
out a Tesla P100 at compute capability `sm_60`, while the image's torch ships cubins
for `sm_70` upward.** `torch.cuda.is_available()` returns **True** there and every
convolution then raises "no kernel image is available for execution on the device".
An availability check is not merely insufficient here, it is *confidently wrong* — the
same trap as `open_dataset(device="cuda")`, one level deeper. Two defences, both
needed: request `"machine_shape": "NvidiaTeslaT4"` (sm_75) in `kernel-metadata.json`,
**and** keep `notebook.py::gpu_is_usable()`, which executes one real `conv3d` and
degrades to DoG if it raises — Kaggle substitutes hardware when a shape is
unavailable, so the request alone is not a guarantee. **Probe the image before
trusting a run**: `kaggle/README.md` → *Probing the image* has the throwaway kernel,
it costs a minute, and it settled all four.

Internet **disabled**, ≤12 h, output `submission.csv`.
`tracksdata` is **not on PyPI**, so **keep the inference path free of tracksdata** —
numpy/scipy/torch/zarr plus our own CSV writer — and use tracksdata only locally for
GT parsing and scoring. Since `.geff` is plain Zarr v3 (see Data), this costs little.
That removes the largest single submission-failure risk. If ILP is needed at inference,
`ilpy` + `pyscipopt` ship manylinux wheels and bundle fine. Offline install:
`pip download <pkgs> -d wheels --only-binary=:all: --platform manylinux2014_x86_64
--python-version 3.12`, then `pip install --no-index --find-links=/kaggle/input/<ds>`.
**3.12, not 3.11**: wheels built for 3.11 carry `cp311` ABI tags, `pip --no-index`
then finds no candidate, and the run dies at minute one. (`cp310-abi3` is *not* a
wrong tag — abi3 is the stable ABI and is forward-compatible.)
Three Kaggle Datasets — **wheels**, **code** (this repo, versioned per push with the git
SHA in the message so an LB entry traces to a commit), **weights**; the notebook is ~30
lines of glue, all logic lives in the repo. 20 GB output storage fills faster than
expected.

**Cross-platform rules** (each of these has caused real bugs):

- **Never kill by pattern on this box — `pgrep -f`/`pkill -f` match your own shell.**
  This has now bitten three times: `pkill -f 'method modelA_corrected'` killed its own
  shell (exit 144), an `until ! pgrep -f 'seq 1 6'` loop matched its own parent and
  spun for four days, and `for pid in $(pgrep -f watch_training.py); do kill $pid` took
  the shell out again mid-session while two multi-day training runs were live. The
  pattern appears in the command line of the very process doing the matching. **Kill by
  explicit PID**, and when *listing*, bracket the first character so the pattern cannot
  match itself: `ps -eo pid,cmd | awk '/[w]atch_train/'`. It is luck that the training
  survived the third one. **And bracketing the matcher is not sufficient** — the fourth
  occurrence bracketed it correctly and still killed the shell, because the *same
  command* also contained `git add scripts/watch_training.py`, i.e. the unbracketed
  literal, elsewhere on the line. The only reliable rule: **capture the PID when you
  launch, or stop the job through the harness (`TaskStop`), and never search for it.**
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
- **BUT THE PUBLIC LEADERBOARD IS ONLY 29 % OF THE TEST SET** (verbatim on the
  leaderboard page; Kaggle's own config carries `leaderboardPercentage: 29`). The final
  standing uses the other **71 %**, which nobody has ever seen. Everything this file
  calls "the leaderboard" is a 29 % sample, and `usesSynchronousReruns: true` means both
  scores are computed at rerun with only the public part revealed. Two consequences that
  should govern the endgame: a change worth less than the public LB's own resolution is
  not evidence of anything, and **our held-out CV is a better predictor of the private
  71 % than the public LB is** — because the public LB is what 862 teams have been
  fitting to. Kaggle also states plainly that "the final standings may be different".

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
| `src/volt_biohub/postprocess.py` | 620 | relink (now bypassed), gap closing, safe divisions, filter, smoothing | **the division term lives here** |
| `src/volt_biohub/edge_head.py` | 217 | our port of the transformer edge linker | **superseded** — loses 0.04 against the copy below |
| `src/volt_biohub/reference_predict.py` | 676 | the upstream inference path verbatim, tracksdata removed | **this is the detector+linker now** |
| `src/volt_biohub/ilp_link.py` | 480 | ILP over the resolved graph; `prune_dp` solves it exactly by tree DP | ours; keep |
| `src/volt_biohub/scoring.py` | 165 | video-level detect→link→score, shared by train and CLI | ours; keep |
| `src/volt_biohub/models/temporal_unet.py` | 141 | `TemporalUNet3D` | main detector |
| `src/volt_biohub/models/simple_node_transformer.py` | 210 | pairwise edge scorer | late-stage option |
| `kaggle/notebook.py` | 80 | the kernel: bootstrap + one `submit.run` call | glue only |
| `kaggle/kaggle_env.py` | 121 | Kaggle mount / test-dir / weights / GPU discovery | ours; keep |
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
- ~~**`predict()` cannot enumerate a directory.**~~ **Fixed 2026-09-08** —
  `--test-dir` enumerates `*.zarr`, works with no `.geff` present, and the splits path
  is unchanged. But read `predict_unet_transformer.py`'s docstring before assuming it is
  the submission path: **that script cannot be imported on Kaggle** (module-scope
  tracksdata reaches it via `io`/`metrics`). `kaggle/submit.py` is the submission entry
  point.
- Isotropic ↔ original-anisotropic round-trips are the highest-risk code here: wrong by
  4× in z, no exception raised, score near zero. **Two real bugs of exactly this shape
  were found in `io.py` on 2026-09-08 and fixed** (see Status), which is what an audit
  told to *assume a site is broken and prove it* turns up. There is now a round-trip test
  — `tests/test_coordinate_roundtrip.py` — and it is the reference for which inverse
  pairs with which forward transform. Anything new crossing that boundary goes in there.
  The trap to know: the `1,4,4` path is a **strided read**, so its exact inverse is
  `coord * downsample`; only an *interpolating* resample needs scale arithmetic.

Hygiene, each capable of wasting a day:

- **`validate_submission()` exists but nothing in the repo's own save paths calls it.**
  `src/volt_biohub/submission.py` is written and tested; the only callers are
  `kaggle/submission.py` and `kaggle/check_offline.py`. `scripts/geffs_to_csv.py` does
  **not** call it, so a local `geffs_to_csv` run still produces an unvalidated CSV. One
  line each at the two save sites closes this.
- **`predict(debug_video=…)` keeps the `.zarr` suffix** (`debug_video.name`), so a
  `--debug-video` run writes `x.zarr.geff` and puts `x.zarr` in the `dataset` column —
  a failure mode `validate_submission` explicitly tests for. Pre-existing.
- **`pos_feats` is dead code.** The window dataset still computes
  `extract_pos_features` per window, stores it on `WindowData` and the collate stacks it,
  but nothing reads it any more. Delete it rather than leaving it to imply the linker
  uses positional features it does not.
- **Nothing feeds an *empty* geff to `geffs_to_csv` in the suite** — the exact line that
  crashed the graceful-degradation path. Verified manually; a revert would leave the
  suite green and reintroduce a failure that only appears at the end of a 12 h run.

- **Three disagreeing defaults per knob.** `det_loss_weight`: `train_epoch` signature
  0.1, `train()` signature 1e1, CLI 1e0 with help text claiming 1e1. Same shape for `lr`
  (1e-3 vs 1e-4) and `num_workers` (4 vs 8 vs help saying 4). Always pass explicitly.
- ~~**The augmentation RNG ignored `--seed`.**~~ **Fixed 2026-09-16.**
  `FrameWindowDataset.__getitem__` built `np.random.default_rng()` fresh per window —
  unseeded OS entropy — because `worker_init_fn` seeds the legacy `np.random` global,
  which `default_rng` does not consult. So `--seed` never reached brightness or flips,
  and two runs with identical configs diverged: **E6 and E7, same recipe and same
  `--seed 0`, scored 0.6802 at epoch 12 and 0.6700 at epoch 10.** Any ablation on a
  delta smaller than that gap was measuring the augmentation draw. Now one generator
  per worker, seeded from `(seed, worker_id)` and advanced across items — per-worker
  rather than per-item deliberately, since `(seed, idx)` would hand each window the
  same flip and brightness every epoch and throw away most of the augmentation's
  value. **E6 and E7 themselves are not reproducible**; treat their absolute numbers as
  one draw each.
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
- ~~`open_dataset(device="cuda")` is the **default** → crashes CPU-only.~~ **Fixed
  2026-09-08** — resolves cuda-if-available. It had gone from latent to actively failing
  the suite under `CUDA_VISIBLE_DEVICES=`. Its eager `da.from_zarr(...).compute()` loads a whole video, which at ~839 MB
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

**2026-09-08 — THE BIGGEST SCORE GAIN IN THE PROJECT, and it needs no new training:
switch detector on predicted cell density.** DoG and the U-Net tie on aggregate but
disagree per sample, and the disagreement is **predictable**. Over all 199 samples,
scored per sample at the tuned config and aggregated by `w_i`:

| strategy | weighted score |
|---|---|
| classical DoG alone | 0.6753 |
| U-Net alone | 0.6789 |
| **switch at 150 cells/frame** | **0.7366** |
| per-sample oracle | 0.7433 (the ceiling) |

**+0.058 over the better single method, capturing 91 % of the oracle headroom.** The
mechanism is exactly the fold split: `spearman(cells_per_frame, unet − dog) = +0.700` —
**the U-Net wins on the crowded samples, DoG on the sparse ones**, and the U-Net wins
107/199 overall.

**It survives a predicted density, which is what makes it shippable.** `T_true` does not
exist at test time, but `check_t_true.py` measures the p90-quantile predictor at R² 0.72
/ 25 % median relative error. Simulating exactly that noise (lognormal, 400 draws):

| threshold | mean | p5 | worst of 400 |
|---|---|---|---|
| 125 | 0.7343 | 0.7308 | 0.7271 |
| **150** | **0.7350** | 0.7321 | **0.7297** |
| 175 | 0.7334 | 0.7301 | 0.7276 |

**The worst single draw (0.7297) still beats the best single method (0.6789) by +0.051**,
and the response is flat from 125–175 so the threshold is not delicate. Cost: run both
detectors — DoG is ~5 s/video on CPU, so it is nearly free next to the U-Net's 1.86
s/frame.

*Caveat on the aggregation:* these weighted figures pool all 199 samples by `w_i`
directly (0.6753 / 0.6789), where the per-fold table below weights fold aggregates
(0.6801 / 0.6790). Both are legitimate; the pooled one is the more direct computation and
is what the switch numbers are built on. Do not mix them.

**2026-09-08 — THE HEADLINE, and it took two corrections to get here.** Four numbers,
all on **full** LOEO folds at the **identical** tuned config (gate 7 µm, separation
4 µm, division gate 6 µm, refined peaks, budget 1.0× `T_true`):

| method | fold 0 `6bba` (128) | fold 1 `44b6` (71) | weighted | **spread** |
|---|---|---|---|---|
| classical DoG | **0.7107** | 0.5117 | **0.6801** | 0.199 |
| U-Net (models A / D) | 0.6794 | **0.6768** | 0.6790 | **0.003** |

**They tie on weighted CV — 0.6801 vs 0.6790 — but the profiles are opposites, and the
spread is what matters.** The hidden test set is a *third, entirely unseen embryo*, so
the gap between our two LOEO folds is the only proxy we have for cross-embryo transfer.
DoG is excellent on `6bba` and collapses on `44b6`; the U-Net scores the same on both to
within **0.003**. A method that swings 0.199 between the two embryos we can see could
land anywhere on a third. **Prefer the U-Net on generalisation grounds despite the tied
average**, and note it is also uniformly the better *detector*: node recall 0.889/0.892
against DoG's 0.833/0.679, and it hits the node budget almost exactly (ratio −0.0003 and
−0.0031 against DoG's −0.060 and −0.270).

**Correction 1 — tuning beat every model change, and it invalidated the earlier
comparison.** Re-measuring the classical floor at the tuned settings moved it from
0.6424 → **0.7107** on fold 0 and 0.4030 → **0.5117** on fold 1: **+0.068 and +0.109
from four scalars, no training.** So 2026-09-07's claim that "the learned detector
overtakes the classical floor at 0.6925" was an artefact of comparing a tuned U-Net
against a baseline frozen at the old defaults. **General lesson: every model comparison
run before the knobs were tuned was measuring the knobs.** Retune the baseline whenever
you retune a model, or the comparison is worthless.

**Correction 2 — the knobs interact three ways, so sweeping one at a time misleads.**
The division second pass was recorded on 2026-09-07 as a flat net loss of ~0.025. That
was true only behind `sep 5.86 / gate 10`. At `sep 4.0 / gate 7.0` it is net **positive**
(0.7007 with it, 0.6997 without, divJ 0.0086). Separation also has a sharp optimum that
is *quantised*: `_peak_window` takes `ceil(sep/1.625)`, so 5.0 and 5.86 both give a
7-voxel window and score identically — there are ~5 distinct settings, not a continuum.
Sweep the distinct windows, not round numbers.

| sep | gate | div | score |
|---|---|---|---|
| 4.0 | **7.0** | **6.0** | **0.7007** |
| 4.0 | 8.0 | 6.0 | 0.6994 |
| 4.0 | 7.0 | off | 0.6997 |
| 5.86 | 8.0 | off | 0.6925 |
| 5.86 | 8.0 | 10.0 | 0.6672 |

**The budget optimum is purely the node penalty, and that is an opportunity.** Edge
Jaccard rises monotonically with budget — 0.6690 → 0.6914 → 0.6984 → **0.7008** at 1.5×
— while the *score* peaks at 1.0×, because the ratio hits +0.4346 and the multiplier
0.957. The detector genuinely wants to detect more; only the penalty stops it. Since
`T_true` is predictable from stored quantiles at Spearman 0.856, **that 0.7008 is
reachable with a better per-sample budget** and is the cheapest remaining upside.

**The loss/metric anti-correlation is structural, not a fold artefact.** Model D (train
`6bba` → test `44b6`) reproduces it on the other embryo: detection loss fell 2.4×
(0.0842 → 0.0351) while CV drifted 0.6518 → 0.6254, best at **epoch 2**. Model A on the
other fold fell 4.4× with best at **epoch 0**. Two embryos, two folds, same shape. This
remains the most important open problem, and `--save-every` plus the new
`epoch_metrics.csv` now make the divergence point measurable — that experiment has not
been run yet.

**Node recall is a bad proxy target — three independent knobs prove it.** Tightening
separation 5.86 → 3.0 µm raises node recall 0.860 → 0.893 and *drops* the score to
0.6478. The U-Net has higher recall than DoG and (before tuning) a lower score. Loosening
the gate raises exposure without raising recall at all. In every case the mechanism is
metric fact 3: node recall gates how much of your linking gets graded.

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

**2026-09-08 — four blockers found and closed, one of them existential.** Suite
**171 → 280 passing** on CPU, ruff 83 → 63. All four were surfaced by adversarial review
of the implementing agents, and each was independently re-verified rather than taken on
the implementer's word.

1. **The Kaggle notebook could never have run.** `tracksdata` — which is *not on PyPI*,
   in an internet-disabled competition — was imported at **module scope** in four places
   on the inference path: `linking.py:30`, `scoring.py:20`,
   `predict_unet_transformer.py:26`, `geffs_to_csv.py:7`. Now tracksdata-free, and
   verified the only way that counts: a clean venv with tracksdata **physically absent**
   (12 wheels from `kaggle/requirements-offline.txt`) ran the whole path on the 4 real
   test samples to a validated `submission.csv`. The DoG fallback runs with **torch not
   installed at all**. `kaggle/check_offline.py` is the gate; note it only blocks *named*
   modules inside the dev venv, so it proves less than it looks — a clean venv is the
   real test.
2. **Two 4× coordinate bugs in `io.py`**, the class this file calls the highest risk in
   the project, previously with no test at all. `open_dataset` ignored `downsample` when
   `load_image=True` (same call reported shape `(5,64,64,64)` vs `(5,64,256,256)`), and
   `rescale_graph_to_original` hardcoded `min(scale)` so it could not invert the resample
   that produces our own isotropic grid. Both fixed; `tests/test_coordinate_roundtrip.py`
   now pins which inverse pairs with which forward transform.
3. **`predict --test-dir` exists**, so inference on the real test set is no longer
   blocked. Two defects on that path also fixed: `skip_failures=True` wrote a zero-node
   geff that then *crashed the CSV writer* (a graceful-degradation path that was a
   guaranteed total failure), and stale-output deletion ran **before** the model loaded,
   so a typo in `--weights` destroyed the previous predictions and produced nothing.
4. **`fetch_data.extract()` returned success after a batch threw**, so a partial
   extraction of 24,886 files looked complete. Now verifies against the filesystem and
   raises `IncompleteTransfer`.

**The flaky test was a CPython bug, and my diagnosis of it was wrong.**
`test_zip64_archive_reports_real_sizes` fails because `zipfile._extract_member` does
`if not exists(upperdirs): makedirs(upperdirs)` with **no `exist_ok`**, and `os.makedirs`
re-raises `FileExistsError` for its *leaf* — so two extract workers sharing a parent
directory race and the loser aborts its whole batch. It is `dest` itself that collides
(`sample_submission.csv`'s parent *is* `dest`), which is why only the one test extracting
into a not-yet-existing dir ever flaked. **Measured 11.3 % (34/300) in a fresh process**,
so my "it needs the rest of the suite in the process first" theory was chance — p(0 in
12) ≈ 0.25 at that rate. I over-read a null result; don't repeat that. Fixed by creating
each parent with `exist_ok=True`: 0/600 after. The real download was never exposed, by
luck of layout — **0 of 24,886 parent directories hold more than one member.**

**Kaggle runtime is tighter than the 12 h cap suggests.** Measured, not assumed: the
U-Net detector costs **1.86 s/frame pinned to 4 threads** (Kaggle-like), which is
**10.3 h for a train-sized 199-video test set** — inside the cap with almost no margin.
0.74 s/frame with all 256 cores here, so this box tells you nothing about it. **A GPU
notebook is required**, and the DoG fallback exists precisely so a weights or torch
failure degrades to something submittable.

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
`dataset_splits.json` now exists beside the data, but from a throwaway script). The
Kaggle team exists — see Gate 0.

**2026-09-16 — THE LOSS/METRIC ANTI-CORRELATION IS DEAD. It was the selection metric
all along.** This file has called it "the single most important open problem" since
2026-09-07 and recorded it as reproducing across three folds. Scored the way it should
always have been — our own 24-epoch full-pass model (`E6`), candidate epochs scored on
the **full 71-sample fold** through the **submission path** with the threshold detector:

| checkpoint | score | node recall |
|---|---|---|
| epoch 0 — **what `best.pth` selected** | 0.6679 | 0.8648 |
| epoch 12 | 0.6802 | 0.9169 |
| **epoch 23** | **0.6852** | 0.9070 |

**More training monotonically improves the real score.** Every "best epoch is 0"
observation in this file came from `epoch_selection_score` on `--select-videos 8` (6 in
`E6`) — an effective n of about 2, where the run-to-run spread swamps the signal. The
selector was reliably picking our *worst* checkpoint of twelve. Combined with the
reference's own `history.csv` (det loss falls monotonically 0.0198 → 0.0025 while their
validation metric rises 0.900 → 0.978 over 400 epochs), the conclusion is settled:
**there is no anti-correlation. There was a broken ruler, twice — once in the gate
config, once in the sample size.**

Consequences:
- **Never select a checkpoint on the per-epoch subsample.** `--save-every` and a
  full-fold score afterwards is the only trustworthy selection. `best.pth` is
  actively misleading and should be treated as "the epoch the cheap print liked".
- Every detector conclusion drawn from "best epoch 0" is void, including the ordering
  hinge and ignore-radius arms and the `det_loss_weight` sweep.
- **Our own detector is close.** E6 at 24 full-pass epochs scores 0.6852 on fold 1
  against the public 400-epoch checkpoint's 0.7830 — a 0.098 gap for 16x less training,
  which is exactly the shape of their curve (0.939 by epoch 51, 0.967 by 151).

`E7_fold1_100ep` and `E7_alltrain_100ep` are now running, 100 full-pass epochs each,
`--save-every 5`. Split 4 was added as an all-train split (199 samples) matching the
reference recipe; its `test` list is a subset of train and exists only to feed the cheap
per-epoch print, which we do not select on.

**2026-09-21 (later) — THE CHECKPOINT THE WHOLE FIELD FORKS IS THE ARGMAX OF NOISE,
and averaging it with its own successor is free.** Read out of the pack's own
`history.csv`: over epochs 370–400 their `validation_score` oscillates between
**0.9696 and 0.9781 with no trend**, and that score is `validation_acc ×
validation_recall` with `validation_acc` pinned at 0.9999 — i.e. effectively
edge-matrix recall, the quantity this project measured as worthless for selection
(2026-09-16). So `edge_predictor_best.pth` at epoch 381 is the maximum of a noisy
sequence, not the best model. `checkpoint_last.pth` ships in the same directory at
epoch **400** (pack50's at **402**) and **all 136 tensors differ**; nobody had looked.

**CONFIRMED ON BOTH FULL FOLDS: weighted LOEO CV 0.9332 -> 0.9407, +0.0075, and every
term moves the same way on both embryos.**

| | fold 0 shipped | **fold 0 soup** | fold 1 shipped | **fold 1 soup** |
|---|---|---|---|---|
| score | 0.9361 | **0.9428** | 0.9168 | **0.9290** |
| adjJ | 0.9279 | **0.9333** | 0.9104 | **0.9207** |
| divJ | 0.0816 | **0.0950** | 0.0641 | **0.0833** |
| node recall | 0.9789 | **0.9826** | 0.9758 | **0.9830** |
| node ratio | -0.0391 | -0.0163 | -0.2247 | -0.1671 |

0.9333 is the highest adjusted edge Jaccard measured in this project. Three properties
make this the best change of the day rather than merely the biggest: it moves the edge
term **and** the division term **and** node recall together, which is a better detector
rather than a redistribution between terms; it is **detector-side**, so by the
contamination asymmetry already measured our CV *understates* it rather than inflating
it; and it costs **nothing** at inference and needs **no code change** — only a new
weights-dataset version, since the kernel resolves `volt-weights` unpinned.

**Souping and blending are complementary, not substitutes — measured.** Turning the
cross-run blend off *under* the soup gives 0.9392 against 0.9428 with it, so the blend
still earns **+0.0036** on top of souping (adjJ 0.9298 vs 0.9333, recall 0.9809 vs
0.9826). That is the decomposition it implies: **within-run averaging cancels
checkpoint-selection noise, across-run blending cancels initialisation and data-order
noise.** Different noise sources, so they add — and the secondary encode is therefore
earning its ~20 % of detector runtime rather than being dead weight. Keep both.

**The soup un-gates ONE of the two edge-side arms, and an earlier version of this
paragraph called it for both on the strength of the first arm to report — which was
wrong.** Everything on top of the soup at blend 0.15 (baseline 0.9428):

| arm | score | adjJ | divJ | vs old operating point |
|---|---|---|---|---|
| **secondary edge head w=0.5** | **0.9439** | **0.9350** | 0.0891 | +0.0011 against +0.0001 — **10×** |
| bidirectional fusion w=0.15 | 0.9433 | 0.9343 | 0.0898 | +0.0005 against +0.0008 — flat |

adjJ 0.9350 is the highest measured in this project. So the division headroom the soup
restores is spendable by the *model ensemble* and not by the *direction ensemble*, and
the honest reading is that the two arms were never one phenomenon: averaging two
independently trained heads adds information, while averaging a head against its own
transpose only adds an agreement test, and an agreement test has nothing to spend extra
headroom on.

The blend weight did **not** move under the new primary, which is worth noting because it
moved twice before when an upstream component changed: 0.00 → 0.9392, **0.15 → 0.9428**,
0.25 → 0.9401. A clean interior optimum in the same place.

**AND THE TWO ARMS ARE EXACTLY ADDITIVE, WITH AN INTERACTION ON THE DIVISION TERM I
WOULD NOT HAVE PREDICTED.** Both together, with bidir at its own measured optimum of
0.25 rather than the 0.15 tested first:

| fold 0 | score | adjJ | divJ | node recall |
|---|---|---|---|---|
| soup + blend 0.15 | 0.9428 | 0.9333 | 0.0950 | 0.9826 |
| + edge head 0.5 | 0.9439 | 0.9350 | 0.0891 | 0.9820 |
| + bidir 0.15 | 0.9433 | 0.9343 | 0.0898 | 0.9834 |
| **+ both, bidir 0.25** | **0.9454** | **0.9352** | **0.1025** | **0.9839** |

+0.0011 and +0.0015 sum to the measured +0.0026, so they cancel *different* noise rather
than the same noise twice.

**And the division story was a one-weight artefact — corrected.** Sweeping fusion on the
soup shows its effect on divJ is *weight-dependent*, not a fixed cost:

| bidir weight on the soup | score | adjJ | divJ |
|---|---|---|---|
| 0 | 0.9428 | 0.9333 | 0.0950 |
| 0.15 | 0.9433 | 0.9343 | 0.0898 |
| **0.25** | **0.9444** | 0.9344 | **0.1000** |

**Swept to both sides, and the optimum is a PLATEAU rather than a spike:**

| bidir weight on the soup | 0 | 0.15 | **0.25** | **0.35** | 0.50 |
|---|---|---|---|---|---|
| score | 0.9428 | 0.9433 | **0.9444** | **0.9444** | 0.9422 |
| divJ | 0.0950 | 0.0898 | 0.1000 | 0.1008 | 0.0947 |

At 0.15 it sells divisions; from 0.25 to 0.35 it **buys** them and is worth +0.0016
alone; by 0.50 it is below w=0. So the earlier claim that "both edge-side arms buy the
edge term by selling divisions, a roughly fixed fraction of forks suppressed" was drawn
from a single weight and is wrong. The mechanism that fits all of it: a harmonic mean
weighted further toward the reverse pass is a **stronger agreement test**, and a dividing
parent's two daughters each *independently* agree with it — each daughter's own best
parent is the divider — so a stronger test preserves forks while killing spurious single
links. A weak test only sharpens the dominant assignment, and that is what suppresses a
second child; too strong a test starts rejecting real links on both sides.

The flatness matters methodologically: a two-point plateau with symmetric falloff is
what a real optimum looks like, where fold-0 noise would give a jagged profile. 0.25 is
kept because it is the value confirmed in combination on **both** folds.

Note also that bidir's own optimum on fold 0 is **0.25** (0.9375) rather than 0.15
(0.9369) at the shipped operating point — the first sweep bracketed it too narrowly and
the 0.40 arm was killed early for cards, so the upper side is still unmeasured.

**CONFIRMED ON BOTH FOLDS, with the same delta on each.** Fold 1 gives 0.9316 against
the soup's 0.9290 — **+0.0026**, identical to fold 0's +0.0026, which is what a real
effect looks like rather than a fold artefact:

| | fold 0 | fold 1 | **weighted** |
|---|---|---|---|
| soup + blend 0.15 (submitted as kernel v14) | 0.9428 | 0.9290 | 0.9407 |
| **+ edge head 0.5 + bidir 0.25** | **0.9454** | **0.9316** | **0.9433** |

**+0.0101 over this morning's 0.9332.** One caveat to state rather than hide: the
division term moves in *opposite* directions on the two folds (+0.0075 on fold 0,
−0.0064 on fold 1) while the net is the same. Fold 1 holds only 26 divisions against
fold 0's ~125, so fold 0 is the division measurement to trust and fold 1's divJ is
close to noise at that n. Individually every non-soup increment is still ≤ +0.0015, an order of magnitude
below the reference's published deltas for the same components — their deltas repair a
defect we do not have — but in combination they are worth having.

Proxy set (the 4 movies the public notebooks validate on), everything else matched,
same card:

| primary | score | adjJ | divJ | node recall | ratio |
|---|---|---|---|---|---|
| epoch 381 — `best`, what the field ships | 0.9457 | 0.9314 | 0.1429 | 0.9728 | −0.045 |
| epoch 400 — `checkpoint_last` | 0.9409 | 0.9298 | 0.1111 | 0.9845 | +0.065 |
| **soup(381, 400)** | **0.9759** | **0.9359** | 0.4000 | 0.9832 | +0.048 |

**The control is the point: epoch 400 alone is WORSE than 381, so the gain is
averaging cancelling selection noise, not a later checkpoint being better.** Read the
magnitude honestly, though — most of that 0.030 is the division term on ~4 GT
divisions, which is near-binary at this n. The defensible claim is **adjJ +0.0045 with
node recall +0.010 at zero inference cost**, and a full fold is what settles it. A soup
is strictly cheaper than an activation ensemble: one model, same size, no extra forward
pass — which matters at 5.7 h of a 12 h cap.

`scripts/soup_checkpoints.py` is the provenance, verified to reproduce the measured
soup bit-for-bit. Two traps in it: `checkpoint_last.pth` is a 25 MB *training*
checkpoint with the weights under `model_state_dict`, and `num_batches_tracked` is an
int64 counter that differs by **8.9e4** between members — large enough to look alarming
in a naive per-tensor diff, and meaningless to average, so it is copied. **Souping
across packs is invalid**: `pubfull` is `unet_transformer_alltrain_seed314159_v1` and
`pack50` is `unet_transformer_5090_50ep_v1` with no seed recorded, so different inits,
different basins under permutation symmetry, and averaging them would destroy both.
They are combined at the activation level instead — that is what the detector blend is.

**Contamination caveat, stated plainly:** both members trained on all 199 movies, so
*no* held-out set exists for them and a soup's CV gain cannot be separated from sharper
memorisation. Only the leaderboard can arbitrate this one — **but the calibration now
favours it.** See the two blend submissions below: a *detector-side* change transferred
at 80–130 % where the edge-head-side reference-path change transferred at 35 %, which is
the asymmetry this file predicts. At that rate the soup's +0.0075 lands near 0.932–0.933.

**2026-09-22 — THE SUBMIT 403 WAS THE WRONG COMPETITION SLUG. My diagnosis was wrong,
it cost ~10 hours and 120 retries, and the evidence was in our own files the whole
time.** `THE COMPETITION SLUG IS `biohub-cell-tracking-during-development`.` The other
one — `biohub-cell-tracking-competition`, which every submit command in this session
used — is not this competition. Submitting to a competition you are not entered in
returns **403 Forbidden**, which is exactly correct behaviour.

`kaggle competitions list -s biohub` returns exactly **one** competition,
`biohub-cell-tracking-during-development`, with `userHasEntered=True` and
`userRank=1643 of 3802`. **Rules were accepted the entire time and no human action was
ever required.** With the right slug the submission went through immediately.

**What I did wrong, because the shape repeats.** I eliminated three hypotheses about
*why* the request was refused — concurrent-rerun cap, daily quota, token/auth — and
concluded by elimination that the rules must have lapsed. I never questioned *what the
request was addressed to*. Two pieces of evidence were already in front of me:

- `competitions leaderboard <wrong-slug>` returns **404 Not Found**, not 403. A missing
  resource, not a permissions failure. I read past the difference.
- `kaggle/kernel-metadata.json` has carried
  `"competition_sources": ["biohub-cell-tracking-during-development"]` since the first
  kernel, and I *read that file in the same session* to check how it pins datasets —
  without comparing its slug to the one I was passing to `submit`.

**The transferable rule: when every hypothesis about a failure's mechanism has been
eliminated, the remaining possibility is usually not the exotic one still standing — it
is that a premise of the question is wrong. Check the identifier before the mechanism.**
This is the third instance in this file of the same failure mode (the six "ILP" arms
that ran with no edge head, the waiter that read the wrong row below, and this).

**One operational fact from the episode that does hold:** a 403 creates no submission,
so none of those 120 attempts consumed a daily slot. And the note about unpinned dataset
versions stands on its own reasoning: a pushed-but-unsubmitted kernel version is not a
frozen experiment, because its graded rerun resolves `volt-wheels`/`volt-code`/
`volt-weights` at submission time and `kernel-metadata.json` lists all three unpinned.
Pin them if two configurations must be in flight at once.

**2026-09-21 (later), CORRECTED 2026-09-22 — LB 0.920 → 0.925, AND OUR CV RANKED THE
TWO BLEND WEIGHTS CORRECTLY. The first version of this entry reported both as 0.925 and
that was a bug in my own waiter, not a fact about the board.** Read from
`competitions submissions` by **ref**:

| change | CV | LB | CV Δ | LB Δ |
|---|---|---|---|---|
| reference path + ILP (`56426110`, v11) | 0.9270 | 0.920 | — | — |
| blend 0.30 (`56432361`, v12) | 0.9308 | **0.918** | +0.0038 | **−0.002** |
| blend 0.15 (`56438177`, v13) | **0.9332** | **0.925** | +0.0062 | **+0.005** |

**How the error happened, because it is the same failure as the slug above.** Two
submission waiters both used `awk 'NR==3'` — the *first data row*, i.e. the **newest**
submission — so each reported whichever submission was most recent rather than its own.
Both printed 0.925. Match by ref, never by row position. That is now three instances in
this file of taking a value without verifying the identifier it belongs to.

**What the corrected numbers actually say, and it is better news than the wrong ones.**
Our CV **ranked the two blend weights correctly**: it preferred 0.15 over 0.30
(0.9332 > 0.9308) and the board agrees decisively (0.925 > 0.918, a 0.007 gap far above
the board's 0.001 display resolution). Blend 0.30 is in fact *worse than no blend at all*
on the board (0.918 against 0.920) while 0.15 is better — consistent with CV's interior
optimum at 0.15, and a genuine confirmation that the harness can rank.

**RETRACTED: "sub-0.005 CV arms are invisible on the board."** That claim rested
entirely on two configurations 0.0024 apart appearing to score identically, which they
did not. The board resolved a 0.0024 CV difference into a 0.007 LB difference. Do not
use display granularity as a reason to withhold a small arm; the earlier and independent
argument — that *deltas* compress on transfer — still stands on its own evidence and is
the one to reason from.

**The transfer rates, recomputed.** blend 0.15 over the reference path: +0.0062 CV →
+0.005 LB, about **81 %**. The reference path itself over the old stack: +0.0346 CV →
+0.012 LB, about **35 %**. The detector-side/edge-head-side asymmetry this file predicted
is still the right shape — the detector-side change transfers far better than the
edge-head-side one — but it is now **one clean point per side rather than two**, so treat
80 % as a single measurement and not a calibrated rate.

**2026-09-21 (later) — THE DIVISION FNs DECOMPOSED, AND IT REVERSES THIS FILE'S OWN
2026-09-18 VERDICT.** That verdict — "the remaining division loss needs a better ranker
or a new feature, not a different threshold" — came from sweeps that *tightened* the
structural gates, on the unstated assumption that the admissible candidate pool was
already exhausted. `scripts/division_audit.py` measures it. On the 21 fold-1 samples
that contain a GT division (26 divisions, TP 5 / FP 19 / FN 21, divJ 0.1111):

| of the 21 FN divisions | |
|---|---|
| a node was never detected | **6** — detection-limited |
| all three nodes detected | **15** — the cells are there, no fork was emitted |
| …of those, divider out-degree 1 | 14 — i.e. inside `add_safe_divisions`' pool |
| …second daughter a genuine **orphan** | **11 — fully admissible, a gate rejected it** |
| …second daughter parented by a foreign node | 4 — the orphan rule; no distance gate reaches these |

So the largest bucket is a structural gate throwing away candidates it was handed.

**AND THEN LOOSENING THAT GATE WAS MEASURED AND LOST, MONOTONICALLY — my prediction
from this decomposition was wrong.** Full fold 0, `--div-diverge-um` against the 2.25
default:

| forward divergence | score | adjJ | divJ |
|---|---|---|---|
| **2.25 µm (shipped)** | **0.9361** | 0.9279 | **0.0816** |
| 1.0 µm | 0.9341 | 0.9272 | 0.0691 |
| −2.0 µm | 0.9320 | 0.9268 | 0.0521 |

**divJ *falls* as the gate opens**, which is the mirror image of 2026-09-18's finding
that tightening it also falls. The gate is at a genuine interior optimum in both
directions, so **2026-09-18's verdict — the division loss needs a better ranker, not a
different threshold — was right, and the reversal above was wrong.**

The two findings are compatible in a way I should have seen immediately: the forks are
**admissible but out-ranked**. The caps are hard, so more candidates means the ranking
decides, and the ranking is `d_par + 0.15·d_sis` over exactly the geometric features
2026-09-12 measured as non-separating at the quartiles. A flooded cap fills with false
forks that outrank the true ones, which is precisely the monotone fall.

**What the sweep could not test, and is now the prime suspect.** `--div-diverge-um`
relaxes only the *distance* clause `dist(g1,g2) − d_sis < diverge_um`. Beside it sits
`len(g1) != 1 or len(g2) != 1` — **both daughters must have exactly one child each** —
which is hard-coded, demands two further frames of correct linking on both branches, and
no CLI knob reaches it. Extending the audit says the sister is at **orphan rank 0 in 6 of
6 cases where rank is computable**, i.e. she *is* the candidate the mutual-nearest rule
picks, so a gate really does reject her; and since the distance clause is now measured
as already optimal, the structural clause is what is left. That is a one-line change
worth one arm, and it is the only division lever this project has not measured.

**MORE SOUP MEMBERS TRADE THE SAME WAY, AND THE EDGE PAIR REVERSES THE TRADE — the one
prediction today that held.** A four-member soup (pubfull 381 and 400 plus two epochs
continuing that run under their own recipe) scores 0.9440 alone, +0.0012 over the
two-member soup, and it gets there by the documented TTA mechanism: adjJ 0.9333 → 0.9359
(+0.0026, highest measured) against divJ 0.0950 → **0.0805**. Averaging smooths the
detection map, and sister separation one frame after a division has p5 of 5.41 µm on a
1.625 µm grid, so daughters merge under the average. **So member count has an interior
optimum rather than being monotone.**

Pairing it with the edge-side arms, whose effect on divisions runs the other way,
recovers it:

| fold 0 | score | adjJ | divJ |
|---|---|---|---|
| soup2 + blend 0.15 (kernel v14) | 0.9428 | 0.9333 | 0.0950 |
| soup4 alone | 0.9440 | 0.9359 | 0.0805 |
| soup2 + edge head 0.5 + bidir 0.25 | 0.9454 | 0.9352 | 0.1025 |
| **soup4 + edge head 0.5 + bidir 0.25** | **0.9463** | **0.9361** | 0.1020 |

**soup4 IS confirmed on fold 1, and it gains far more there: 0.9367 against soup2+pair's
0.9316, i.e. +0.0051 where fold 0 gave +0.0009**, with divJ 0.1061 against 0.0769. So the
caveat recorded here an hour earlier — that soup4's increment was fold-0-only and inside
noise — is resolved in its favour. Best measured configuration, both folds:

| | fold 0 | fold 1 | **weighted** |
|---|---|---|---|
| start of 2026-09-21 | 0.9361 | 0.9168 | 0.9332 |
| soup2 + blend 0.15 (kernel v14, pending) | 0.9428 | 0.9290 | 0.9407 |
| soup2 + edge head 0.5 + bidir 0.25 | 0.9454 | 0.9316 | 0.9433 |
| **soup4 + edge head 0.5 + bidir 0.25** | **0.9463** | **0.9367** | **0.9449** |

**+0.0117 over the day's start**, and +0.0042 over what is queued. Expected on the board,
splitting by the measured transfer rates: soup4-over-soup2 is detector-side (+0.0016 at
~100 %) and the edge pair is edge-head-side (+0.0026 at ~35 % → +0.0009), so ~+0.0025
over whatever v14 scores.

**THE MEMBER-COUNT CURVE, measured at three points, and it flattens at 4.** All with the
same edge pair on fold 0:

| soup members | score | adjJ | divJ |
|---|---|---|---|
| 2 | 0.9454 | 0.9352 | 0.1025 |
| **4** | **0.9463** | 0.9361 | 0.1020 |
| 8 | 0.9466 | **0.9370** | 0.0956 |

adjJ rises **monotonically** with members while divJ falls, and the two cancel: +0.0009
from 2→4, then **+0.0003** from 4→8, which is inside the noise floor. So the smoothing
mechanism is confirmed across three points rather than inferred from two, and **4 members
is the operating point** — indistinguishable from 8 on score, fewer artifacts to manage,
and the only count confirmed on both folds. Do not chase more SWA epochs for this; the
curve is flat and the division term is what pays for the edge term.

**A TRAP THAT NEARLY COST THE ATTRIBUTION, and it constrains the order of operations.**
A code competition's graded rerun executes **at submission time against the CURRENT
dataset versions**, and `kernel-metadata.json` lists `volt-wheels`/`volt-code`/
`volt-weights` **unpinned**. So uploading soup4 — or pushing volt-code — before v14 is
submitted would silently change what v14's rerun runs, and v14 exists precisely to
measure the soup against byte-identical code. **Nothing may be uploaded to any of the
three datasets until v14 is submitted and scored.** More generally: a pushed-but-
unsubmitted kernel version is not a frozen experiment, because its inputs are still
live. Pin the dataset versions in `kernel-metadata.json` if two configurations ever need
to be in flight at once.

**MEASURED, AND IT IS A NO-OP: 0.9428 against the soup's 0.9428**, adjJ identical at
0.9333, divJ 0.0950 → 0.0956, node recall and ratio unchanged. So the grandchild
requirement was not the blocker either, and my second division prediction is falsified
as cleanly as the first.

**That completes a diagnosis by elimination. Every structural gate in
`add_safe_divisions` has now been individually tested and none of them binds:**

| gate | how tested | verdict |
|---|---|---|
| divergence *distance* | swept −2.0 / 0.0 / 1.0 vs 2.25 | 2.25 optimal; looser loses monotonically |
| grandchild requirement | `--div-no-grandchildren` | **no-op** |
| symmetry τ | 0.6 → 0.75 (2026-09-18) | rejected **zero** candidates |
| parent / sister distance | widened to 11.0 / 18.0 (2026-09-18) | loses |

~~**What is left is the caps picking by a ranking that cannot separate.**~~
~~**So the division term is stuck at ~0.095 without a NEW FEATURE, and this is now a
closed line rather than an open lever.**~~ **RETRACTED 2026-09-22 — the cap does not
bind, the ranking is not the constraint, and the reasoning that closed this line was
wrong.** It rested on two measurements that cannot both be true, and I never noticed the
contradiction: loosening the divergence gate lost *monotonically* (divJ 0.0816 → 0.0521),
which requires false forks out-ranking true ones under a **binding** cap, while scaling
both caps by 1.25× measured **neutral** (divJ 0.0822 → 0.0800), which says the cap
barely binds at all. Nobody had looked at the pool the ranking ranks.

**`VOLT_DIV_DUMP` + `scripts/division_candidate_audit.py` now measure it directly.** One
row per admitted candidate — every fork clearing all four structural gates — labelled
against GT (parent within 7 µm at ±1 frame *and* both daughters matching in either
pairing). On the 6 division-richest fold-0 samples at the best configuration:

| | |
|---|---|
| frames holding candidates | 164 |
| frames where `n_props > frame_cap` | **21 (12.8 %)** |
| median `n_props` / `frame_cap` | **1.0 / 2.0** |
| candidates accepted | **197 of 237 (83.1 %)** |
| **true forks accepted** | **6 of 6, every one at rank 0** |
| true forks lost to the frame cap | **0** |
| true forks lost to the global cap | **0** |

**When a true fork is in the pool it ranks FIRST and is accepted.** The shipped ordering
scores AUC **0.531** over the admitted pool — no discriminative power — which is not a
problem to fix but a knob with nothing to do. The pool simply does not *contain* the true
forks, so "admissible but out-ranked" was the wrong diagnosis of the right observation.

**And that resolves the contradiction rather than merely replacing it.** With a
non-binding cap and 83 % acceptance, loosening a gate adds candidates that are then
*accepted* with no ranking filter in front of them — so the divergence sweep's monotone
fall is exactly what a non-binding cap predicts, and it never was evidence about the
ranking.

**What the gate sweeps could not reach is a SELECTION RULE, not a threshold.**
`tree.query(pos[ch])` takes the **single** nearest orphan to the existing child. A
spurious detection nearer than the true sister therefore makes her unproposable by any
gate at any width — which is why every gate came back empty *and* why the earlier FN
audit found 11 structurally admissible FNs that were nevertheless never admitted. Those
two findings were never in conflict; both describe a pool that is missing its candidates.

**The metric says over-proposing here is nearly free**, which is what makes this worth a
knob: only **40 of 4,053** emitted forks are charged as FP at all, because forks outside
annotated regions are ignored. So the division term is *recall*-limited, a losing
candidate usually costs nothing and a winning one is a full TP.

`--div-n-cand` implements it as a **fallback**, deliberately not a multi-propose: the
first of the k nearest orphans to clear every gate wins and the rest are dropped, so the
pool size and the out-degree are unchanged and no node can reach out-degree 3 (which the
scorer truncates silently, by edge id). The mechanism has a test: a duplicate detection
of the existing child sits nearest and has no child of its own, so it fails the
divergence clause and the true sister behind it gets her turn — `n_cand=1` finds no fork
in that fixture and `n_cand=2` finds the right one. **MEASURED AND NEGATIVE, exactly as the rejection log said it would be: fold 0 0.9459
against 0.9463, with divJ 0.1020 → 0.0996 and node recall and node ratio BIT-IDENTICAL
(0.9830, −0.0350).** Those two unchanged columns are the mechanism check — a fork adds an
edge and never a node, so identical recall confirms the arm changed only forks — and the
division term falling confirms which forks: since every true daughter that clears the
gates is already the nearest orphan, the 2nd and 3rd can contribute nothing *but* false
ones.

This is the first division prediction of mine in this project that held, and the reason
is worth keeping: it came from a **measurement of the rejection log** rather than from a
mechanism story. The five that failed were all mechanism stories. The knob stays at its
default of 1, which is an exact no-op, and the fixture test keeps documenting a real
behaviour that simply does not occur at this operating point.

**AND THE HEADROOM IS LARGE: 99 of 124 GT divisions never reach the pool at all.**
Confirmed on the **full 128-sample fold 0**, not just the division-rich subset, with four
times the positives:

| | 6 division-rich samples | **full fold 0** |
|---|---|---|
| GT divisions | 25 | **124** |
| represented by an admitted candidate | 6 (24.0 %) | **25 (20.2 %)** |
| **missing from the pool entirely** | 19 | **99** |
| true forks accepted | 6 / 6 | **25 / 25** |
| their rank, p25/p50/p75 | 0 / 0 / 0 | **0 / 0 / 0** |
| lost to the frame cap | 0 | **0** |
| lost to the global cap | 0 | **0** |
| frames where `n_props > frame_cap` | 12.8 % | 22.1 % |
| candidates accepted | 83.1 % | 68.2 % |
| shipped ranking AUC | 0.531 | **0.541** |

**Every one of 25 true forks that reaches the pool is accepted, and every one ranks
first.** The cap binds more often at full-fold scale (22.1 % of frames) and still loses
nothing, because the ordering puts the real fork on top whenever there is one. So the
division loss is almost entirely *proposal* failure — the first number this project has
had for that — and a better ranker has nothing to win: it would have to reorder cases
that do not exist. The best single feature over the admitted pool is forward divergence
at AUC 0.663, and it is already a gate rather than a ranking term.

**MY OWN FIX WAS THEN FALSIFIED BY THE NEXT MEASUREMENT, BEFORE THE ARM LANDED — record
it as a wrong prediction, not as a near miss.** I argued the single-nearest-orphan rule
was the culprit and built `--div-n-cand` around it. `VOLT_DIV_REJECT` logs every orphan
inside the sister gate with the first gate that refuses it, and it says otherwise. Of the
**18** true second daughters ever considered:

| refused by | n | median `k_rank` |
|---|---|---|
| **`d_par`** (parent distance > 9.0 µm) | **7** | 0 |
| **`symmetry`** (τ > 0.6) | **5** | 0 |
| would pass every gate | 6 | 0 |

**Every true daughter that would pass is already the NEAREST orphan — max `k_rank` 0.**
The true sister is never hidden behind a duplicate in these data, so `n_cand > 1` reaches
nothing and the mechanism I wrote a test for does not occur at this operating point. The
test and the knob are kept because the knob is a no-op at its default and the fixture
pins a real behaviour, but the hypothesis is dead.

**The gates that DO bind are `d_par` and `symmetry`, and they are mis-set rather than
unreachable:**

| gate | shipped | true daughters need | if moved there, false also admitted |
|---|---|---|---|
| `d_par` | 9.0 | min 9.34, **p50 10.41**, max 11.83 | 10 → 3/7 true; **12 → 7/7 true, 4,647 false** |
| `symmetry` τ | 0.6 | min 0.69, **p50 0.84**, max 1.07 | 0.9 → 4/5 true, 710 false; 1.2 → 5/5, 1,535 |

That 1:664 ratio at `d_par` 12 is the same 1:556 base rate measured on 2026-09-12, which
is *why* every widening arm lost: the gates sit exactly at the knee, and with a
non-binding cap the flood is accepted rather than out-ranked.

**But the 2026-09-18 widening arm tested the wrong pair.** It moved parent 9 → 11 and
**sister** 14 → 18 while leaving symmetry at 0.6, on fold 1, under the pre-reference-path
pipeline. Sister was never the blocker — the largest `d_sis` among refused true daughters
is **13.98 µm, inside the shipped 14.0** — and symmetry is an independent blocker for 5
of 18 that it never touched. Parent 11 also only reaches 5 of 7.

**One limitation of the diagnostic, stated so the cost figures are not over-read:** the
reject log records only the *first* failing gate, so a row refused by `d_par` was never
tested against symmetry, the grandchild clause or divergence. The 4,647 false admits are
therefore an **upper bound** on what a looser `d_par` adds to the pool, and divergence —
which does most of the filtering in the admitted pool — would absorb an unknown share.

**AND IT DID. THE GATE WIDENING PAYS, AND IT IS THE LARGEST DIVISION-TERM GAIN IN THIS
PROJECT.** `--div-parent-um 12 --div-symmetry-tau 0.9` on the full fold 0, one variable
against the 0.9463 baseline:

| | baseline | parent 12 + symmetry 0.9 |
|---|---|---|
| score | 0.9463 | **0.9469** |
| **division Jaccard** | 0.1020 | **0.1133** — +0.0113, the highest measured on fold 0 |
| adjusted edge Jaccard | 0.9361 | 0.9356 — −0.0005 |
| node recall | 0.9830 | 0.9831 |
| node ratio | −0.0350 | −0.0345 |

Node recall and the node ratio barely move, so this is a **pure division effect** rather
than a redistribution through the node-count multiplier — the failure mode that made
every node-retention knob reverse between folds. The edge term pays 0.0005 for the extra
forks' false edges and the division term returns 0.0113 at the 0.1 weight, netting
+0.0006.

**So the division term is NOT closed, and the reason the earlier widening failed is now
concrete rather than mysterious.** 2026-09-18 moved parent 9 → 11 and *sister* 14 → 18 at
symmetry 0.6, on fold 1, under the pre-reference-path pipeline. Three of those four
choices were wrong: sister was never the blocker (max `d_sis` among refused true
daughters is 13.98, inside the shipped 14.0), parent 11 reaches only 5 of 7, and symmetry
was left untouched while refusing 5 of 18 on its own. **The lesson is about the
instrument, not the knob: a gate sweep guided by a rejection log found in one arm what
four blind arms missed**, because the log says which clause fires and at what value,
rather than leaving the sweep to guess a direction and a magnitude.

**AND FOLD 1 KILLS IT. The gate widening does NOT ship, and my "largest division gain in
the project" was a fold-0 statement that does not survive weighting.** Same souped
secondary on both sides, so a clean one-variable comparison:

| | fold 0 (85 % of weight) | fold 1 (15 %) | **weighted** |
|---|---|---|---|
| score Δ | +0.0006 | **−0.0044** | **−0.0002** |
| divJ | 0.1020 → 0.1133 (+0.0113) | 0.1311 → **0.0946** (−0.0365) | — |
| adjJ | −0.0005 | −0.0008 | — |

**A wash, and by this file's own rule it was never shippable on a fold-0 number alone:**
*never tune a hyperparameter on a delta smaller than the gap between the two folds.* The
+0.0006 on fold 0 is a fifth of the 0.0019 gap between the folds at this configuration.

The honest reading, stated rather than resolved: fold 0 carries 124 GT divisions and fold
1 about 26, so this file's standing advice is that **fold 0 is the division measurement to
trust**. By that rule the +0.0113 is signal and the −0.0365 is noise on ~26 events. But
the *weighted* score is what the competition grades, fold 1 contributes 15 % of it, and a
swing that large at that n cancels the gain — so the estimate itself is too noisy to act
on. Both things are true, which is exactly why it is a coin flip and not a win.

What survives is the **method**, not the arm: a gate sweep guided by a rejection log found
a real fold-0 effect where four blind sweeps found nothing, and it took one arm instead of
four. The mechanism is also confirmed — divJ moves by ±0.01 to ±0.04 while node recall and
the node ratio stay put, so these gates genuinely control which forks get admitted and
nothing else. That is worth keeping for any future attempt on the term.

Symmetry 1.2 was still running when fold 1 landed; its fold-0 number is now of academic
interest only, since the binding problem is the fold disagreement rather than the level.

Also worth noting: **7 of the 25 GT divisions were never even considered**, i.e. the true
daughter was not an unclaimed orphan within 14 µm of the existing child, or the divider
was not a single-child node. Those are detection- and linking-limited and no division
knob reaches them.

The only fork feature ever measured to separate is the parent's two-frame intensity
*drop* (AUC 0.632), already measured at +0.0008 alone and subsumed by the blend. A
learned fork *classifier* is now a lower priority than it looked, for the same reason:
ranking is not where the loss is.

**AND THE OBVIOUS NEW SIGNAL IS MEASURED AND DEAD: the edge head carries NO division
information.** The natural feature is the head's own probability for the divider→orphan
pair, mirroring the relink's learned bonus, which is our biggest post-processing win at
+0.0073. It was not readable at all — `predict_video` records only pairs that exceeded
threshold 0.5 *and* won the greedy resolution, so an unparented target reaches
post-processing with no learned evidence attached, and an unparented target is exactly
what a second daughter is. It is now recovered (`orphan_parents` out-param →
`orphan_parents_per_frame` → `--div-orphan-probs`), diagnostic-only, and the answer is
flat:

| among the gate-refused candidates | n | with a recorded probability |
|---|---|---|
| **true** second daughters | 12 | **1** |
| true, would-pass (control) | 6 | **0** |
| **false** | 11,470 | 983 (8.6 %) |

**AUC 0.502 — chance.** And the coverage pins the interpretation rather than leaving it
open: recording the top **50** of a ~600-way softmax should reach 8.3 % of pairs at
random, the false candidates appear at 8.6 %, and the true daughters at 1/12 = 8.3 %.
**The divider's rank among an orphan's candidate parents is indistinguishable from
uniform.** The first version of this test recorded only the top 5 and could not separate
"no signal" from "signal below the 5th source"; at 50 that ambiguity is gone.

The mechanism is structural, not a training deficiency: the softmax is over *sources* per
target, so it ranks a cell's possible parents, and a divider 9–12 µm away is out-competed
by dozens of nearer cells. The head was never asked "is this a division", and it has not
learned it as a side effect. So a learned fork classifier would have to be **trained for
the task** — the existing head cannot be repurposed — and that is a new model, not a
feature. Keep the plumbing: it is a no-op at its default, all three probes score
bit-identically (0.9776, divJ 0.2000), and it is the artifact behind this negative.

One counting trap, because the first version of this analysis fell into it: a divider
with out-degree 1 *necessarily* parents one daughter, so "a daughter already has a
parent" is the normal case and evidence of nothing. Subtracting the daughters linked to
the divider itself is what separates the 11 from the 4.

**2026-09-21 (later) — TWO WAYS TO SPEND A SECOND MODEL, both on the linking axis.**
Implemented in `reference_predict`, both no-ops at weight 0, both touching only which
links are believed and never which cells exist.

- **Bidirectional harmonic fusion.** Swapping the frame pair makes the same
  source-softmax answer the complementary question, and the transpose puts it back on
  the same index grid; a weighted harmonic mean is then an agreement test rather than an
  average. **The −0.001 recorded for this on 2026-09-19 is void** — it was measured
  through `edge_head.pair_probs`, the reimplementation now known to lose 0.04.
- **The secondary checkpoint's edge head, which was being thrown away.** `make_blend`
  hand-assembled only the secondary's UNet and 1×1 detection head, ran the trunk once
  *per frame*, and discarded the features — so its trained transformer was never
  consulted at all, on the axis carrying half our remaining loss. `Secondary` now loads
  the whole checkpoint: one encode per window instead of W, the edge head reachable, and
  the coordinate space validated (two models indexed at shared coordinates would
  otherwise read the wrong voxels silently). Proxy at weight 0.5, matched baseline on
  the same card: 0.9457 → **0.9579**, adjJ 0.9314 → 0.9356.

Two combiners, deliberately: detection maps average in *logit* space after
moment-matching (no shared scale); edge probabilities average *arithmetically* (same
question, so an ensemble mean). That is the opposite of the harmonic mean above, which
combines complementary questions.

**Measured on the full fold 0, one variable each against the shipped 0.9361 — and both
edge-side additions buy the edge term by selling the division term.** That shared
mechanism is the useful part, not the individual numbers:

| fold 0, one variable | score | adjJ | divJ | node recall |
|---|---|---|---|---|
| shipped (reference path + ILP prune + TTA + blend 0.15) | 0.9361 | 0.9279 | 0.0816 | 0.9789 |
| **+ bidirectional fusion w=0.15** | **0.9369** | 0.9290 | 0.0791 | 0.9801 |
| + secondary edge head w=0.3 | 0.9355 | 0.9289 | 0.0667 | 0.9790 |
| det_threshold 0.98 instead of 0.965 | 0.9359 | 0.9277 | 0.0820 | 0.9788 |

**Bidirectional fusion is real but a tenth of its published size.** The 2026-09-19
verdict of −0.001 was void (measured through `edge_head.pair_probs`), and the corrected
sign is **positive at +0.0008** — against the reference's +0.0053. Bankable, not worth a
slot alone.

**The secondary checkpoint's edge head is flat-to-negative** (−0.0006), and the 4-movie
proxy that showed 0.9457 → 0.9579 was exactly the effective-n≈2 noise flagged when it
was taken — most of that "gain" was the division term on ~4 events. Two edge heads of
the same architecture trained on the same 199 movies are too correlated for averaging to
reduce much variance, and the secondary is the weaker run. Kept at weight 0 because
`Secondary` also carries the detection blend that does ship.

**`det_threshold` is exhausted**, as the reference's own analysis says: 0.98 lands inside
the noise floor of 0.965, because the misses are displaced peaks rather than
below-threshold ones.

**The shared mechanism, and why it makes the division gate worth more than its own
headroom.** Both edge-side arms move adjJ up (+0.0011, +0.0010) and divJ down (−0.0025,
−0.0149). Averaging edge probabilities makes the dominant assignment more confident,
which is exactly what suppresses a fork's second child. So these two are not
independently small — they are *gated on the division term*, and if loosening forward
divergence raises divJ with headroom to spare, both flip to net positive on top of it.
**Test the combination, not the winners.**

**And the detection objective we default to is NOT the one that trained the field's
best detector — measured, not inferred.** Warm-starting their epoch-400 weights and
reading the loss on 12 iterations of all-train:

| loss the warm-started model is scored under | det | edge |
|---|---|---|
| ours by default (`--det-target gaussian --det-pu-mask`) | **0.0386** | 0.0001 |
| `--det-target voxel --no-det-pu-mask` (the inherited loss) | **0.0023** | 0.0001 |
| **their own recorded epoch-400 values** | **0.0025** | 0.00016 |

The legacy arm matches theirs to 8 % and our default reads **16× higher**, so the two
are genuinely different objectives, not two parameterisations of one. Their
`training_config.json` has no `det_target` key at all — those flags are ours —
confirming they trained the inherited loss for all 400 epochs.

Two things follow and a third does not. **Any continuation of their run must pass
`--det-target voxel --no-det-pu-mask`**, or gradient descent immediately drags the
detector off the optimum it spent 137 GPU-hours reaching; this is what gated the SWA
run rather than being discovered eight hours into it. **And the compatibility held for
the whole run, not just its start**: 8 epochs over 6 h 04 m stayed at `det=0.0023`
against their recorded 0.0025, so continuing their training neither improved nor
degraded the detection objective — which is exactly what makes the resulting checkpoints
usable as soup members rather than as a different model. And **the best detector in this
project was trained with the loss `detection_targets.py` was written to replace**,
which is at least a reason to stop treating the Gaussian+PU default as settled — every
ablation of it ran inside their first epoch (see 2026-09-14). What does *not* follow is
that the legacy loss is better: their run was ~326× longer than any of ours, so budget
is confounded with loss shape and this observation cannot separate them.

**`--unet-weights` is a silent no-op on a full checkpoint** and `--init-weights` is the
fix. The old flag loads into `unet` with `strict=False`, a full checkpoint's keys are
`unet.`-prefixed, so nothing matches and training proceeds from scratch after printing a
reassuring count. `--init-weights` loads trunk, detection head and transformer strictly,
from either a bare state dict or a training checkpoint — which is what continuing the
public 400-epoch run for more soup members requires.

**2026-09-22 — THE KNOB SWEEP AROUND THE BEST CONFIGURATION. Baseline is soup4 + raw
pack50 secondary + edge head 0.5 + bidir 0.25 = fold 0 0.9463, adjJ 0.9361, divJ 0.1020,
node recall 0.9830, node ratio −0.0350.** One variable each, full fold 0:

| arm | score | adjJ | divJ | node recall | ratio | verdict |
|---|---|---|---|---|---|---|
| **souped secondary** (pack50 best + last) | **0.9469** | 0.9367 | 0.1016 | 0.9835 | −0.0293 | **+0.0006, keep** — free |
| edge-head ensemble 0.65 (was 0.5) | 0.9450 | 0.9354 | 0.0960 | 0.9822 | −0.0407 | −0.0013 |
| `min_track_len` 8 (was 6) | 0.9439 | 0.9338 | 0.1008 | 0.9740 | −0.0760 | −0.0024 |

| bidirectional fusion 0.35 (was 0.25) | 0.9459 | 0.9356 | 0.1025 | 0.9833 | −0.0330 | −0.0004 |
| **exact repeat of the baseline** | **0.9463** | **0.9361** | **0.1020** | **0.9830** | **−0.0350** | **bit-identical** |

**The repeat is the cleanest noise-floor measurement this project has made, and it is
tighter than the figure on record.** Re-running the baseline with the division dump
active reproduces **all five metrics exactly**, which establishes two things: the
`VOLT_DIV_DUMP` instrumentation is a verified no-op on the scoring path, and the
same-code noise floor is **below 0.0001**. Read that as a lower bound, not a
replacement for the ±0.0002 already recorded — that figure came from *cross-card*
comparisons, and node counts genuinely do differ by a few between cards (the
`field >= pooled` plateau tie, 29,406 against 29,405 on the same video today). What the
repeat shows is that a handful of tied voxels does not move the weighted aggregate at
four decimals. So the +0.0006 of the souped secondary and the −0.0004 of bidir 0.35 are
both above the floor, while remaining small enough to bundle rather than spend a slot on.

**Bidirectional fusion is confirmed at 0.25 under the final configuration.** Under soup2
alone, 0.25 and 0.35 tied *exactly* (0.9444 both) and the choice rested on fold-1
agreement; under soup4 with the edge pair the plateau tilts to 0.25 by 0.0004. No change
to what ships.

**The edge-head ensemble weight now has a clean interior optimum at 0.5** — 0.3 gave
−0.0006 on the old primary, 0.5 gives +0.0011, 0.65 gives −0.0013. Stop sweeping it.

**And `min_track_len` is exhausted in both directions, with the node-count mechanism
behaving exactly as the metric predicts and still losing.** My reasoning for trying 8 was
that the soup had moved fold 0's ratio from −0.0391 to −0.0350 with *higher* node recall,
so a more aggressive deleter should have room. The multiplier duly improved — ratio
−0.0350 → −0.0760, worth about +0.004 on its own — but node recall fell 0.9830 → 0.9740
and adjJ fell 0.0023, so **J falls faster than the multiplier rises**. Together with 3 and
4 being bit-identical (the ILP has already dropped every component below 4) and 4 ≈ 6 on
fold 0, the knob is closed. The general shape is worth remembering: a node-count knob can
move the multiplier in the predicted direction and still lose, because node recall gates
how much of the linking gets graded (metric fact 3).

**Souping the secondary is the third averaging axis, and FOLD 1 SAYS IT IS FOURTEEN TIMES
BIGGER THAN FOLD 0 SUGGESTED. New best weighted LOEO CV: 0.9466.**

| | fold 0 | fold 1 | **weighted** |
|---|---|---|---|
| soup4 + raw pack50 secondary + edge head 0.5 + bidir 0.25 | 0.9463 | 0.9367 | 0.9449 |
| **+ souped secondary** | 0.9469 | **0.9450** | **0.9466** |
| adjusted edge Jaccard | 0.9361 → 0.9367 | **0.9261 → 0.9319** | — |
| division Jaccard | 0.1020 → 0.1016 | 0.1061 → **0.1311** | — |

**+0.0083 on fold 1 against +0.0006 on fold 0, and it is not the division lottery** —
adjusted edge Jaccard alone moves +0.0058 there, well clear of any noise floor. Fold 1's
divJ rests on ~26 events and should be discounted; its adjJ does not.

**The fold asymmetry reproduces a pattern already in this file, which is what makes it
credible rather than a fluke.** The dual-seed blend measured +0.0033 on fold 0 and
**+0.0092 on fold 1** under the old detect-and-link. Detector-side variance reduction pays
far more on `44b6` — the crowded, 0.77 %-annotated embryo where the detector is weakest —
than on `6bba`, where it is already near its ceiling. That is the same direction as the
standing argument that **our CV understates detector-side changes**, and the hidden set is
a third unseen embryo, i.e. the regime where the larger number applies. Note also that
fold 1's node ratio is −0.1830, deep in the multiplier's bonus region, so extra retained
nodes are nearly free there while fold 0 sits at −0.0345 and cannot absorb them — the same
per-fold asymmetry that reversed every node-retention knob, here working in our favour.

It costs nothing at inference: a different weights file, no extra encode. The earlier
reading of this arm — that +0.0006 was "the size the blend weight predicts", 0.15/0.85 of
the primary soup's +0.0067 ≈ +0.0012 — held for fold 0 and **understated the whole
picture**, because it assumed the effect scales only with the blend weight and not with
how much headroom the fold's detector has. Persisted and md5-verified as
`weights/soup_members/soup_pack_secondary_best.pth`
(`97e5d6eb6a6cd72416d122a7f7351ed8`) with `PROVENANCE.md` beside it.

**2026-09-22 — THE PUBLIC CHECKPOINT INVENTORY, hashed rather than guessed at. Eight
distinct architecture-compatible checkpoints exist, from FIVE independent runs, and the
12 h cap is what decides how many we can use.** Every one is `TemporalUNet3D` + node
transformer, 136 tensors, config `downsample [1,4,4] / window 2 / layers [32,64,128] /
out 32`, so all are drop-in for `--secondary-weights`.

| sha256 (first 8) | bytes | dataset | what it is |
|---|---|---|---|
| `9bac2fa0` | 8,363,159 | `pilkwang/biohub-temporal-unet3d-seed314159-v1` | **pubfull best, epoch 381** — our primary's soup member |
| `ee6c1237` | 25,070,547 | same | pubfull `checkpoint_last`, **epoch 400** — soup member |
| `12f6881e` | 8,363,159 | `pilkwang/biohub-tracking-support-pack-50ep-v1` | **pack50 best** — our secondary |
| — | — | same | pack50 `checkpoint_last`, epoch 402 — soup member |
| `dfb848aa` | 8,363,159 | `shehailrs/biohub-tracking-350ep-public-weight-snapshot` | **epoch 350 of the pack50 run** — its manifest names `…-50ep-v1` *version 9* as the source, so a valid third soup member |
| `347915de` | 8,357,783 | `pilkwang/biohub-tracking-support-pack-v1` | **independent run** — the ORIGINAL support pack (252 downloads against the 50ep pack's 20,689), one checkpoint, no `checkpoint_last` |
| `102f847e` | 8,357,362 | `aashishnegi23/cellmot-ft-detector-biohub` | **independent**, a fine-tuned detector; a byte size no other checkpoint has |
| `4b44531a` | 8,357,783 | `ideaplatsteven/biohub-celltrack-pipeline` | **independent** ("norel7") |

Two datasets that look like new checkpoints and are not: `leonixis/biohub-p314159-pipeline`
repackages pubfull — its `edge_predictor_best.pth` hashes to `9bac2fa0`, identical — and
it usefully corroborates the soup rationale from a second source, carrying
`best_epoch 381, best_score 0.97798, epoch 400, method
unet_transformer_alltrain_seed314159_v1`. `dmitriygerasimov/…-epoch4-0818` is an epoch-4
probe.

**Souping vs blending decides which of these is usable, and the runtime cap decides how
many.** Same-run checkpoints (381/400/350/402 and our SWA epochs) may be **souped**,
which is free at inference. Different runs must be combined at the **activation** level —
the detection blend — and each extra member costs an extra `encode`. The shipped config
already runs **97.4 s/video → 5.38 h** visible and graded reruns have taken ~1.6× the
visible estimate, i.e. ~8.6 h against a 12 h cap. Two more encodes would put that at
risk, so **the plan is one secondary, chosen as the best of the candidates, not a 3-way
blend** — which also needs no code change, since `Secondary` takes exactly one model.
That is why the arms measure candidates *in the same slot* rather than stacking them.

**2026-09-21 — THE ENDGAME PLAN, and it follows from the 29 % split rather than
from the rank table.** Board as of today: 3,778 teams, **Volt rank 2,038 at 0.908**.
The cliff has hardened — **703 teams at exactly 0.947**, 170 at 0.946, 87 at 0.948 —
so 0.9465 is rank 943 and 0.9475 is rank **240**. That table is seductive and it is
the wrong thing to optimise, for one reason: it describes the **29 %** public sample,
and 703 teams sitting on one number are 703 teams whose configuration was chosen
against that sample. On the private 71 % that pile scatters, and nothing we do
changes where it scatters to. Our rank is decided by our own private score.

So the endgame rules are:

1. **Choose both final submissions on CV, not on the public LB.** Ours is a
   71/128-sample held-out-by-embryo estimate; theirs is four movies at effective
   n ≈ 2 with a pre-registered rule that says "local CV only vetoes, it never
   promotes". We have the better instrument and should use it as one.
2. **Weight the folds.** Fold 0 is 85 % of the metric weight. A fold-1 number is
   not the result — this file quoted one for days.
3. **Detector-side changes cannot be ranked on our CV at all**, because the public
   checkpoint trained on all 199 movies including both folds. Measured today: the
   same 8-view TTA scores **−0.015 on the contaminated checkpoint and +0.0071 on our
   honestly held-out one**. The hidden set is a third unseen embryo, i.e. the
   held-out regime, so for anything touching detection the in-domain sign is the one
   to distrust. Use E7_fold1 for those arms.
4. **The two picks should differ in what could go wrong, not in tuning.** One is the
   CV-best pipeline. The other should be the configuration whose failure mode is
   uncorrelated with it — realistically the previous shipped stack, which is a
   different linker end to end, so a systematic problem with the learned edge head
   on an unseen embryo cannot take both slots down.
5. **Identity verification is still open and is a human action** —
   `requiresIdentityVerification: true`, and prize eligibility depends on it.

**2026-09-20 — THE ENDGAME FACTS. Read these before planning any more work.**

**1. The public leaderboard is 29 % of the test set; the final standing is the other
71 %.** Verbatim on the leaderboard page, and `leaderboardPercentage: 29` in Kaggle's
config. See the expanded note under Data. This is the most consequential fact in the
competition and it was absent from this file until today.

**2. Rank is wildly non-linear in score, and the cliff is the whole game.** Measured
from the downloaded leaderboard, 3761 teams, 2026-09-20. We are **rank 2023 at 0.908**,
team "Volt".

| target | rank | places gained |
|---|---|---|
| 0.908 — us | 2023 | — |
| 0.920 | 1627 | +396 |
| 0.940 | 1343 | +680 |
| 0.946 | 1091 | +932 |
| **0.947** | **376** | **+1647** |
| 0.950 | 115 | +1908 |
| 0.966 — 7th, prize cutoff | 7 | — |

Pulled through the API, the pile is even denser than the CSV suggests: **697 teams at
exactly 0.947**, 165 at 0.946, 86 at 0.948 — 862 teams inside 0.002, i.e. **18.5 % of
the entire board on one public-notebook configuration**. Note `scoreTruncationNumDecimals: 3`,
so some of that pile is display granularity rather than identical pipelines.

**3. That cluster is fitted to the 29 %.** Its weights (`pilkwang/…seed314159-v1`) carry
a `split_manifest.json` whose train list is **all 199 train videos**, so any hold-out
drawn from train is in-sample for it; and the two notebooks above the cluster that have
been reverse-engineered *self-declare* `leaderboard_feedback_used_for_configuration:
true`, with per-movie overrides tuned against the four public test movies. One
participant measured local-CV to LB correlation at **r ≈ −0.2** in the 0.93–0.95 band.
Multiple threads predict a large shake-up. **So our position is better than rank 2023
suggests**: every number in this file was measured on held-out CV and we have made 7 LB
submissions, not 86.

**4. Two final submissions are selectable** (`numScoredSubmissions: 2`, rules §2.2.b),
and **the auto-selection criterion is not documented for this competition** — the rules
only say "or automatically selected by Kaggle". Do not assume best-public is taken;
**select both slots by hand**. Suggested shape: one CV-validated safe pick, one
higher-variance pick.

**5. Deadlines, from Kaggle's config rather than recollection.** Entry and team-merger
both **2026-09-22 23:59 UTC**; final submission **2026-09-29 23:59 UTC**; 5
submissions/day; max team size 5. Also `kernelsPublishingDisabledDeadline: 2026-09-22`
— the public notebook frontier freezes then. And **`requiresIdentityVerification: true`**:
prize eligibility needs it, so it should be confirmed done well before the end.

**6. Our own detector has converged and is not the path.** `E7_alltrain` on all 199
samples: epoch 70 → 0.7998, **epoch 99 → 0.8007**. Twenty-nine epochs bought +0.001
against the public pack's 0.8760 through the identical stack. Training to their 400
epochs (~200 GPU-hours, 8 days) is therefore not a gamble worth taking — the curve is
flat, not slow.

**7. We were running the wrong checkpoint.** The public cluster blends *two*: primary
`pilkwang/biohub-tracking-support-pack-50ep-v1` (20,017 downloads) and secondary
`pilkwang/biohub-temporal-unet3d-seed314159-v1`. **We have been using the secondary
alone and had never downloaded the primary.** Different md5, identical architecture and
byte size. Measuring it now.

**2026-09-21 — THE 0.04 IS THE LINKER, AND IT IS OUR REWRITE THAT LOSES IT. Measured
against the reference's own code on the exact movies the public notebooks validate
on.** This is the most important entry in this file since the clamp, and it reverses
three standing conclusions at once.

**The comparable instrument.** The public 0.942/0.948 notebooks carry a local
validator that selects **2 held-out train movies per embryo prefix, division-preferring,
alphabetical** — four movies, effective n ≈ 2 — and scores them with a scorer that is
term-for-term ours (same 7 µm bipartite match, same `TP+FP+FN` weights, same
`J*(1 − 0.1*(N_pred − N_true)/N_true)`, same micro-averaged division Jaccard). Their
title claims **proxy 0.9417 for LB 0.942**. Reproducing their selection rule exactly
gives `44b6_12dfb391, 44b6_267148e4, 6bba_062c8d37, 6bba_07e24132` on a visible run
(the four placeholder test stems are excluded from candidates) and
`…, 6bba_05db0fb1, 6bba_062c8d37` at rerun. **So we can put our pipeline and theirs on
the same four movies and compare directly**, which no measurement in this file had
done.

| on their 4-movie proxy set | score | adjJ | edge J | divJ | node recall | n_pred (44b6_12dfb391) |
|---|---|---|---|---|---|---|
| **reference path, `--use-ilp`** | **0.9299** | **0.9299** | 0.9228 | 0.0000 | 0.9812 | **42,184** |
| reference path, ILP off | 0.8876 | 0.8830 | 0.8927 | 0.0455 | 0.9896 | 50,994 |
| **our shipped stack** | 0.9013 | 0.8888 | ~0.905 | **0.1250** | 0.9809 | 52,070 |
| our stack + ILP prune of the relink | 0.8970 | 0.8858 | 0.8909 | 0.1111 | 0.9759 | — |
| our stack, our edge head + ILP prune | 0.8677 | 0.8600 | — | 0.0769 | **0.9428** | — |

Read the first two rows first. **The ILP is worth +0.047 of adjusted edge Jaccard in
the reference's own implementation**, close to the +0.037 their published ablation
claims and the exact opposite of the −0.057 we recorded on 2026-09-18. It is spent on
*precision*: edge false positives roughly halve (71/29/13/35 → 38/16/3/20) while true
positives move by single digits, and the node count falls 51k → 42k, which also earns
back the node-count multiplier. **Node recall stays 0.9812**, so the over-deletion we
measured was never inherent to the solver.

**What we had wrong, in two layers.**

1. **The input to the solver.** The reference thresholds the source-softmax at 0.5 and
   resolves greedily to ≤1 parent / ≤2 children *first*, then hands the solver that
   graph — so the ILP only decides what to **keep**. We handed it every pair above a
   probability threshold and asked it to choose the topology as well; with
   `node_weight` 0.0 a marginal node is then always cheaper to delete than to keep, and
   recall fell to 0.9419. `ilp_link.py`'s own docstring asserted the opposite ("the
   candidate graph must be the raw, ungated one"; solving a resolved set is "measured
   worthless") — it was reasoning from a *description* of their pipeline rather than
   from their code, and the description was right while the inference was backwards.
   Corrected, with `ilp_link.prune` and a `prune_fn` hook in `run_all`.
2. **And fixing that is not enough, which is the more important half.** Our own
   edge head + our own ILP prune reaches only 0.8600 with node recall **0.9428**, while
   the reference's identical-in-principle arrangement reaches 0.9299 at 0.9812 on the
   same weights and the same movies. So the residual is in how **`kaggle/submit.py`
   computes the probabilities**, not in how they are spent.

**The leading suspect, and it is a defect this file previously cleared.**
`load_detector._forward` duplicates one frame into a fake pair
(`torch.stack([t, t])`) and reads the trunk features from slot 0. That was measured and
justified — *for detection* — at node recall 0.9973 fake against 0.9967 real. **For the
edge head it is not the same question at all**: the head's entire job is to compare a
cell's trunk representation at t against candidates at t+1, and a temporal trunk told
that nothing moved between the two frames has had exactly that signal removed. The
reference runs the U-Net on a real `(t, t+1)` window and reads slots 0 and 1.
`--real-pairs` now does that here, and it is the one-variable test.

**Consequences if it holds.** "The learned transformer edge head is worse" (recorded
twice, believed twice, 0.7632 against 0.7830) and "the ILP loses by 0.05" would both be
statements about the rewrite, not about the method — and this file already reached that
verdict once, on 2026-09-14: *"`kaggle/submit.py` reimplements an inference path that
already exists faithfully in this repo, and the rewrite loses 0.129 … Stop maintaining
the rewrite; ship the path that already works."* We rebuilt the rewrite instead. The
same 0.04 is still there.

**Confirmed on the full fold, and the control is a warning.** Full 71-sample
fold 1, same checkpoint throughout:

| fold 1 (71 samples, 15 % of weight) | score | adjJ | divJ | node recall | node ratio |
|---|---|---|---|---|---|
| shipped stack (LB 0.908) | 0.8760 | 0.8657 | 0.1034 | 0.9841 | −0.071 |
| **reference path + prune + our post-processing** | **0.9124** | **0.9042** | 0.0822 | 0.9700 | −0.238 |

| fold 0 (128 samples, **85 % of weight**) | score | adjJ | divJ | node recall | node ratio |
|---|---|---|---|---|---|
| shipped stack | 0.8953 | 0.8879 | 0.0738 | 0.9859 | +0.066 |
| **reference path + prune + our post-processing** | **0.9296** | **0.9236** | 0.0597 | 0.9764 | −0.043 |

**Weighted LOEO CV 0.8924 → 0.9270, +0.0346** — the largest validated gain in
the project since the post-processing stack, and four times the sum of every
increment banked since. Both folds move by the same amount, which is what a
real effect looks like. The division term gives a little back on both
(0.1034 → 0.0822 and 0.0738 → 0.0597), so ~0.002 of the edge gain is being
spent there and the division pass is worth re-tuning against the new topology.
Runtime 19.6 s/video on a shared card, 1.09 h for a train-sized test set.

Extrapolating is the honest part: our CV has sat **0.0156** below the LB in
level (0.8924 against 0.908), which would put this at ~0.943, while the two
observed *deltas* transferred at only 40–70 %, which would put it at
0.922–0.932.

**SCORED 0.920, rank 2038 -> 1647 of 3781.** Submission `56426110`, kernel v11,
git `fc5090c`. The first time this project has been in the top half of the
board, and +391 places.

**But the transfer rate is the worst of the three we have, and it is the
contamination showing up exactly where the decomposition said it would.**

| change | CV | LB | CV Δ | LB Δ | transfer |
|---|---|---|---|---|---|
| post-processing stack | 0.8687 | 0.905 | +0.086 | +0.059 | 69 % |
| relink learned bonus | 0.8760 | 0.908 | +0.007 | +0.003 | 43 % |
| **reference path + ILP** | **0.9270** | **0.920** | **+0.0346** | **+0.012** | **35 %** |

**And the sign of the CV/LB level gap has flipped.** Our CV sat 0.016–0.036
*below* the LB for every geometric configuration; it is now **0.007 above** it.
That is the predicted consequence of leaning on the contaminated half: the
decomposition put 96 % of this gain on a learned edge head that trained on both
our folds, so the more weight the pipeline puts on that head, the more our CV
inflates. **Operational rule from here: discount any further gain measured on
the edge-head side, and treat the CV as an upper bound rather than a
conservative estimate.** The detector-side arms (blend, TTA) are the ones our
CV still *understates*, for the opposite reason.

So the reference path is genuinely better on an unseen embryo — +0.012 is far
above the board's 0.001 resolution — and roughly two thirds of what it looked
like locally was memorisation. Both halves of that were predicted before the
submission, which is the first time this project has called a transfer rate in
advance. The visible run is clean —
validator PASS, 115,196 nodes and 109,905 edges over the four placeholders —
and it settles the runtime question with a measurement rather than an
extrapolation from this box: **78.5 s/video on the T4, 4.34 h for a
train-sized test set**. That is 4× the 19.6 s/video here, so the A6000-to-T4
ratio for this path is much worse than the 1.5× the detector alone showed, and
4.34 h against a 12 h cap is comfortable but no longer generous. Anything that
multiplies the detector cost again — eight-view TTA, a second seed — has to be
budgeted against that number, not against the old one. But the same pipeline on our **honestly held-out**
detector (E7 fold 1 epoch 99) scores **0.7579 against our own stack's 0.7799**
— i.e. −0.022, the opposite sign.

**Read that control carefully, because it confounds three things.** Our E7
model is 100 epochs against the pack's 400 and scores 0.80 against 0.876
through the identical stack, so its *edge head* is undertrained in exactly the
way its detector is; the reference path also changes the peak window (3-voxel
against our 5) and adds flip TTA, and on our weaker map that combination took
node recall to 0.8789 against 0.9377. So it does not isolate contamination. What
it does establish is that the reference path is **not** unconditionally better —
it is better *with this checkpoint*.

**Tuning the prune, fold 1, against the 0.9124 baseline.** Two knobs, and they
say the same thing:

| disappearance | 0.5 | 1.0 | **1.5** | 2.0 (ref) | 3.0 |
|---|---|---|---|---|---|
| score | 0.9032 | 0.9131 | **0.9134** | 0.9124 | 0.9120 |
| node recall | 0.9821 | 0.9778 | 0.9744 | 0.9700 | 0.9641 |

| min_track_len | 3 | 4 | 6 (shipped) |
|---|---|---|---|
| score | 0.9140 | **0.9140** | 0.9124 |
| node recall | 0.9817 | 0.9817 | 0.9700 |

**Both together are worth nothing on the weighted metric, and the reason is the
node-count penalty.** `min_track_len 4 + disappearance 1.5` on fold 1 gives
**0.9145** (+0.0021, the two partly add), and on fold 0 it gives **0.9294
against 0.9296** — nothing. Weighted, 0.9272 against 0.9270: **+0.0002, exactly
the noise floor.** The mechanism is visible in the node ratio. Fold 1 sits at
**−0.238**, deep in the multiplier's bonus region, so keeping more nodes is free
there; fold 0 sits at −0.043 and keeping more nodes pushes it to **+0.0149**,
across zero and into the penalty. **So every node-retention knob has a per-fold
optimum set by where that embryo sits on the node-count curve**, and tuning them
on fold 1 — 15 % of the weight — and carrying the result to fold 0 is precisely
the mistake this file warns about. Both increments are dropped, not banked.

**3 and 4 are bit-identical on all five metrics**, which is not a coincidence and
pins the mechanism: a chain of L nodes carries L−1 edges worth at most 1.0 each
against a disappearance cost of 2.0, so a 3-node component is break-even at
best and the solver has already dropped every component below 4. The length
filter therefore has nothing left to do below 4, and **its old value of 6 was
cutting tracks the ILP had already paid for**.

**The reference's own retune runs backwards here** — their ablation prices
disappearance 3.0 at +0.004 on 18 of 20 volumes, and it is our worst setting —
because their post-processing does not re-add structure and ours does. And
`min_track_len` 4 *gains* 0.0016 where it *lost* 0.003 on the old pipeline:
once the ILP deletes nodes, every later stage that also deletes is cutting the
same tracks twice. But 0.5 collapses to 0.9032, so the prune is doing real work
and is merely slightly over-tuned; the optimum is interior and flat between 1.0
and 2.0. Both gains are ~0.001, i.e. above the 0.0002 noise floor and below the
0.005 that justifies a submission slot on its own — bank them.

**The dual-seed blend is the largest increment on top of the reference path, and
it gains on both terms.** Fold 0, against the shipped 0.9296:

| blend weight | 0 | 0.10 | **0.15** | 0.20 | 0.30 | 0.45 |
|---|---|---|---|---|---|---|
| score | 0.9296 | 0.9347 | **0.9361** | 0.9345 | 0.9336 | 0.9310 |
| adjJ | 0.9236 | 0.9275 | **0.9279** | 0.9267 | 0.9259 | 0.9246 |
| divJ | 0.0597 | 0.0723 | **0.0816** | 0.0775 | 0.0775 | 0.0639 |
| node recall | 0.9764 | 0.9781 | 0.9789 | 0.9780 | 0.9772 | 0.9770 |
| node ratio | −0.0430 | −0.0404 | −0.0391 | −0.0390 | −0.0402 | −0.0407 |

**The optimum moved, and we submitted the old one.** 0.3 was found under the
old detect-and-link on fold 1 (0.2 → 0.8765, 0.3 → 0.8852, 0.4 → 0.8816); under
the reference path on fold 0 the peak is **0.15**, worth **+0.0025** over the
0.30 that went out as `56432361`. Six points, unimodal, and adjJ and divJ and
node recall all peak together at 0.15 — which is what a real optimum looks like
rather than a lucky draw. A knob tuned against one linker does not carry to
another; that is now the third instance, after the division gates and the
disappearance cost.

**+0.0040 on the fold that carries 85 % of the weight**, and unlike the
node-retention knobs it moves adjusted edge Jaccard *and* the division term
while barely touching the node count — so it is not trading one against the
other. `reference_predict.make_blend` runs it inside the copied path, blending
whichever frame of the window the caller names rather than a duplicated one.

**Confirmed on both folds at w = 0.15: fold 0 0.9361, fold 1 0.9168, weighted
0.9332** — against 0.9308 for the w = 0.30 configuration sitting in the graded
queue and 0.9270 for the one that scored 0.920. Fold 1 prefers 0.15 to 0.30 too
(0.9168 against 0.9150), so the retune is not a fold-0 artefact the way the
node-retention knobs were.

Fold 1 with the w = 0.30 blend and the appearance term is **0.9150** (adjJ 0.9042 -> 0.9092, a +0.005 jump, while divJ
falls 0.0822 -> 0.0580 on that fold's handful of division events). Weighted with
fold 0's 0.9336 that is **0.9308 against the shipped 0.9270, +0.0038.**

**Superseded by kernel v13** (git `b5f5e0f`), the same path at the swept w = 0.15
— visible run clean, **102.3 s/video, 5.66 h** for a train-sized test set. The
graded reruns have been taking roughly 1.6x their visible-run estimate, so
expect ~9 h: inside the 12 h cap, and the point at which nothing else that
multiplies detector cost can be stacked on top.

**Submitted as `56432361`** (kernel v12, git `04f8bf3`). Visible run clean —
blend active, validator PASS, 116,006 nodes — and **90.6 s/video, 5.01 h for a
train-sized test set** against the 12 h cap, up from 4.34 h without the blend.
Worth noting for the endgame: the 0.920 rerun took ~7 h of wall clock against a
4.34 h estimate, so a 5.01 h estimate is probably ~8 h in practice. That is
still inside the cap but it is the first configuration where the margin is
worth thinking about, and it rules out stacking anything else that multiplies
the detector cost.

Two reasons to expect the leaderboard to like it more than this number does.
It is **detector-side**, so our contaminated fold understates it by the argument
above; and it is variance reduction across two independently-seeded nets, which
pays more on an unseen embryo than in-domain. Under the old detect-and-link it
was +0.0033 on fold 0 and **+0.0092 on fold 1**.

**What does move the division term is the appearance feature,**What does move the division term is the appearance feature, and it is the one
increment today with a clean mechanism.** Fold 0, against the shipped 0.9296:

| `--div-rise-weight` | 0 | 10 | 30 |
|---|---|---|---|
| score | 0.9296 | **0.9304** | 0.9303 |
| divJ | 0.0597 | **0.0669** | 0.0664 |
| adjJ | 0.9236 | 0.9237 | 0.9237 |
| node recall | 0.9764 | 0.9764 | 0.9765 |
| node ratio | −0.0430 | −0.0430 | −0.0430 |

**And it is subsumed by the blend, so it does not ship.** Fold 0: blend alone
0.9336, blend + rise 0.9337. Both work on the division term and the blend
already takes divJ 0.0597 -> 0.0775, so there is nothing left for the ranker to
reorder (0.0775 -> 0.0784). The feature is real -- AUC 0.632, the only fork
feature this project has found that separates -- and worth +0.0008 on its own,
but it is not additive with the thing we would ship anyway. Keep the code and
the measurement; ship `--div-rise-weight 0`.

Flat from 10 to 30, so the term saturates: by weight 10 it already dominates
the geometric ordering wherever it disagrees, and tripling it reorders nothing
further. No more sweeping.

**adjJ, node recall and the node ratio are unchanged to four decimals** — this
only reorders which forks win the cap, so it is a pure division-term effect and
**cannot reverse between folds the way the node-retention knobs did**, since
those reversed through the node-count multiplier. +0.0008 of score for +12 %
relative on divJ, and 10 was a first guess at a weight for a feature measured
at AUC 0.632, so the sweep is worth finishing.

**Not the caps.** divJ fell 0.1034 → 0.0822 when the prune landed and the
obvious suspect was the fork caps, which are fractions of node and edge counts
that the prune cut by a fifth. Scaling both by 1.25 gives 0.9123 against
0.9124, divJ 0.0800 against 0.0822 — nothing. The division loss under the new
linker is structural, not a cap artefact.

**The decomposition, and it puts almost all the gain on the memorisation-exposed
half.** Full fold 1, one variable at a time:

| arm | score | adjJ | node recall |
|---|---|---|---|
| our detection + our relink (shipped) | 0.8760 | 0.8657 | 0.9841 |
| **reference detection + our relink** | **0.8776** | 0.8705 | **0.9931** |
| reference detection + their edge head + ILP | **0.9124** | 0.9042 | 0.9700 |

**The detection half is worth +0.0016. The linker half is worth +0.0348** — 96 %
of it. Their peak picker and flip TTA do produce a better node set (recall
0.9931, the highest measured in this project), and our geometric relink simply
does not convert it; what converts it is the learned edge head choosing the
topology, with the ILP pruning what it over-commits.

**And the specific risk is nameable.** Both the 0.8760 baseline and the 0.9124
candidate use the same contaminated detector, so the detector half is not what
differs. What differs is the linker: ours is geometric, theirs is a learned edge
head choosing the topology — and that head is part of a checkpoint that trained
on these 71 videos. If its advantage is association memorised from them, the
hidden embryo will not reproduce it. **There is no local instrument that can
settle this**, because the public checkpoint saw all 199 training videos and
there is no held-out fold for it. The leaderboard is the only test, which makes
this exactly the kind of genuinely uncertain change a submission slot is for —
and the 0.908 submission stays banked as one of the two final picks regardless.

**The gain is concentrated in the tail, which is the shape that should
transfer.** Per sample, adjusted edge Jaccard, ours against the reference path
plus our post-processing:

| sample | ours | reference path + our stack | Δ |
|---|---|---|---|
| `6bba_062c8d37` | 0.9908 | 0.9864 | **−0.004** |
| `44b6_12dfb391` | 0.8971 | 0.9454 | +0.048 |
| `6bba_07e24132` | 0.7637 | 0.8483 | +0.085 |
| `44b6_267148e4` | 0.7297 | 0.8647 | **+0.135** |

It does nothing on the sample we already score 0.99 on and the most on the two
worst. That matters for the hidden embryo specifically: this file's own
decomposition found that lifting the worst 20 samples of fold 1 to our p90
would move weighted adjJ 0.863 → 0.919, and a third unseen embryo puts *more*
samples in that tail, not fewer. A change that only raised the mean would be
the one to distrust.

**Where our stack is genuinely ahead: divisions.** The reference-with-ILP scores divJ
**0.0000** on this set — its `division_weight` 1.2 is a cost and it prunes the forks —
against our **0.1250**. At the 0.1 weight that is +0.0125 of score we hold and they do
not, and it is why `0.9299 + 0.1 × our divisions ≈ 0.942` reproduces their proxy almost
exactly. **The target configuration is their linker with our division pass**, not one
or the other.

**Runtime, measured rather than feared.** The reference path costs ~10 s/video without
the ILP and ~25–30 s/video with it on an A6000, the solver being single-threaded SCIP.
Extrapolating from the T4 detector measurement that is roughly 3 h for a train-sized
199-video test set against the 12 h cap. `tracksdata`, `ilpy`, `pyscipopt` and
`rustworkx` wheels all ship in the public pack, so the offline constraint that kept the
ILP out is not a constraint.

**A second measurement error of the same shape: our "TTA" is not their TTA — and TTA
is not a loss at all once the ruler is fixed.** Their `PredictConfig.det_tta` is
**flip-xy, four views**; ours averaged the full **eight-view D4 group**, rotations
included, and a 90° turn through an anisotropic PSF is a far stronger perturbation
than a mirror. Separately, every detector-side number we have was measured on a
checkpoint that trained on both our folds. On the **honestly held-out** detector
(E7 fold 1 epoch 99, trained on `6bba`, tested on `44b6`), full fold:

| arm | score | adjJ | divJ | node recall |
|---|---|---|---|---|
| no TTA | 0.7799 | 0.7752 | 0.0462 | 0.9377 |
| **4-view flips** (theirs) | **0.7864** | 0.7775 | 0.0896 | 0.9369 |
| 8-view D4 (ours) | **0.7870** | 0.7801 | 0.0690 | 0.9410 |
| feature TTA | 0.7794 | 0.7748 | 0.0462 | 0.9378 |

**The sign flips with contamination** — the same 8-view average is −0.015 on the
public checkpoint and +0.0071 here — and **the flip subgroup captures essentially all
of it at half the cost**, which is what the reference ships and therefore what the
reference path runs. Feature TTA stays out: neutral on the honest detector too. The
general rule this establishes: **no detector-side change can be ranked on our CV**,
because the hidden set is a third unseen embryo and that is the held-out regime.

**Also settled today: the dual-seed blend transfers to the heavy fold.** w = 0.3,
moment-matched in logit space with the 0.90 retention guard: fold 0 **0.8953 → 0.8986**
(adjJ 0.8879 → 0.8913), fold 1 **0.8760 → 0.8852** (adjJ 0.8657 → 0.8735), weighted
**0.8924 → 0.8966**. Both folds gain on adjJ, so it is not the division lottery. Weight
0.2 and 0.4 bracket it (fold 1: 0.8765 / 0.8816), and 0.5 and 0.8 fall back — a clean
interior optimum. `kaggle_env.find_secondary_weights` looks for
`secondary/edge_predictor_best.pth` beside its config, and `find_weights` now refuses
to return that file as the primary.

**And a new division feature, the first one that separates.** `CLAUDE.md` has recorded
since 2026-09-12 that geometric fork features overlap at the quartiles and named
appearance as the untested axis. Measured over **all 151 GT divisions against 1,729
same-frame controls** (`scripts/division_appearance.py`): absolute brightness barely
separates (AUC 0.53–0.59), but the **two-frame intensity *rise* does, inverted** — a
parent about to divide has *lost* ~10 % of its peak intensity relative to itself two
frames earlier (median 0.901) while a non-dividing cell is flat (1.002), giving
**AUC 0.632**. My stated prediction was the opposite (condensing chromatin should
brighten), and the direction that actually shows up is the one the mechanism implies:
the mass is already splitting, so the single centroid dims. It is the strongest fork
feature measured in this project against a field of ~0.50, and it is free at inference
— the volumes are already in memory. Not yet wired into `add_safe_divisions`, whose
ranking is still `d_par + 0.15 * d_sis`.

**2026-09-19 — SEVEN POST-PROCESSING IDEAS, ONE WIN, AND THE PIPELINE IS AT A
LOCAL OPTIMUM IN EVERY DIRECTION TESTED.** All on the full 71-sample fold against
the shipped **0.8760** (LB 0.908). Recording the negatives in one place because
the next session's cheapest win is not re-running them:

| arm | score | Δ |
|---|---|---|
| **shipped: postprocess + learned bonus 15.0** | **0.8760** | — |
| bidirectional harmonic fusion, w 0.50 / 0.30 / 0.15 | 0.8754 / 0.8752 / 0.8750 | −0.001 |
| edge softmax temperature 2.0 (smooth) | 0.8750 | −0.001 |
| edge softmax temperature 0.5 / 0.25 (sharpen) | 0.8737 / 0.8736 | −0.002 |
| feature TTA (D4 average of U-Net features) | 0.8646 | **−0.011** |
| ILP over learned candidates | 0.8190 | **−0.057** |
| finer peak picker, separation 3.0 | 0.8527 | −0.023 |
| 8-view detection TTA | 0.8609 | −0.015 |

**A prediction of mine was falsified by its own control, which is the part worth
keeping.** Two smoothing operations lost (fusion −0.001, feature TTA −0.011), so I
reasoned the relink cost must read the *ranking* of the edge probabilities rather
than their calibration, and predicted that **sharpening** would help — with
temperature 2.0 as a control that should lose. Measured: sharpening lost *more*
(0.8736 at T=0.25) and the control lost *least* (0.8750). The mechanism was wrong.
What the sweep actually shows is that the model's trained calibration is already
matched to the micrometre cost scale, and both directions degrade it.

**So we now hold two orthogonal 1-D sweeps with interior optima at the shipped
values** — bonus magnitude (0.8687 at 0, rising to 0.8760 at 15, falling to 0.8738
at 100) and probability shape (temperature 1.0 beating both 0.25 and 2.0). Two
independent axes both peaking where we sit is real evidence of a local optimum
rather than an unexplored plateau.

**The unifying caveat on every one of these, and it is the transferable lesson.**
Each of these components is *measured positive in the reference's own ablation*:
ILP +0.037, TTA +0.009, fusion +0.005, feature TTA the 0.941→0.946 step. None
transferred. The reason is consistent: their deltas are measured inside a pipeline
whose linker is the ILP, and ours is a geometric relink that already reaches node
recall 0.984. Components that repair a weak linker have nothing to repair here,
and components that smooth probabilities degrade a cost that is already
well-scaled. **Do not port another component on the strength of its published
delta alone — check first what it is repairing, and whether we have that
defect.**

**2026-09-18 — 0.905 → 0.908, and the CV has now ranked correctly twice.**
Submission `56340649` (kernel v10, git `308c77a`), the relink learned bonus at
15.0.

| change | CV | LB | CV Δ | LB Δ |
|---|---|---|---|---|
| threshold replaces the node budget | 0.7830 | 0.846 | — | — |
| post-processing stack | 0.8687 | 0.905 | +0.086 | +0.059 |
| **relink learned bonus 15.0** | **0.8760** | **0.908** | +0.007 | +0.003 |

**The gap is stable and the delta compresses.** CV sits 0.032–0.036 below the LB
in level, consistently, because fold 1 tests on `44b6`, our worst embryo. But the
*change* transfers at only 40–70 %: +0.086 became +0.059, +0.007 became +0.003.
Two points is not a calibration curve, but the direction has been right twice and
the compression was present both times.

**Operational consequence: a CV gain below ~0.005 is not worth a submission
slot**, because it lands inside the leaderboard's own resolution once compressed.
Bank such changes and ship them together with the next real one — which is not a
licence to bundle blindly, since the 0.578 submission bundled four changes and
became unattributable; it means a sub-0.005 arm should be *held*, not submitted
alone.

**2026-09-18 — THE LEARNED BONUS: 0.8687 → 0.8760, and it is the transformer we
deleted twice.** Full 71-sample fold, one variable, reproducible (the 15.0 arm
repeated on a different GPU model gives a bit-identical 0.8760):

| bonus | 0 | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 | 8.0 | **15.0** | 30.0 | 100.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| score | 0.8687 | 0.8697 | 0.8706 | 0.8706 | 0.8722 | 0.8737 | 0.8753 | **0.8760** | 0.8748 | 0.8738 |
| adjJ | 0.8579 | 0.8587 | 0.8593 | 0.8609 | 0.8627 | 0.8644 | 0.8658 | 0.8657 | 0.8639 | 0.8630 |

A clean interior optimum at **15.0**, which is **15× the reference's 1.0** — and
the units say why. The bonus is subtracted from a cost in *micrometres*, and
median inter-frame motion is 1.82 µm, so a bonus of 1.0 can only break near-ties.
At 15 the geometric term is nearly vestigial and the cost has degenerated to
**Hungarian on learned probabilities within a motion gate**. Past that it decays
again: the gate stops discriminating and wrong-but-confident pairs win.

**This is the third way to spend the edge head's probabilities and the only one
that works**, which retires two earlier conclusions at once:

| | score | why it fails |
|---|---|---|
| as edges, resolved greedily | 0.7830 | softmax abstention leaves a node with no parent; pruning deletes it (recall 0.9808 → 0.9390) |
| as ILP candidates | 0.8190 | `node_weight` is 0.0, so deleting a node is free (recall 0.9419) |
| **as a bonus on the geometric cost** | **0.8760** | geometry still assigns; the model only reorders within the gate, so it can neither abstain a node into deletion nor drop one |

So "the learned edge head is worse, drop it" (measured twice, believed twice) was
a statement about *our two ways of using it*, not about the head. **The general
lesson: when a component loses, separate the component from the interface before
concluding it has no signal.**

**2026-09-18 — THE ILP HONESTLY MEASURED: −0.05, and the edge head's
probabilities pay only as a COST BONUS.** Once the head actually loaded (see the
measurement error below), the three ways to spend the same probabilities separate
cleanly on the full 71-sample fold:

| how the edge-head probabilities are spent | score | adjJ | divJ | node recall |
|---|---|---|---|---|
| **as a bonus on the relink's geometric cost** | **0.8706** | **0.8593** | **0.1129** | **0.9841** |
| not at all (shipped) | 0.8687 | 0.8579 | 0.1077 | 0.9830 |
| as edges, resolved greedily | 0.7830 | 0.7819 | 0.0110 | 0.9808 |
| as ILP candidates, threshold 0.48, gaps on | 0.8190 | 0.8167 | 0.0238 | 0.9419 |
| as ILP candidates, threshold 0.10, no gaps | 0.8093 | 0.8027 | 0.0667 | 0.9548 |

**The ILP loses by 0.05, not the 0.008 the broken runs suggested, and node recall
is the mechanism**: 0.9419 against 0.9830. `ILPSolver`'s `node_weight` defaults to
0.0, so deleting a node is free while an edge is worth at most 1.0 against a
disappearance cost of 2.0 — every marginal node is cheaper to delete than to keep.
The reference is insulated because its motion relink runs *afterwards* and
re-derives edges from geometry, repairing the over-deletion. With the relink off —
their own best arm — the cost balance has to carry itself, and ours does not. This
does not refute their +0.037; it says the ILP is not transplantable into a
pipeline whose linker already reaches node recall 0.983.

**What does pay is the third option, which we had never tried.** The reference's
cost is `motion + 0.05*raw − MOTION_RELINK_LEARNED_BONUS * prob` with the bonus at
**1.0**; we shipped the first two terms and nothing else. As a bonus the head
cannot cause either failure above — geometry still decides the assignment, and the
model only pulls a confident pair forward by at most one micrometre of apparent
distance. **+0.0019 over the shipped config**, which is ten times the ±0.0002
cross-card noise floor, and adjJ, divJ and node recall all move the same way,
which is what a real effect looks like.

**2026-09-18 — A MEASUREMENT ERROR WORTH MORE THAN THE RESULT: six "ILP" arms
ran with no edge head and measured something else entirely.** `submit.run` loaded
the transformer edge head only `if learned_edges`, and the ILP arms did not pass
that flag — so `edge_head` was `None`, the guard `ilp and edge_head is not None`
was False, and **the ILP never ran**. What those arms actually varied was
`--no-gaps`, which they all carried.

**The tell was in the numbers and I missed it.** The six arms produced exactly two
distinct outcomes, and they cluster by **GPU card** (cards 0 and 7 → 0.8605,
card 3 → 0.8601), not by parameter. Six different cost settings collapsing onto
two values is the signature of a parameter that is not reaching the code; the
cross-card noise floor established the same day (±0.0002) is what the two clusters
actually measure. **Rule: when an arm is insensitive to its own parameter, check
that the parameter arrives before explaining why it does not matter.** I wrote a
mechanism for the null result instead — that the solver spends its freedom in the
unannotated 96.5 % the metric ignores — which is plausible, may even be true, and
was not what those runs showed.

Fixed two ways: the head now loads whenever *any* consumer asks for it
(`learned_edges or ilp or learned_bonus`), and a checkpoint with no `transformer.*`
weights now **raises** instead of silently degrading to the geometric linker and
reporting the result as that arm.

**The real finding from those runs, correctly labelled: gap closing is worth
+0.008 here.** 0.8687 with it, 0.8605 without, on the full fold. That contradicts
the reference's own clean-split ablation, which prices gap closing at **+0.0004**
— and the difference is explicable: their pipeline has the ILP choosing topology,
so a gap is usually already bridged, while ours relies on the relink and needs the
repair. Another instance of the standing lesson that their component deltas are
measured at their operating point, not ours.

**2026-09-18 — THE FINER PEAK PICKER: +0.008 node recall, −0.016 score, and it
re-confirms a trade-off this file already recorded.** We diverged from the
reference by construction and nobody had noticed: their `pool_kernel_from_um` is
`round(µm/voxel)` forced odd, so `pool_kernel_um` of **3.0, 4.0 and 5.0 all give a
3-voxel window** (min separation 3.25 µm), while our `_peak_window` is
`2·ceil(sep/iso) − 1`, so our shipped `separation_um 4.0` gives a **5-voxel**
window (4.875 µm). We suppress 1.5× more aggressively than the pipeline we forked.

| | shipped, sep 4.0 | sep 3.0 |
|---|---|---|
| score | **0.8687** | 0.8527 |
| adjJ | 0.8579 | 0.8483 |
| **node recall** | 0.9830 | **0.9911** |
| node count ratio | −0.071 | **+0.031** |
| division Jaccard | 0.1077 | **0.0435** |

Node recall 0.9911 is the highest measured in the project, so the mechanism is
real: at 4.875 µm we were merging peaks the reference keeps, and sister separation
one frame after a division has p5 5.41 µm. **But it loses on the other two terms
and the division half was predictable from this file.** "Over-detection is free on
edges but taxed on divisions" — the fork caps are *fractions* of node and edge
counts (0.76 % per frame, 0.375 % global), so more nodes admit more false forks,
and divJ halved. The node-count bonus also flipped from −0.071 to +0.031, worth
about 0.01 on its own. The lesson is the one in metric fact 3's consequences:
**detector and linker cannot be swept independently**, and a recall gain has to be
paid for somewhere.

**2026-09-18 — 0.846 → 0.905 ON THE LEADERBOARD, AND THE HARNESS RANKED IT
CORRECTLY FOR THE FIRST TIME.** Submission `56332095` (kernel v9, git `0b978aa`)
scored **0.905 public**, against the field median of 0.902. First time this
project has been above the middle of the board.

| | CV fold 1 | LB | gap |
|---|---|---|---|
| threshold 0.5, one-pass linker | 0.7830 | 0.846 | +0.063 |
| **post-processing, threshold 0.965** | **0.8687** | **0.905** | +0.036 |

**This file has twice recorded that our CV tracks the leaderboard in level but
fails to rank two configurations. It ranked this one.** CV predicted +0.086 and
the board gave +0.059 — same sign, same order of magnitude, and the config that
won on CV won on the board. The LB sits *above* CV in both rows because fold 1
tests on `44b6`, the crowded 0.77 %-annotated embryo that is our worst fold, so a
fold-1 number is a conservative estimate rather than an optimistic one. That is
worth knowing before spending a submission slot on confirmation: the harness can
now carry a ranking decision, and the remaining slots are better spent on
genuinely uncertain changes.

Progress in context: 0.588 (2026-09-11) → 0.846 (2026-09-16) → **0.905**. The
top-25 % line is 0.944 and top-10 % is 0.946, so **+0.041 remains.**

**The division term decomposed, and the reference's gates are already at a local
optimum.** Full fold, shipped config, forks labelled against GT:

| | |
|---|---|
| forks emitted | 4,053 |
| **charged as FP** | **40 (1.0 %)** — the rest fall in unannotated regions and are ignored |
| TP / FN | 7 / 19 → recall **0.269** |
| precision | 0.149 |

FP being 6× TP suggested tightening should pay. **It does not** — three arms, all
at or below the shipped 0.8687:

| arm | score | divJ |
|---|---|---|
| **reference defaults** | **0.8687** | 0.1061 |
| symmetry τ 0.6 → 0.75 | 0.8682 | **0.1061 — identical** |
| forward divergence 2.25 → 3.5 µm | 0.8641 | 0.0625 |
| wider gates (parent 11.0, sister 18.0) | 0.8679 | 0.1000 |

Two things to take from it. Tightening a structural gate removes true forks at
least as fast as false ones, so the average precision of 0.149 does not describe
the *marginal* candidate. And **the symmetry test is not the active filter** — at
τ 0.75 it rejected exactly zero candidates, leaving divJ bit-identical, while
divergence halved it. So do not sweep symmetry again; divergence at 2.25 µm and
the mutual-nearest-orphan rule are what admit the candidates.

**Consequence: the remaining division loss needs a better ranker or a new
feature, not a different threshold.** That converges with the 2026-09-12 finding
that geometric features do not separate real from false forks at the quartiles —
but that study was run at divJ 0.015 behind a 6 µm gate, and this one is at
divJ 0.106 with the gate open, so the conclusion now rests on a real operating
point rather than on 3 proposed forks. Untested and still open: appearance
features (intensity drop, nuclear shape), which is the one axis that study
explicitly did not cover.

**Our own detector, measured honestly for the first time against the public one.**
E7 fold 1 epoch 95 (trained on `6bba`'s 128 samples, tested on `44b6`) through
the identical post-processing stack:

| detector | score | adjJ | divJ | node recall |
|---|---|---|---|---|
| public 400-epoch pack (trained on **all 199**, so contaminated) | 0.8687 | 0.8579 | 0.1077 | 0.9830 |
| **ours, E7 epoch 95, honestly held out** | 0.7755 | 0.7704 | 0.0508 | 0.9402 |

The public number is an upper bound — it has seen these 71 videos — but a
**0.043 node-recall gap** is too large to be only contamination. Our detector is
genuinely behind, and the public checkpoint stays the submission detector (legal:
pre-trained models are explicitly allowed). The apples-to-apples test of our
recipe is `E7_alltrain_100ep`, which trains on the same 199 samples and is
therefore contaminated the same way; it is at epoch 55/100.

**2026-09-18 — THE POST-DETECTION STACK IS THE GAP, AND IT IS WORTH +0.076.
Measured on the full 71-sample fold through the submission path, one variable at
a time, same public checkpoint throughout:**

| arm | score | adjJ | divJ | node recall | node ratio |
|---|---|---|---|---|---|
| threshold 0.5, one-pass gate 8 µm (**scored 0.846 on LB**) | 0.7830 | 0.7819 | 0.0110 | 0.9808 | +0.0424 |
| threshold 0.965 | 0.7931 | 0.7920 | 0.0110 | 0.9805 | −0.0635 |
| **+ full post-processing — SHIPPED** | **0.8687** | **0.8579** | **0.1077** | 0.9830 | −0.0708 |
| post-processing, threshold 0.93 | 0.8649 | 0.8556 | 0.0938 | 0.9829 | −0.0448 |
| post-processing + 8-view TTA, threshold 0.965 | 0.8609 | 0.8561 | 0.0484 | 0.9788 | −0.1489 |
| post-processing + 8-view TTA, threshold 0.93 | 0.8593 | 0.8543 | 0.0492 | 0.9802 | −0.1083 |
| 8-view D4 TTA alone, threshold 0.965 | 0.7916 | 0.7907 | 0.0088 | 0.9779 | −0.1430 |
| learned transformer edge head | 0.7632 | 0.7614 | 0.0178 | 0.9390 | −0.2232 |

`src/volt_biohub/postprocess.py` is the public 0.942–0.948 notebooks' shared
post-processing cell, ported in their order (which is load-bearing): two-pass
Hungarian with velocity extrapolation → single-frame gap closing → structurally
gated divisions → short-track filter → line-fit smoothing.

**The division term finally moved, and the reason our gate was wrong is
embarrassing.** divJ has sat at ~0.01 since the first submission against a term
worth 0.1. Our `DIVISION_GATE_UM` is 6.0 µm; measured GT sister separation one
frame after a division has a **median of 10.4 µm** (p90 13.0). We were rejecting
most real divisions by construction, which is also why the 2026-09-12 fork-ranker
study found only 3 real forks proposed in 14 samples — **the proposal ceiling that
study measured was an artefact of the gate, not a property of the detections.**
The reference opens the gates (parent 9.0, sister 14.0) and defends precision
structurally instead: mutual-nearest orphans, forward divergence ≥2.25 µm, a
symmetry test at τ 0.6, and hard per-frame (0.76 %) and global (0.375 %) caps.
That combination gives divJ 0.1077 where opening our gate alone gave 390 false
forks and lost score. **Ranking was the right diagnosis; the gate was still
blocking the candidates.**

**Two components are deliberately NOT in it, and both were measured rather than
assumed.** The ILP: those notebooks run it and then **discard its edge set
wholesale**, keeping only per-pair probabilities as a small additive bonus in the
Hungarian cost, so their shipped edges come from the two-pass Hungarian. Task #17
is therefore dropped, not deferred. The learned transformer edge head: 0.7632
against Hungarian's 0.7830 on the same fold. It emits 25–27k edges/video, so the
old "emits zero edges" claim was false, but abstaining leaves isolated nodes that
pruning then deletes and node recall falls 0.9808 → 0.9390. The +0.073 recorded
for it on 2026-09-14 came from 8 contaminated videos at effective n ≈ 3. Kept
behind `--learned-edges` as the artefact behind the number.

**8-view TTA LOSES, and the mechanism matters because every public notebook above
0.942 uses it.** Three arms: alone it is flat (0.7916 vs 0.7931), and on top of
post-processing it costs 0.008 at threshold 0.965 and 0.009 at 0.93. **The damage
is concentrated entirely in the division term** — divJ 0.1077 → 0.0484/0.0492,
i.e. it halves — while adjJ barely moves (0.8579 → 0.8561). Averaging logits over
the D4 symmetries of (y, x) acts like a blur in that plane, and sister separation
one frame after a division has **p5 5.41 µm on a 1.625 µm grid**, so the two
daughters merge into one peak under the average and the fork stops being a
candidate. Retuning the threshold underneath does not recover it, which rules out
"the operating point moved" as the whole story (that effect is real too — node
ratio −0.149 against −0.071).

**So TTA is not wrong in general, it is wrong at this resolution.** It is the
first concrete argument for `--downsample 1,2,2` since the idea was raised: on a
0.8125 µm grid the daughters stay resolved under the blur, and TTA plus a finer
grid is a coherent pair where either alone is not. That makes task #19 worth
doing after all, for a reason other than the one it was filed under.

**And on Kaggle it is not free, which is a correction to a claim made twice in
this file.** "Runtime is free, 0.66 h against a 12 h cap" was measured on an
A6000. The detector measured **13.3 s/video on the real T4 kernel**, so eight
views is ~5.9 h before post-processing, against ~1.6 h without. Extrapolate
inference cost from the T4 measurement in `kaggle/README.md`, never from this
box.

**The threshold is a real interior optimum, not an inherited constant.** 0.965
beats 0.93 by 0.004 under post-processing, so the value taken from the public
notebooks survives being checked rather than being assumed.

**Two bugs in the new stack, both caught by `validate_submission` rather than by
the score** — which is the argument for the validator earning its place in the
scoring harness and not only at the save site. Line-fit smoothing pulls a border
node marginally outside the grid, and a z/y/x of −0.3 becomes **−1** once the
writer multiplies by the downsample factor: 11 nodes in two videos, and the file
is rejected outright. And gap closing and the division pass each add edges under
their own local guard, so neither sees that the other has claimed a node — the
scorer does **not** error on a merge, it silently collapses the duplicates and
keeps the lowest edge id, so an in-degree of 2 is a free false positive with no
visible failure. In-degree is now capped at 1 and out-degree at 2, longest links
dropped first.

`tests/test_tta_and_postprocess.py` pins the TTA inverse transforms by the one
property that holds for *any* model: a full-group symmetry average is exactly
equivariant, so rotating the input must rotate the output. A "does it run" test
passes with the inverses omitted, which is precisely the bug that silently
destroys a detection map by averaging misaligned views.

**2026-09-16 — THE CLAMP IS BROKEN: 0.581 → 0.846 FROM ONE VARIABLE.** Submission
`56278746` replaced the node budget with a detection threshold (`det_threshold 0.5`),
changing nothing else against `56233167`. **+0.265.** Rank 3133 → 2625 of 3608.

**What the clamp was.** Four submissions scored 0.588 / 0.578 / 0.581 / 0.575 across
four structurally different pipelines — DoG at two separations, a 2,400-iteration
U-Net, and a 400-epoch U-Net trained on all 199 samples — whose CV node recall spanned
**0.65 to 0.98**. A leaderboard that will not move across that range is not measuring
the detector. The one thing all four shared was `predict_n_true(q0.9)`: every one took
its node count from the same log-linear fit, so every one emitted the same wrong count
on the hidden embryo no matter what sat underneath it. Detection was **purely
budget-limited** — emitted vs asked on the four visible samples was 13900/13851,
50600/50641, 7100/7105, 35200/35168, *exactly* the budget every time.

**Why the fit was the problem.** It is a compromise between two embryos with opposite
biases — `44b6` median ratio 1.028 at **p10 0.464**, `6bba` 1.002 at **p10 0.617** — so
even in-domain a tenth of samples get half the nodes they need. The hidden set is a
third embryo and nothing there can check it.

**The detector's own confidence is a better estimator of cell count than the fit.**
Emitted against the *true* counts once the threshold replaced the budget:

| sample | threshold | budget | true `N` |
|---|---|---|---|
| `44b6_0113de3b` | **27,192 (1.06×)** | 13,851 (0.54×) | 25,755 |
| `44b6_0b24845f` | 41,969 (1.28×) | 50,641 (1.54×) | 32,795 |
| `6bba_05b6850b` | 7,837 (1.23×) | 7,105 (1.12×) | 6,362 |
| `6bba_05db0fb1` | **74,994 (1.07×)** | 35,168 (0.50×) | 69,800 |

**So metric fact 4 needs rewriting.** This file has treated `N_true` as a knob to
predict since the beginning — "the per-sample predicted target is worth 0.993 vs 0.936
weighted multiplier" — and built `predict_n_true`, the density switch and every budget
sweep on top of it. The multiplier arithmetic was right and the conclusion was wrong:
a threshold gets the node count right *for free*, and the entire `N_true` apparatus was
a workaround for a saturated detector that stopped being saturated. **Do not reintroduce
a top-N budget.** Full-fold CV confirms the ordering: budget 0.7410, threshold 0.5
0.7830, threshold 0.9 0.7889.

**Where we now stand, and what is left.** 0.846 is rank ~2625/3608, top 72.8 %. Field
median is 0.902 (rank 1886), top-25 % is 0.944, top-10 % is 0.947. The remaining known,
measured items are unchanged by this and still add up:

1. **The learned edge head — −0.073 measured.** Still deleted. `softmax` over sources
   gives an abstain option the Hungarian linker structurally cannot have.
2. **8-way flip/rot TTA** — every public notebook above 0.942 uses it; runtime is 0.88 h
   against a 12 h cap, so it is free.
3. **Gap closing by interpolated node insertion, min-track-length 6, line-fit
   smoothing** — the public stack's post-processing, none of it implemented.
4. **ILP** (`tracksdata` ships as a wheel; `pyscipopt`+`ilpy` confirmed bundleable) —
   publicly +0.04.
5. A real training run. `E6` is still only 5 % of the reference's 137.8 GPU-hours.

**2026-09-14 — ROOT CAUSE FOUND, AND IT IS OUR OWN REWRITE. `kaggle/submit.py`
reimplements an inference path that already exists faithfully in this repo, and the
rewrite loses 0.129.** All arms below use the **same public weights**, the **same 8
videos**, the **same scorer**. Levels are inflated (those videos are in the checkpoint's
199-dataset training set) — read the deltas.

| arm | score | node recall | nodes/frame |
|---|---|---|---|
| **reference full pipeline** (threshold detector + learned edge head) | **0.8607** | 0.9960 | ~440 |
| reference detections + **our** Hungarian linker | 0.7880 | 0.9940 | 487 |
| **our pipeline as the notebook ships it** | **0.7322** | 0.9206 | 306 |
| ours, separation 4.0 | 0.8098 | 0.9716 | 317 |
| ours, separation 4.0 + oracle budget | 0.8206 | 0.9781 | 403 |

**Two swaps account for it, and both rest on a claim that was true of our old checkpoint
and false of the architecture.**

**(1) Deleting the learned edge head: −0.073.** `kaggle/submit.py:17` and
`detectors.py:8` both assert the transformer "currently emits zero edges". With the
public weights it emits **25k–68k edges per video** and reaches **J = 1.000** on
`44b6_0113de3b` where our Hungarian linker gets 0.636. The zero-edge observation came
from our own 12-epoch checkpoint and was generalised to the architecture; it is what
removed the strongest component of the reference pipeline. The mechanism it provides is
an **abstain option**: `softmax` over *sources* means at most one source per target can
exceed 0.5, and when the model is unsure none do, so the target gets no parent and no
false-positive edge. `linear_sum_assignment` cannot abstain — every detection inside the
gate gets linked, and every wrong link lands in the denominator.

**(2) Top-N budget + 7-voxel suppression instead of threshold + 3-voxel: −0.056.**
- **Suppression radius dominates.** `_peak_window(5.86, 1.625)` = **7**, a 4.875 µm
  exclusion radius against the reference's 1.625 µm. `separation_um` 5.86 → 4.0 alone is
  worth **+0.078** and takes node recall 0.914 → 0.965. `DEFAULT_SEPARATION_UM` was
  `2*sigma` = 5.86 while **every tuned CV number in this repo used 4.0**, and
  `kaggle/notebook.py` passes only `budget` and `gate_um`, so both submissions ran at
  5.86. Now 4.0.
- **The predicted budget is miscalibrated per video and biased against dense ones.**
  `predict_n_true(q90)/estimated_number_of_nodes` over all 199: median 1.015 but **p10
  0.519**, and by `N_total` quartile **1.125 / 1.242 / 0.962 / 0.559** (spearman
  −0.498). In production `44b6_0113de3b` got 13,900 nodes against 25,755 true — **0.54×,
  capping node recall at 0.54**. The reference needs no budget at all: its threshold
  lands at 0.94–1.53× of true.
- **`positive_only=False` pads the budget with negative-logit voxels** —
  `detectors.py:86` — i.e. voxels the detector explicitly rejected. 8 % of emitted nodes
  overall, **21–22 % on 2 of 8 videos**. Pure node penalty plus match-stealing.

**The premise behind the budget design is false for these weights.** `detectors.py:10-15`
justifies a budget over a threshold because "the map is saturated — median sigmoid
1.0000 at GT centroids, ~6,000 voxels/frame above 0.99". With the public checkpoint only
**1.59 % of voxels have logit > 0**, and a 3×3×3 local max yields **236 peaks on a 64³
frame**. Saturation was a property of our undertrained model, not of the detector.

**Ruled out by measurement, so nobody re-investigates them:** normalisation is
byte-identical across our inference, the reference inference and the reference training
(same quantiles, same epsilon, same lower clamp, no upper clamp anywhere); the
fake-pair trick costs **nothing** (node recall 0.9967 real pair vs 0.9973 fake — the
temporal attention degenerates cleanly on two identical tokens); coordinates are
correct with no factor of 4 and no axis swap, verified by a scale-axis probe; sliding
windows are equivalent at W=2.

**And the fix is already in the repo.** `scripts/predict_unet_transformer.py` differs
from the reference only in added docstrings — `_detect_cells_pooled`,
`pool_kernel_from_um`, `build_graph` and `_load_frame` are **byte-identical**. We do not
submit with it solely because it imports tracksdata at module scope, and the public
pack's `wheels/` proves tracksdata bundles fine. **Stop maintaining the rewrite; ship
the path that already works.**

**2026-09-14 — EVERY RECENT EXPERIMENT WAS UNDERPOWERED, effective n = 3 to 5.**
`--limit 8/10` takes the head of a sorted split, and `summarise` weights by
`w_i = TP+FP+FN ≈ GT edge count`, which ranges 49–1879 per sample. So:

| sample set | effective n | top sample's share of weight |
|---|---|---|
| `split1[:8]` | **3.16** | **52.8 %** |
| `split1[:10]` | 4.78 | 39.5 % |
| `split0[:30]` | 25.5 | — |
| `split0` full | 111.3 | — |

With per-sample `adjJ` spread 0.356–0.733, the standard error at effective n≈3 is
**±0.08–0.10**. The E2 2×2 loss arms, "the loss-shape fix did not work", the budget
sweeps "flat to four decimals", DoG-vs-U-Net and RANDOM_init were all inside their own
noise. **Decisive comparisons run on a full split or not at all.**

**The CV/LB agreement was a three-way confound.** "CV 0.5823 vs LB 0.588" compared the
`44b6` family at effective n=3.16 with a **DoG** detector through the submission path
against the `6bba` family at n=111 with a **U-Net** through the pipeline path. No
sample, no detector and no code path in common. The agreement was luck, and it was the
evidence for "the harness now predicts the leaderboard". It does not, yet.

**Also verified: our `metrics.py` has diverged from the official scorer** — 86 added
lines (dt≠1 filter, merge collapse, out-degree cap) plus a near-rewritten
`division_metrics.py` whose FP rule charges cases the reference does not. Proven
**numerically inert on our actual submission** (bit-identical on all four datasets), but
it spawned a false belief worth correcting: **the official scorer does NOT delete
`dt != 1` edges**, so `linking.py:225-227`'s instruction to express gap closing only by
inserting an interpolated node is wrong. A long edge is one cheap FP; an interpolated
node costs the node-count penalty *and* can steal a GT match. Restore both files to the
reference for hygiene.

**And `node_recall` as we report it is macro-averaged and ~0.09 optimistic** (0.7349
macro vs 0.6498 micro on the shipped submission). It is also not an upper bound on
anything: adding isolated nodes at exact GT coordinates gives **node recall 1.000 with
edge Jaccard 0.000**, because the extra nodes win the bipartite match and orphan the
linked detections. Stop using it as a proxy for achievable edge Jaccard.

**2026-09-14 — SECOND SUBMISSION SCORED 0.578, DOWN FROM 0.588, and the experiment was
designed badly.** Submission `56230904` bundled **four** changes: model swap
(`modelA_corrected`, trained on `44b6` → `E4_lr1e4`, trained on `6bba`), density switch
retired, division gate on (478 forks), isolated nodes pruned. Our CV said each was an
improvement; the leaderboard says the package is **−0.010**. With four variables and one
number, **nothing here is attributable** — a direct violation of this file's own
experiment discipline, on the one measurement where slots are limited to five a day.

What can be said. Divisions (+0.005 on CV) and pruning are small and almost certainly
not negative, so the loss is most likely the **model swap or the retired switch**. Both
point the same way: `modelA` was trained on **`44b6`**, the crowded 327-cells/frame
embryo, and `E4_lr1e4` on `6bba` at 97 — so if the hidden third embryo is crowded,
`modelA` transfers better and our fold-1 CV cannot see that. The retired switch also put
DoG on sparse samples, which this pack's own numbers say is not worthless.

**The standing lesson: one variable per submission.** Our CV now tracks the LB in
*level* (0.5823 CV against 0.588 LB) but has twice failed to rank two configurations
correctly. Until it ranks, the LB is the only arbiter of a config change, and it is
rationed.

**2026-09-14 — ONE FULL PASS BEATS EVERY SHORT RUN, first confirmation of the budget
diagnosis.** `E6_long_fullpass` (fold 1, batch 8, lr 1e-4, `det_neg_weight 1e-2`, no
`--max-iters`, so an epoch is a real pass of ~1,568 iterations):

| run | iterations in epoch 0 | det loss | CV |
|---|---|---|---|
| every E1–E5 arm | 200 | 0.0685–0.0704 | 0.6357–0.6663 |
| **E6, one full pass** | **~1,568** | **0.0150** | **0.6775** |

A single honest epoch beats the best of nine tuned short runs, at 4.6× the detection
loss reduction. 44 min/epoch, so 24 epochs is ~17 h. This is the first result in the
project that improves by *doing the same thing properly* rather than by changing
something.

**2026-09-14 — WHAT THE FIELD ACTUALLY RUNS, read out of their own artifact rather than
inferred.** The support pack (`pilkwang/biohub-temporal-unet3d-seed314159-v1`, 340 MB,
5,995 downloads) ships `repo/`, `weights/`, **`wheels/`** and a full
`history.csv`. Four things in it change decisions here.

**1. `tracksdata` CAN ship offline, so our foundational constraint was unnecessary.**
`wheels/tracksdata-0.1.0rc6.dev3+g980c2d30a-py3-none-any.whl` — they built a wheel from
the git repo and bundled it, alongside ~80 others (`ilpy`, `pyscipopt`, `rustworkx`,
`numba`, `llvmlite`, `geff`, `zarr 3.2.1`, `sqlalchemy`). This file has said since the
beginning "**tracksdata is not on PyPI, so keep the inference path free of
tracksdata**", and on 2026-09-08 four module-scope imports were removed to satisfy it.
Not on PyPI is true; unbundleable is not. **The cost of that belief was the ILP solver**,
which we therefore never enabled and which is publicly worth +0.04. The tracksdata-free
path is still a fine fallback — it runs with torch absent entirely — but it is a choice,
not a requirement, and `kaggle/README.md` should stop presenting it as one.

**2. Their checkpoint selection uses the metric this project proved worthless.**
`history.csv`'s `validation_score` is **exactly `validation_acc × validation_recall`**,
verified to six decimals on every epoch sampled. `validation_acc` sits at 0.9999 for all
400 epochs — saturated, carrying no information — so their selection is effectively
edge-matrix recall, and `best_epoch 381 / best_score 0.978` was chosen by it. We replaced
that number with the real competition scorer on 2026-09-07 (see Known defects). **So we
hold a real advantage in checkpoint selection over the pack the whole field forks** —
worth using once we have a run long enough for selection to matter.

**3. Their history REFUTES the anti-correlation, at real scale.** Over 400 epochs
detection loss falls monotonically and their validation metric rises monotonically:

| epoch | 1 | 21 | 51 | 101 | 151 | 251 | 351 | 381 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| det loss | 0.0198 | 0.0083 | 0.0058 | 0.0043 | 0.0036 | 0.0028 | 0.0025 | 0.0024 | 0.0025 |
| val score | 0.900 | 0.933 | 0.939 | 0.958 | 0.967 | 0.966 | 0.971 | **0.978** | 0.975 |

**They move together.** No anti-correlation, no best-epoch-0. And the arithmetic that
settles our own runs: all-train is ~18,933 windows, so at batch 8 **one of their epochs
is ~2,367 iterations**. Our entire 12 x 200 = **2,400-iteration run is their epoch 1** —
where their own curve already reads 0.900 and climbs for 380 more epochs. Every
detector experiment in this file compared noise inside their first epoch.

**4. The budget, and where it flattens.** `history.csv` records **137.8 GPU-hours** for
400 epochs (~0.34 h/epoch). The curve is steep early and flat late: 0.939 by epoch 51,
0.958 by 101, **0.967 by 151**, then only +0.011 over the next 230 epochs. So ~150
epochs ≈ **52 GPU-hours** buys most of it — two days on one A6000, which we have.

**Their inference configuration**, from `PredictConfig` in the pack (ours differs on
every line that matters):

| knob | theirs | ours |
|---|---|---|
| `det_threshold` | **0.5** | 0.99 |
| `det_tta` | **True** (flip-xy) | off |
| `pool_kernel_um` | **3.0** | 4.0 sep |
| `edge_activation` | softmax | n/a (geometric linker) |
| `use_ilp` | available, weights `edge -1.0 / appearance 0.1 / disappearance 0.1 / division 1.0` | never enabled |

On top of the pack their notebooks add 8-way TTA, a dual-seed logit blend, one- and
two-frame gap closing, a min-track-length filter of 6, line-fit smoothing and the
`DeepCenterUNet3D` centre prior (`pilkwang/biohub-deepcenter-unet3d-center-prior-v1`,
5,390 downloads) as a node-rescue veto.

**2026-09-14 — WHY THE FIELD IS AT 0.90 AND WE ARE AT 0.588. Answered, and neither half
is what this file has been chasing.** The public support pack everyone forks
(`pilkwang/biohub-temporal-unet3d-seed314159-v1`, **5,995 downloads**) ships its
training manifest. Compared to ours:

| | public pack | ours |
|---|---|---|
| `unet_layers` / `unet_out_channels` | `[32,64,128]` / 32 | **identical** |
| checkpoint bytes | 8,363,159 | 8,357,783 — **the same model** |
| learning rate | **1e-4** | 1e-4 (found independently, E4) |
| `det_neg_weight` | **0.01** | 0.1 — *we called 0.01 a defect* |
| `det_loss_weight` | 1.0 | 1.0 |
| `epochs_target` | **500** | 12 |
| `max_iters` | **None** — full passes | **200** |
| train datasets | **199 (all-train)** | 128 (one LOEO fold) |

**Same architecture, same learning rate, and the `det_neg_weight` this file lists under
"Known defects" is the value they actually train with.** The difference is the training
budget: 500 epochs of full passes is roughly **784,000 iterations against our 2,400 —
about 326x.**

**So the nine experiments in this file measured nothing.** Ignore radius, ordering
hinge, `det_loss_weight` 1e0/1e1/1e2, lr 1e-4/3e-4/1e-3, 12 vs 30 epochs — every one
ran at ~0.3 % of a real training budget, and the CV differences between epochs of a
2,400-iteration run are early-training noise. **The "loss/metric anti-correlation" this
project treated as its central mystery for a week is most likely that noise**, not a
loss pathology. E5 supports this: within-embryo (`6bba`→`6bba`) the CV is *flat* across
12 epochs, neither improving nor degrading — the signature of a run that has not started
converging. Do not cite the anti-correlation as a fact again until it survives a real
run.

**But the training budget is only the smaller half of the gap.** Scoring the public
500-epoch weights through *our* pipeline, fold 1, 10 videos:

| detector in our pipeline | score | edge J | node recall |
|---|---|---|---|
| our E4_lr1e4 (2,400 iters, held out) | 0.6348 | 0.6099 | 0.7632 |
| **public 500-epoch pack** | **0.6955** | 0.6786 | **0.9034** |
| public DoG notebook, their pipeline | ~0.73 | — | — |
| **field median, their pipeline** | **0.902** | — | — |

*Contamination caveat, and it matters*: the pack trained on **all 199 samples**, so it
has seen these videos. 0.6955 is an upper bound, not an honest CV, and the true
held-out advantage is smaller than +0.06. Only the leaderboard can settle it.

**The decomposition that follows is the important part.** Their detector reaches node
recall 0.90 and *our* pipeline converts that into 0.6955, while *their* pipeline
converts the same detector into ~0.94. So roughly:

- **detector training budget: worth ~0.06** (node recall 0.76 → 0.90)
- **everything after detection: worth ~0.24**

and we deleted or never enabled all of it — the `SimpleNodeTransformer` edge head, the
**ILP solver** (publicly measured at +0.04), 8-way flip/rot TTA, **one- and two-frame gap
closing by interpolated node insertion**, a min-track-length filter of 6, and line-fit
trajectory smoothing.

**This reverses the priority this file has held since 2026-09-11.** "Detection owns 86 %
of the loss" was measured correctly — 86.3 % of edge FN is a missing endpoint — but the
conclusion drawn from it was wrong. **Gap closing attacks missing endpoints directly**,
by inserting an interpolated node where a track breaks, without needing a better
detector at all. That is the single largest unexploited lever in the project and this
file has had it filed under "Not implemented at all" since the beginning.

**Revised order:**

1. **Post-detection stack first** — gap closing, min-track-length, ILP, TTA. Worth
   ~0.24 by the arithmetic above, needs no training, and `pyscipopt`+`ilpy` are
   confirmed bundleable (cp312 manylinux, 15 MB) so the ILP can ship offline even though
   tracksdata cannot.
2. **Use the public pretrained weights** — explicitly legal (pre-trained models are
   allowed) and worth ~0.06 immediately — while a real 500-epoch run trains.
3. **A real training run.** `E6_long_fullpass` (24 epochs, full passes, their recipe) is
   running; even that is only 5 % of their budget.
4. Everything this file previously called a detector priority is on hold until (1)–(3)
   land. Knob sweeps at 2,400 iterations are not experiments.

**2026-09-12 — The division term is NOT worth 0.1 to us, and a learned fork ranker is
premature.** The plan treated divisions as the largest unclaimed lever, on the
reasoning that opening the gate finds more real forks but drowns them in false ones,
so the fix is a better *ranker*. Measured on 14 fold-0 samples that contain divisions,
taking every fork the linker's second pass proposes and labelling it against GT:

| | |
|---|---|
| real forks proposed | **3** |
| false forks proposed | **1,667** |
| ratio | **1 : 556** |

| feature | real p25/p50/p75 | false p25/p50/p75 | separable |
|---|---|---|---|
| sister separation µm | 7.27 / 8.01 / 8.66 | 6.49 / 7.46 / 8.62 | **overlaps** |
| symmetry | 0.38 / 0.39 / 0.55 | 0.44 / 0.68 / 0.86 | **overlaps** |
| far-daughter µm | 5.19 / 5.56 / 5.67 | 4.90 / 5.31 / 5.79 | **overlaps** |
| near-daughter µm | 2.00 / 2.27 / 3.10 | 2.27 / 3.52 / 4.43 | **overlaps** |

Two conclusions, and the second matters more. A ranker over these geometric features
cannot work — every distribution overlaps at the quartiles, and the class balance is
1:556. And **even a perfect ranker is capped by what is proposed**: three real
divisions across fourteen samples means the division Jaccard ceiling on our current
detections is around **0.015 of score, not 0.100**. The daughters mostly are not
detected, so the fork is never a candidate.

So divisions are not an independent lever. They are downstream of detection like
everything else, and they should be revisited only after node recall moves. Caveat
worth stating: n=3 real forks is a small sample for the feature comparison, and only
geometry was tested — appearance features (intensity drop, nuclear shape) are
untested. But the 1:556 base rate and the proposal ceiling do not depend on that.

**2026-09-12 — Fold choice was distorting every recent comparison, and the numbers are
far apart.** Detector debugging moved to fold 1 because it trains on the label-rich
embryo (median 8 annotated nodes per frame against fold 0's 2). That is right for
*training* and wrong for *scoring*: fold 1 tests on `44b6`, which is the crowded,
0.77 %-annotated embryo carrying only **15 % of the metric weight**. Through the
submission path:

| fold | tests on | share of metric weight | DoG score | node recall |
|---|---|---|---|---|
| 1 | `44b6` (71 samples) | 15 % | 0.5348 | 0.6024 |
| 0 | `6bba` (128 samples) | **85 %** | **0.6856** | 0.7850 |

Weighted, the shipped classical config is worth roughly **0.663**, not the 0.53 the
fold-1 runs suggested. **Train on fold 1, report on both, and never quote a fold-1
number as the result.**

**The organizers' starter detector does not transfer, measured not assumed.** Their
60-line notebook thresholds a smoothed volume at the 90th percentile and takes
connected-component centroids rather than top-N DoG maxima. On 10 fold-1 videos it
scores **0.1440** at pct 90 and **0.2952** at its best setting (pct 99, gate 8),
against our DoG's 0.5348 — because on a crowded embryo a percentile threshold merges
adjacent nuclei into single blobs, yielding 54–77 nodes/frame where ~260–330 are
needed, and node recall caps at 0.38. Whatever reaches ~0.73 publicly, it is not this.

**2026-09-12 — The learned detector IS real, and everything it learns happens in the
first 200 iterations.** Seven runs now peak at epoch 0, which raised the obvious
question nobody had asked: does training do anything at all? Scored through the
**submission path** on the same 10 fold-1 videos:

| detector | score | edge J | node recall |
|---|---|---|---|
| randomly initialised U-Net | 0.3523 | 0.3381 | 0.5015 |
| **trained U-Net (epoch 0)** | **0.6077** | 0.5864 | **0.7439** |
| classical DoG | 0.5348 | 0.5100 | 0.6024 |

**Training is worth +0.255 over random init, and the U-Net beats DoG by +0.073 on
matched samples.** So the learned path is not an illusion and should not be abandoned
— but 200 iterations of batch 16 captures all of it, and every epoch after makes it
worse. Note also that DoG at 0.5348 is well below the ~0.73 a public DoG notebook
reaches, so our classical arm is weak too.

**Two explanations tested and eliminated.** Closing the loss's null space (ignore
radius, ordering hinge) made it worse, monotonically. Rebalancing the trunk's
objectives did essentially nothing: `det_loss_weight` 1e0 / 1e1 / 1e2 gave best CV
0.6541 / 0.6591 / 0.6662, **all at epoch 0**. So the detection head being a
33-parameter probe on a trunk shared with the edge task is not the binding problem
either.

**What that leaves, and it is embarrassing in a useful way:** `train()` builds
`AdamW(model.parameters(), lr=lr)` with **no scheduler, no warmup and no weight decay
tuning** (`train_unet_transformer.py:1391`), and runs a constant 1e-3 for the whole
run. A model that reaches a good region in 200 steps and then degrades under a
constant, fairly high LR is the textbook symptom. That is now the live hypothesis and
it is being tested at 1e-4 and 3e-4.

**2026-09-12 — E1: the loss/metric divergence is REAL, not a bent ruler.** This was the
project's leading excuse for not fixing the loss, and it is now refuted. The
`reports/diagnosis_loss_metric.md` H1 hypothesis said the "det loss falls 4.4x while CV
falls" observation might be an artefact of scoring at gate 12 µm with the division pass
on — the one linker configuration where a *better* detector scores *worse*. Re-run on
**fold 1** (the label-rich embryo, 4x the positives), `det_neg_weight 1e-1`, selection
at the tuned gate with divisions on:

| epoch | 0 | 1 | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|---|
| det loss | 0.0685 | 0.0451 | 0.0371 | 0.0338 | 0.0328 | 0.0310 | **0.0295** |
| CV | **0.6541** | 0.6465 | 0.6452 | 0.6269 | 0.6427 | 0.6087 | 0.6360 |

**Detection loss falls 2.3x, CV is best at epoch 0, and the ruler is not the
explanation.** Three folds, three configurations, same shape. Training this loss longer
makes the detector worse, and no amount of gate tuning changes that.

**A methodology error worth recording, because it cost a run.** The paired arm changed
*two* variables at once — ignore radius 7.0→8.45 **and** ordering hinge 0→1.0 — so it
could attribute nothing. It also revealed a scale problem: the hinge contributed ~1.10
against a BCE of ~0.03, a **40x imbalance**, so that arm was almost purely the hinge.
Relaunched as a proper 2x2 at `ordering_weight 0.05`. One variable at a time, and check
the *magnitudes* of a summed loss's terms before trusting the sum.

Worth watching in the rerun: the hinge arm's CV was **rising** (0.475 → 0.562 over six
epochs) while the control's fell. Lower absolute level, opposite direction, and it had
not converged when another tenant's process filled the card and OOM'd it. That is the
first training dynamic in this project that improves with epochs.

**2026-09-12 — Phase 0 landed, and three measurements redirect the remaining work.**

**The linker is not the problem, and now we know by how much.** For every GT edge,
did both endpoints match a prediction, and did we then predict the edge?
**P(correct link | both matched) = 0.9578** over 10 fold-0 samples — above the
[0.816, 0.916] range aggregate arithmetic allowed. The edge-FN split is
**86.3 % missing endpoint / 13.7 % missing link**. Corroborated independently: feeding
*ground-truth* node coordinates to the unchanged linker scores edge Jaccard **0.9847**
and a total of **1.0738**. So ILP is worth ~+0.03 (recovering part of that 13.7 %),
and **detection owns roughly six times more**. Stop proposing learned linkers.

**The detection loss has an exact null space containing what the metric reads.** Not
"the map is saturated" — that is a sigmoid artifact; in *logit* space the map spans
−8.4 to +42 and ranks fine, which is why both scorers peak logits. The real mechanism:
`_top_peaks` keeps a voxel iff it wins a ±3-voxel cube whose **corner is 8.44 µm**
away, while `ignore_mask` zeroes every bright voxel beyond **7.0 µm** of an annotated
centroid. 7.0 < 8.44, so **73–79 % of annotated centroids have a zero-gradient voxel
inside their own deciding neighbourhood**; adding ±5 to logits only in the ignored
region changes the loss by a **bit-identical zero** while moving node recall
1.000 → 0.250. `(1-target)**4` silences the rest of the cube, and there is **no
ordering term anywhere**. Consequence: the map possesses only **332–517 maxima per
frame** against budgets up to 698. `reports/diagnosis_loss_metric.md` established this
on 2026-09-07 and was committed and then ignored for four days while the project tuned
scalars. **Read it before touching the detector.**

**What the 0.94 cluster runs is our own architecture.** The dominant public stack is
the organizers' baseline we forked — `TemporalUNet3D` → `SimpleNodeTransformer` →
**`tracksdata` ILP solver** — and we kept the U-Net, deleted the transformer linker and
never enabled the ILP. Public measurements: greedy → ILP is **+0.04**; every notebook
above 0.942 uses 8–9-way flip/rot TTA; the top forks blend two independently-seeded
detectors at α≈0.5; then gap closing by interpolated node insertion, a min-track-length
filter of 6, and line-fit trajectory smoothing.

**The "metric hack" is dead and was never the node count.** Notebooks titled "metric
hack" appended a synthetic hub node at `t=−1000, (−10000,−10000,−10000)` merging every
track into one weakly-connected component plus fake forks, exploiting the old rule that
a division TP needed only shared-component reachability. Patched 2026-07-17
(`aa65e90`); `metrics.md` now says "Merely sharing a weakly connected component is not
sufficient." **Our `division_metrics.py` contains the post-patch
`_is_strongly_connected_division`, so our CV measures the current metric.**
**CORRECTED 2026-09-20: the 0.946/0.947 cluster is NOT a defanged fork of those hack
notebooks.** It is `reyhanksatria/biohub-cell-tracking-0-947-lb` with a clean documented
lineage (pilkwang's learned-graph baseline → `nusrati/0-938` → `evgendvorkin/0-927`,
bidirectional fusion → Reyhan Ksatria for 0.941→0.947); the runnable public copy is
`zhincez/biohub-0-947-lb-runnable-with-public-datasets`. No hub nodes anywhere in it.
The hack was patched 2026-07-17 (`aa65e90`) and **every submission was rescored by
2026-07-23** (Kaggle admin: "The rescore is complete"), after which the augmentation is
*negative* (measured 0.9839 → 0.9794). The 0.963–0.966 figures still visible on some
notebook badges are fossils of the pre-patch metric — the badges were never refreshed,
the leaderboard was. Also
note: the widely-forked *offline* metric still uses the pre-patch rule and reports
division Jaccard **double** — score locally with ours, never theirs.

**There is no node-count exploit.** `m = 1.1 − 0.1·N_pred/N_true` caps at +10 %, and J
collapses faster than m rises: at 0.4× budget you gain 6 % of multiplier and lose 35 %
of J. Measured optimum is 0.9–1.0×, where we already sit.

Fixed this session: divisions re-enabled at the swept 6.0 µm gate; `predict_n_true`
moved to `detectors.py` so CV and submission cannot diverge again, with
`score_video(predicted_budget=True)` now the default (**−0.0112** against the oracle on
24 samples — that difference was invisible before and is part of the CV/LB gap);
isolated nodes pruned from the submission (3.1–6.9 % of what we shipped). 291 tests.

**2026-09-11 — FIRST LEADERBOARD SCORE: 0.588, AND IT INVALIDATES THE PROJECT'S WHOLE
SENSE OF SCALE.** Submission `56173960` (density switch, model A, gate 8 µm) scored
**0.588 public** against a CV of 0.7366. The CV/LB gap is real and worth chasing — but
it is the *second* most important thing in this result. The first is the field:

| | score |
|---|---|
| leaderboard max | **0.970** |
| 7th place (prize cutoff) | **0.958** |
| **median of 3,402 teams** | **0.902** |
| teams ≥ 0.90 | **1,812** |
| teams ≥ 0.70 | **2,797** |
| **us** | **0.588 — rank ~2932 / 3402** |

**Half the field is above 0.90 and we are in the bottom 14 %.** Every internal
comparison in this file was made against our own previous numbers, so "the biggest
score gain in the project, +0.058" was measured on a scale where the gap to a *median*
competitor is **+0.31**. Tuning four scalars bought 0.068; we need roughly five times
that just to reach the middle of the board. **Stop treating knob sweeps as the main
line of work.**

**What the field's scores imply about the method.** A score of 0.90 is essentially a
weighted adjusted edge Jaccard of 0.90, which requires near-complete detection — our
node recall is 0.86–0.89 and our edge Jaccard 0.69. Linking is not the gap (geometry
already solves it, and the margin analysis holds). **The detector is the entire gap**,
and this file already documents it as broken on four independent counts: the map is
saturated (median sigmoid 1.0000 at GT centroids), the detection loss is
*anti-correlated* with the score, the best epoch is 0, and `det_neg_weight` runs 10×
weaker than designed. We knew the detector was broken and spent the effort tuning
around it. The leaderboard says that was the wrong call.

**Revised priorities, replacing step 5's ordering:**

1. **Find out what a 0.90 actually takes before building anything.** The public
   notebooks and discussion are visible now and are the cheapest information in the
   project. 1,812 teams are doing something we are not.
2. **Score the unmodified upstream baseline.** We forked `royerlab/kaggle-cell-tracking-competition`
   and replaced its learned linker with a geometric one after finding it emitted zero
   edges. If the baseline as shipped scores ~0.9, we broke something rather than
   improved it, and that is a one-run question.
3. **Fix detection properly, not around the edges** — `det_neg_weight`,
   `bright_threshold`, and the loss/metric divergence, in that order. Or replace it:
   **Spotiflow trains natively from point labels and is explicitly legal here.** This
   file listed it as a fallback; at 0.588 it is a primary candidate.
4. The node-budget and switch work stands, but it is worth ~0.06 on a 0.38 gap.

**Do not read the CV/LB gap as the main problem.** Even a perfect CV→LB transfer would
have put us at 0.7366, which is still rank ~2800. The harness is not what is wrong.

**2026-09-11 — THE CHAIN IS CLOSED: first real submission made, from the server, by
CLI.** `kaggle competitions submit -k <notebook> -v <version>`; no browser and no MCP
server needed. Submission is by **notebook**, and `-f submission.csv` names *the
kernel's output file*, not a file to upload — the local CSV is never sent, and Kaggle
reruns the notebook against the hidden test set. Full procedure in `kaggle/README.md`
→ *Submitting*.

**It took four kernel runs to get there, and every failure was an environment
assumption this file had recorded as fact.** Worth reading as a group, because the
shape repeats: each was invisible locally, each produced a plausible-looking error
somewhere other than its cause, and a one-minute throwaway probe kernel would have
caught all four at once.

| run | died on | actual cause |
|---|---|---|
| 1 | `StopIteration` | competition mounts at `competitions/<slug>/test`, not `<slug>/test` |
| 2 | missing `requirements-offline.txt` | pushed while the dataset **version was still processing** — `datasets status` says `ready` about the *dataset*, not the pending version |
| 3 | same error again | same race; the fix is to diff `datasets files` sizes against local, not to poll status |
| 4 | — | **completed, validated, submitted** |

Latent behind run 1, and never reached: the wheels were built `--python-version 3.11`
against an image that is **3.12.13**, so five of twelve carried `cp311` tags and
`pip --no-index` would have found no candidate one line later.

**The runtime worry was unfounded, by 24x.** Measured on the real T4 kernel: DoG 3.8–4.0
s/video, U-Net 13.3 s/video, **8.9 s/video mixed under the density switch → 0.49 h for a
train-sized 199-video test set**, against the 12 h cap. The 16.5 h figure was a CPU
extrapolation off this box and does not survive contact with a GPU notebook. So the cap
is *not* a design constraint: flip TTA, a larger model, or running both detectors on
every sample are all affordable now, and that reopens options step 5 had closed.

**A new and genuinely awkward finding: node count is not reproducible across GPUs.**
Same checkpoint, same config, same input — `44b6_0b24845f` gave **50,561** nodes on this
box's A6000 and **34,837** on Kaggle's T4, 31 % short of a budget it filled locally. Not
precision (`load_detector` is fp32 on both). It is the saturated map again:
`_top_peaks` marks `field >= pooled`, so on a plateau every voxel tied with the window
maximum counts as a peak, and *how many compare exactly equal* depends on the cuDNN
algorithm, which differs by card. Three consequences, and the third is the one that
costs time if forgotten: the node-budget calibration in metric fact 4 was measured on
hardware the submission does not run on; it is a second independent argument for fixing
saturation (`det_neg_weight`, `bright_threshold`) rather than tuning around it; and
**any CV-vs-LB gap on this submission is partly this**, so rule it out before hunting a
scoring bug.

## Plan

**Rewritten 2026-09-21 (end of day). Steps 1–7 of the old plan are done or superseded;
the historical reasoning is preserved in Status, which is where the measurements live.**
Seven days to the 2026-09-29 deadline. **Public LB 0.925**; best measured weighted LOEO
CV **0.9466** (soup4 + souped pack50 secondary + edge head 0.5 + bidir 0.25). The
division gates gained on fold 0 and lost on fold 1, netting −0.0002 weighted, so they are
not part of it.

### The three things that actually matter now

**1. Get the measured configurations onto the board, in order.** Two are ready and the
ordering is a hard constraint, not a preference — a graded rerun executes at *submission*
time against the **current** dataset versions, and `kernel-metadata.json` lists all three
datasets unpinned:

- **v14** — soup2, weighted CV 0.9407. **SUBMITTED 2026-09-22 as `56460859`** once the
  competition slug was corrected to `biohub-cell-tracking-during-development` (the 403
  was never a rules problem — see Status). Validator PASS, 97.4 s/video → 5.38 h.
  **Until it has scored, nothing may be uploaded to volt-weights or volt-code**, or v14
  stops measuring what it was built to measure.
- **v15** — soup4 + edge head 0.5 + bidir 0.25, weighted CV **0.9449**, both folds. The
  weights are already persisted and md5-verified at `weights/soup_members/`.

**2. Select both final submissions by hand, on CV rather than on the public LB.** The
public board is 29 % of the test set and the 0.947 pile is 700 teams fitted to that 29 %.
Our instrument is a 199-sample held-out-by-embryo estimate; theirs is four movies at
effective n ≈ 2. Pick the CV-best pipeline for one slot, and for the other a
configuration whose **failure mode is uncorrelated** — realistically the 0.925 stack,
which differs in the whole detector, so one systematic problem cannot take both down.

**3. Identity verification — human action, prize eligibility depends on it**
(`requiresIdentityVerification: true`). Still open.

### What is measured out, so it does not get re-proposed

- **Divisions by gating: every *threshold* is exhausted, but the line is REOPENED** —
  the 2026-09-22 pool audit retracts "closed by elimination" (see Status). The caps do
  **not** bind (median 1 candidate against a cap of 2; 12.8 % of frames over), and every
  true fork already in the pool ranks first and is accepted (6/6). So a better *ranker*
  has nothing to do — the pool is missing its candidates, because the mutual-nearest rule
  proposes only the **single** nearest orphan and a spurious detection nearer than the
  true sister is unreachable by any gate width. **That sub-hypothesis is itself now
  falsified** — the rejection log says every true daughter that clears the gates is
  already the *nearest* orphan (max `k_rank` 0), so `--div-n-cand` reaches nothing. The
  gates that bind are **`d_par`** (7 of 18 refused, needing up to 11.8 µm against the
  shipped 9.0) and **`symmetry`** (5 of 18, needing up to 1.07 against 0.6); the
  2026-09-18 widening arm moved parent and *sister* on fold 1 and left symmetry alone,
  and sister was never the blocker. `--div-parent-um 12 --div-symmetry-tau 0.9` is the
  targeted arm, running on fold 0. Over-proposing is cheap on this metric — only 40 of
  4,053 emitted forks are charged as FP — but the gates sit at a 1:664 knee, so this is
  a genuine trade rather than a free win, and 19 of 25 GT divisions missing the pool is
  the headroom it is competing for.
- **det_threshold**: exhausted in both directions (0.95 / 0.965 / 0.98 all within noise).
- **The port inventory**: every remaining component is worth ≤ +0.0015 here against the
  reference's published ≥ +0.005, because their deltas repair a defect we do not have.
  Do not port another on the strength of its published delta.
- ~~Sub-0.005 CV arms are invisible on the board.~~ **RETRACTED 2026-09-22** — it rested
  on a waiter bug that reported one submission's score for another. The board resolved a
  0.0024 CV difference into a **0.007** LB difference (0.918 vs 0.925), and our CV
  **ranked** the two correctly. Small arms are worth shipping; the reason to bundle is
  that deltas compress on transfer, not that the board cannot see them.
- **Trackastra / Ultrack**: need instance masks we do not have.

### The levers that remain, best first

- ~~**The division gates, which the rejection log found and which PAY.**~~ **Measured out
  on fold 1: +0.0006 on fold 0 but −0.0044 on fold 1, weighted −0.0002.** A wash, and a
  fold-0 delta a fifth the size of the inter-fold gap was never shippable by this file's
  own rule. The *method* survives and is the transferable part — a gate sweep guided by a
  rejection log that names which clause fires and at what value found a real fold-0 effect
  in one arm where four blind sweeps found none — but the arm itself does not ship.
- **Souping the secondary: +0.0017 weighted, and free.** Already banked into the v15
  candidate. Its fold asymmetry (+0.0006 on fold 0, **+0.0083 on fold 1**) is the shape to
  expect from any detector-side variance reduction, and the reason to keep trusting the
  fold-1 number on that axis specifically.
- **More soup members — member count has an INTERIOR optimum** (2 → 0.9428, 4 → 0.9440,
  8 → 0.9466 on fold 0), because averaging smooths the map and sisters at p5 5.41 µm merge.
  4 is the operating point. `scripts/soup_checkpoints.py` builds any subset, and
  `shehailrs/…350ep` adds a third *same-run* member for the secondary.
- ~~**A genuinely independent third detector** for the blend.~~ **CLOSED BY MEASUREMENT
  2026-09-22, and not for lack of material.** Three independent public runs were found and
  hashed (see the checkpoint inventory); two have now been measured in the secondary slot
  on full fold 0, and both lose badly:

  | secondary | fold 0 | Δ vs pack50 |
  |---|---|---|
  | **pack50, souped** | **0.9469** | **+0.0006** |
  | pack50, raw (shipped) | 0.9463 | — |
  | `aashishnegi23/cellmot-ft-detector` | 0.9388 | **−0.0075** |
  | `pilkwang/…support-pack-v1` | 0.9321 | **−0.0142** |

  **So the detection blend needs a PEER, not merely an independent model.** Its measured
  +0.0036 is not "any second net helps" — an independent-but-weaker checkpoint actively
  costs, by up to 0.014, and the loss shows up in adjJ, divJ and node recall together
  (packv1: 0.9238 / 0.0827 / 0.9747), i.e. it degrades the detection map rather than
  trading terms. `pubfull` and `pack50` are the only two peer-quality checkpoints in
  public circulation, and we already use both. Extending this axis would mean training a
  peer ourselves, which is ruled out independently: `E7_alltrain` at 100 epochs is 0.076
  behind through the identical stack, and their 400 epochs cost 137 GPU-hours.
  Two further constraints keep it narrow even if a peer appeared: the **12 h cap** allows
  only one secondary, since each member costs an extra encode and v14 already runs ~8.6 h
  graded.
- **The division pool's remaining 80 %** — 99 of 124 GT divisions never become candidates
  at all, and ~0.09 of score sits behind the term. But note what is now *closed* inside it
  (see Status): the ranking does not bind, `n_cand` is measured negative, and the edge head
  carries no division signal (AUC 0.502). What is left is detection- and linking-limited —
  7 of 25 true daughters were never even inside the sister gate — or needs a fork
  classifier trained for the task, which is a new model rather than a knob.
- **The decision rule for spending a slot.** One clean point per side: the detector-side
  blend transferred at ~81 % (+0.0062 CV → +0.005 LB) and the edge-head-side reference
  path at ~35 % (+0.0346 → +0.012). Weight arms by which side they touch, but treat each
  figure as a single measurement rather than a calibrated rate.
- **Always pass `-c biohub-cell-tracking-during-development`.** The slug
  `biohub-cell-tracking-competition` 403s on submit and 404s on reads, and cost ten hours
  of misdiagnosis.

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
