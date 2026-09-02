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
   else is discarded, not penalized. **Dense over-detection is nearly cost-free.**
3. `edge_FN = gt_num_edges - TP`. With (2): **recall dominates, global precision
   barely matters.**
4. **The node-count penalty is soft and non-monotonic:**
   `J * (1 - 0.1 * (N_pred - N_true)/N_true)`, clamped at 0, `N_true =
   estimated_number_of_nodes` from the GEFF metadata `extra`. Doubling node count
   costs ~10% relative; predicting *fewer* nodes multiplies J *above* 1.0 (hence
   scores > 1.0). Total node count is a scalar hyperparameter with an interior
   optimum — tune it last, it interacts with detection and linking both.
5. **7 µm is generous** — ~4 z-voxels, ~17 xy-voxels. Sub-voxel localization is
   worthless here.

Consequences:

- Tune the detector for **recall at a fixed node budget**, not F1: sweep (detection
  threshold, suppression radius) against CV score, never against detection P/R.
- Set the suppression radius near the **cell pitch derived from
  `estimated_number_of_nodes` (~15 µm)**, not near 7 µm and **not** off GT
  nearest-neighbour spacing — GT is 0.2–1.6% sampled, so its ~23 µm spacing is a
  sampling artifact (see Data). The one real cost of over-detection is a spurious point
  winning the assignment the true node should have taken.
- The node penalty is mild in both directions: predicting 2× `N_true` costs ~10%
  relative, and the bonus for under-predicting is capped at +10% (the ratio bottoms out
  at −1). Since a missed node costs two edges outright, **the asymmetry favours
  over-detection** — but the budget is worth a deliberate sweep, not a shrug.
- Never spend effort on centroid refinement, sub-voxel peak fitting, or segmentation
  masks. None of it is measured.
- A missed detection costs two edges, not one.
- Aggressive detector + *selective* linker beats conservative detector + permissive
  linker.

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
| Train | University Linux server, RTX 6000 rack | all real training |
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

**Linux server.** `git clone` (the venv is not portable), `uv sync`, export
`VOLT_DATA_DIR`, `uv run pytest -m "not slow"`. Establish before trusting it: which
RTX 6000 (Ada `sm_89` 48 GB vs Pro Blackwell `sm_120` 96 GB — if Blackwell, widen
the cu130 marker to Linux); whether compute nodes have outbound internet, since
`tracksdata` is a git dependency and `uv sync` resolves over the network; SLURM or
bare SSH (either way, training must checkpoint-and-resume so preemption is free);
whether the data filesystem handles many small files, since zarr is exactly that.
The Kaggle CLI is in the dev group, so `uv sync` provides it; auth is the new-style
`KGAT_…` token in `~/.kaggle/access_token` at mode 600 (or `$KAGGLE_API_TOKEN`) — **not**
the retired `kaggle.json` username/key pair. The competition rules must be accepted in
the browser or the API returns 403. Data is CC0, so university storage is fine.

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

**How sparse "sparse" actually is** (measured on two samples): `6bba_c328f2fd` has 511
nodes / 500 edges / **2** divisions against `estimated_number_of_nodes = 31228` — **1.6%**
of cells, ~5.3 annotated nodes per frame. `44b6_0113de3b` has 52 nodes / 50 edges / **0**
divisions against 25755 — **0.2%**, exactly **one annotated node per frame, i.e. a single
lineage for the whole video**. Annotation starts at t=4, not t=0. This is far sparser than
"a subset", and it means:

- Per-frame edge supervision is 1–5 rows of the transition matrix. The learned linker is
  extremely data-starved; the detector, which sees whole images, is the better lever.
- Per-sample metric weight `w_i = TP+FP+FN` therefore ranges from ~50 to ~500+, so the
  weighted score is dominated by the densely annotated videos.
- Divisions are a handful per video, so per-video division Jaccard is pure noise; only
  the micro-average over all videos means anything.

**Measured motion and spacing** — these are the numbers the hyperparameters come from:

- GT displacement t→t+1: **median 1.7–2.9 µm, p90 ~4.5 µm, max 7.8–13.4 µm.** The 7 µm
  matching tolerance is *larger than the median inter-frame motion*. A linking gate of
  ~5–10 µm covers essentially all real motion.
- **Do not read the suppression radius off GT nearest-neighbour spacing.** Measured, it is
  ~23 µm median, but that is an artifact of annotating 1.6% of cells, not the cell pitch.
  Derive the pitch from `estimated_number_of_nodes` instead: ~260–310 cells in a
  ~104 µm cube gives **~15 µm cell-to-cell spacing**. The current defaults (5 µm in
  training, 3 µm at inference) sit far below that, so the detector over-detects heavily —
  nearly free per metric fact 2, but it spends the node budget.

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

**The local `test/` folder is fake** — it holds copies of train samples. At notebook
rerun a hidden test set is swapped in, **approximately the size of the training set**.
So the 12 h budget must be measured against roughly train-sized throughput, and a local
`test/` score means nothing.

`tests/conftest.py` auto-builds `tests/data/division_clip` from frames 26–31 of
`6bba_c328f2fd` (real division at original t=28 → fixture t=2) with recomputed quantile
attrs. **The dev-loop asset**: 5 frames, known division, seconds on the 5070 or on CPU.

**Data placement.** The Windows box has ~100 GB free, so the full 88 GB does not belong
here — pull a handful of samples (including `6bba_c328f2fd` for the fixture) locally and
put the complete set on the Linux server. Needs `~/.kaggle/access_token` (see the Linux
section for the token format).

## Codebase map

| Path | Lines | Role | Expectation |
|---|---|---|---|
| `src/volt_biohub/metrics.py` | 536 | official scorer | semantics frozen |
| `src/volt_biohub/division_metrics.py` | 574 | division TP/FP/FN | semantics frozen |
| `src/volt_biohub/io.py` | 396 | zarr/geff I/O, resample, normalize | heavy rework |
| `src/volt_biohub/img_proc.py` | 187 | normalization, coord rescaling, `nms_3d` | keep, extend |
| `src/volt_biohub/models/temporal_unet.py` | 141 | `TemporalUNet3D` | main detector |
| `src/volt_biohub/models/simple_node_transformer.py` | 210 | pairwise edge scorer | late-stage option |
| `scripts/dataspec.py` | 40 | path resolution | keep |
| `scripts/fetch_data.py` | 320 | Kaggle subset/full downloader, manifest cache, resumable | ours; keep |
| `scripts/train_unet_transformer.py` | 1294 | joint training | rework or replace |
| `scripts/predict_unet_transformer.py` | 677 | inference; `--use-ilp`, `--det-threshold` (0.99) | rework |
| `scripts/geffs_to_csv.py` / `csv_to_geffs.py` | 85 / 89 | submission round-trip | wrap in a validator |
| `scripts/evaluate.py` | 142 | score pred vs GT geffs | keep, add per-video diagnostics |
| `scripts/augmentations.py` | 63 | brightness shift + axis flips only | extend |
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

- `compute_detection_loss` marks a **single voxel** per GT cell positive and penalizes
  every other voxel at `neg_weight`. With sparse GT, real unannotated cells are trained
  as negatives. **The biggest modelling weakness in the baseline** and the first thing
  worth fixing. It is also why the inference default is `--det-threshold 0.99`: the help
  text admits the detector is badly calibrated and leans on an extreme threshold.
- **Checkpoint selection does not use the competition metric.** `train()` keeps the best
  epoch by `test_acc × test_recall` (`train_unet_transformer.py:1180`), where accuracy is
  per-cell agreement on the edge matrix. The real scorer is in-repo and unused here.
- **The two knobs the metric says matter most are unreachable from the CLI.**
  `PredictConfig` holds `pool_kernel_um` (suppression radius), `threshold` (edge
  probability), `edge_activation` and `det_tta`, but `main()` forwards only
  `--det-threshold` and the ILP weights. Sweeping today means editing the dataclass.
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
- **No `--seed`.** `train()` takes one; no CLI argument reaches it. Runs are not
  reproducible today.
- `--method baseline` in the README/commands matches nothing — `DEFAULT_METHOD` is
  `unet_transformer`, so it looks for `weights/baseline/split_0/edge_predictor_best.pth`,
  which nothing writes.
- `dataset_splits.json` is silently auto-generated as a seed-0 90/10 split when absent,
  and it is a *video* split, not embryo-grouped. Write it deliberately.
- Both `train` and `predict` **raise** if a zarr lacks `image_statistics.quantiles`. The
  real data does carry it on both train and test (see Data), so this is latent rather
  than active — but it is a hard crash, not a fallback, if the hidden test set ever
  differs.
- `open_dataset(device="cuda")` is the **default** → crashes CPU-only. Needs a device
  resolver. Its eager `da.from_zarr(...).compute()` loads a whole video, which at ~839 MB
  is survivable; `load_image=False` gives metadata only.
- Detections are stored as `int16` (`predict_unet_transformer.py:293,496`). Fine for
  256³-scale volumes, silently wrong if a future sample exceeds 32767 in any axis.

## Commands

```powershell
uv sync                                    # add --extra viz for napari
uv run pytest -m "not slow"
uv run ruff check .
uv run python scripts/train_unet_transformer.py --method baseline --split 0 --epochs 1
uv run python scripts/predict_unet_transformer.py --method baseline --split 0
uv run python scripts/evaluate.py --pred-dir <dir> --gt-dir $env:VOLT_DATA_DIR
uv run python scripts/geffs_to_csv.py --in-dir <dir> --csv submission.csv
uv run python visualize/visualize_predictions.py --method baseline --split 0
```

Same on Linux with `$VOLT_DATA_DIR` and forward slashes.

## Status

Renamed `tracking_cellmot` → `volt_biohub`, `CELLMOT_DATA_DIR` → `VOLT_DATA_DIR`.
Windows torch on the cu130 index, CUDA verified with `sm_120` present. Previously
undeclared runtime deps (numpy, polars, dask, geff) now declared; ruff in the dev
group. `.gitignore` / `.gitattributes` in place.

Kaggle auth works and `scripts/fetch_data.py` has cached the full 24,886-file manifest to
`C:\data\biohub\manifest.csv`. A 7-sample dev subset (`6bba_c328f2fd` for the fixture plus
3 per embryo) is on the dev box. **`pytest -m "not slow"` is fully green: 122 passed** —
the scorer is verified against real data, which closes the "do we trust the scorer"
question. The whole inherited pipeline has been read end to end (see "The pipeline as it
stands" and "Known defects").

Not done: full 88 GB set on the Linux server, `inspect_data.py` across all 199 samples,
embryo-grouped splits, any modelling, Kaggle team.

## Plan

**Gate 0 — done except the server and the team.** Dev subset downloaded, `pytest -m "not
slow"` green at 122 passed. Remaining: `fetch_data.py get --all` on the Linux server, and
**form the Kaggle team before Sept 22**.

1. **Measure before designing** — `scripts/inspect_data.py` over all 199 samples, to turn
   the two-sample spot checks in Data into distributions: GT nodes/edges/divisions vs
   `estimated_number_of_nodes` per sample, displacement percentiles, how many samples are
   single-lineage like `44b6_0113de3b`, and how sparsity correlates with embryo. The
   per-sample metric weight `w_i` is set by these counts, so knowing which samples
   dominate the score is knowing where to spend effort.
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
5. **Detection (week 2)** — fix the sparse-negative problem: Gaussian heatmap targets
   (σ ≈ nucleus radius from step 1) and mask the negative term on high-intensity
   unannotated voxels; or bootstrap classical peaks ∪ GT as pseudo-labels. The right
   framing is positive-unlabeled learning — unannotated cells are unlabeled, not
   negative. Switch checkpoint selection to the real scorer at the same time. Fallback
   if the UNet won't converge in budget: Spotiflow 3D, which trains natively from point
   labels and is legal here (pre-trained models are explicitly allowed).
6. **Linking (week 3)**, in priority order: (a) **global frame-to-frame registration
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
