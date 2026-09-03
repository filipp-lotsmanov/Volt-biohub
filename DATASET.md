# Dataset structure

Everything below is measured from the 203 datasets in this competition, not
inferred from the competition description. Reproduce with
`scripts/explore_data.py`; validate with `scripts/check_integrity.py`.

## Layout on disk

```
data/            # $VOLT_DATA_DIR, or <repo>/data — see dataspec.DATA_ROOT
├── train/
│   ├── {name}.zarr          image volume, OME-Zarr v3
│   └── {name}.geff          ground-truth lineage graph
└── test/
    └── {name}.zarr          image volume only, no labels
```

`{name}` is `{prefix}_{hash}`, e.g. `6bba_05b6850b`. The prefix identifies one
of two source recordings; the hash identifies a crop taken from it.

| Split | Datasets | `44b6` | `6bba` |
|---|---|---|---|
| train | 199 | 71 | 128 |
| test  | 4   | 2  | 2  |

**The local `test/` folder is a placeholder.** All four volumes are
byte-identical to their `train/` counterparts (verified by checksum), and they
carry no `.geff`. They exist so you can validate the submission format. The
real test set is mounted server-side at submission time and is never visible.
Any score computed on local `test/` is meaningless — the model trained on those
exact volumes.

Consequence: **discover dataset names by scanning the directory at runtime.**
The number of hidden test datasets is unknown.

## Image volumes

Every one of the 203 datasets is identical in geometry:

| Property | Value |
|---|---|
| Shape `(T, Z, Y, X)` | `(100, 64, 256, 256)` |
| dtype | `uint16` |
| Chunking | `(1, 64, 256, 256)` — one whole 3D volume per timepoint |
| Physical size | 104 × 104 × 104 µm |
| Voxel size `(z, y, x)` | 1.625, 0.40625, 0.40625 µm |
| On disk | 81.6 GB total |
| In memory as float32 | 1.56 GB per volume |

### Anisotropy

**A voxel is not a cube.** The z axis is 4× coarser than y and x. This has three
consequences that are easy to get wrong:

1. Any distance must be computed in micrometres, not voxels. Two nodes 2 voxels
   apart in z are 3.25 µm apart; 3 voxels apart in x is only 1.22 µm. Voxel-space
   distance ranks them backwards.
2. Non-maximum suppression must convert to physical units before comparing.
   `img_proc.nms_3d` does this correctly.
3. Orthogonal views must be aspect-corrected for display, or the depth axis
   looks 4× shorter than it is.

Nuclei also appear elongated along z beyond what the voxel size accounts for —
the microscope's axial point spread function is worse than its lateral one. The
detector must find z-elongated ellipsoids, not spheres.

### Chunking

The chunk is one full timepoint, so reading frame `t` costs a single ~8 MB chunk
decompression, and reading a time-clip costs one per frame. This matches how the
temporal U-Net consumes data. **No rechunking is needed.**

The corollary: reading a spatial sub-region is no cheaper than reading the whole
frame, since the chunk decompresses either way.

### Intensity

Raw intensity varies enormously between datasets. The 90th percentile spans 112
to 3569, a 32× range, and the two families differ systematically — `44b6` is
roughly 2× brighter than `6bba`.

Each zarr carries precomputed statistics in its attrs:

```
image_statistics.quantiles = {0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 1.0}
```

These are free to read and make normalization a metadata lookup rather than a
volume scan.

**The released loader normalizes by true min and max** (`io._process_on_gpu`
requests `q=0.000` and `q=1.000`, and the stored statistics satisfy that
lookup). Because 52 of 199 datasets have a maximum exceeding twice their 99.9th
percentile — worst case 15× — min/max normalization compresses real signal into
a narrow band. Measured spread of the normalized 90th percentile across
datasets:

| Clip range | Spread |
|---|---|
| 0.0 / 1.0 (current) | 42× |
| 0.001 / 0.999 | 10.4× |
| 0.01 / 0.99 | 5.4× |

The loader clamps to `[0, 4]`, not `[0, 1]`, so values above the upper clip
point survive as 1–4 and stay distinguishable. On this data even the brightest
voxel lands at 1.3–3.0 under 1/99 clipping, so nothing is lost to the clamp.

### Tissue morphology

Cells do not fill the volume uniformly, and the arrangement differs sharply
between datasets. Some are thin curved sheets with near-empty corners
(`6bba_05b6850b` holds ~15 cells); others are densely packed tissue filling most
of the frame (`44b6_5f15d135`). Signal-vs-depth is non-monotonic — it tracks the
sheet's geometry, not attenuation, so per-slice intensity correction would be
the wrong fix.

Because most of many volumes is background, most inference compute is spent on
empty space. Cropping to the tissue region is a legitimate speedup against the
12-hour budget.

## Ground-truth graphs

GEFF is a zarr group. Its internal layout:

```
{name}.geff/
├── nodes/
│   ├── ids                  int64 array of node ids
│   └── props/{t,z,y,x}/values
├── edges/
│   └── ids                  (n_edges, 2) array of [source_id, target_id]
└── attrs (zarr.json)
    └── geff/
        ├── axes             per-axis name, unit, min, max, scale
        └── extra.estimated_number_of_nodes
```

Read it with `tracksdata`'s `IndexedRXGraph.from_geff`, or directly as zarr.
Node props are **exactly** `t, z, y, x` — no `track_id`, no `seg_id`, no radius,
and no edge properties at all. Coordinates are integer voxel indices, `int64`.

### Aggregate statistics

| Quantity | Value |
|---|---|
| Total nodes | 133,318 |
| Total edges | 128,883 |
| Total divisions | **151** |
| Datasets with zero divisions | **112 of 199** |
| Median nodes per dataset | 659 |
| Median labeled nodes per annotated frame | 6.61 |
| Median labeled nodes per frame of video | 6.59 |
| Datasets containing at least one division | 87 of 199 |

The two per-frame figures differ because annotation does not always cover the
whole movie. The median dataset annotates every frame, so they nearly coincide
here, but the distinction matters for any dataset that does not.

### Topology guarantees

Verified across all 199 graphs:

- Out-degree ≤ 2 — a division has exactly two children, never three
- In-degree ≤ 1 — no merges
- Every edge spans exactly one timepoint (`t_target = t_source + 1`)
- No isolated nodes
- No cycles (implied by strictly forward edges)

Divisions carry no label. **A division is a node with two outgoing edges** —
topology, not an attribute. The metric detects them the same way.

### Sparsity

Ground truth annotates a small fraction of the cells present. `T_true`
(`estimated_number_of_nodes`) is a coarse count of *all* cells:

| Family | `T_true` median | Labeled fraction |
|---|---|---|
| `44b6` | 32,681 | 0.77% |
| `6bba` | 9,691 | 9.7% |

A 12.6× difference in labeling density between families. `T_true` itself spans
3,783 to 78,644 across datasets — a **21× range**, or 38 to 786 cells per frame.

Sparsity is why the sparse-supervision loss exists: unannotated regions are
masked out of the loss entirely, so predicting "cell" on an unlabeled cell is
neither rewarded nor punished. It is also why the metric matches by bipartite
assignment rather than requiring exhaustive coverage.

### Track structure

| Family | Median tracklet length | Tracklets per dataset |
|---|---|---|
| `44b6` | 50 frames | ~4 |
| `6bba` | 22 frames | ~30 |

No singleton tracklets exist anywhere. `44b6` looks like deliberate lineage
tracing — a few cells followed for a long stretch. Tracks are long, so a single
missed detection destroys 20–50 frames of track, which makes gap closing
valuable.

Annotation coverage is not uniform in time: 29 datasets begin after `t=0` and
25 end before the last frame. The median dataset annotates every frame it
covers.

### Motion

Frame-to-frame displacement in physical units, median across datasets:

| Percentile | µm |
|---|---|
| p50 | 1.82 |
| p90 | 3.63 |
| p99 | 6.84 |

Maximum observed: 60.76 µm in `6bba_f20478e9`, where p99 is only 11 µm — a
single extreme outlier. That dataset also has a near-black frame at `t=65`,
suggesting a dropped frame or stage shift rather than cell motion. 80 datasets
have a maximum above 10 µm; 7 above 20 µm. Across all datasets 2,700 edges
exceed 7 µm.

Split by axis, the same displacements look very different:

| Percentile | axial (z) µm | lateral (xy) µm |
|---|---|---|
| p50 | 0.00 | 1.18 |
| p75 | 1.625 | 1.82 |
| p90 | 1.625 | 2.60 |
| p99 | 4.875 | 5.17 |

The axial column is exact multiples of the z voxel, which is the tell: most
edges have **zero** z change, and any single-voxel z step contributes 1.625 µm
against a lateral median of 1.18 µm. So z changes are rare but individually
large. That inflates their share of squared displacement to 0.44, above the 0.33
an isotropic random walk would give, without implying real axial drift — the
effect is z quantization, not motion. Treat the axial share as a quantization
artefact unless a specific dataset shows a systematic z trend.

### Nearest-neighbour spacing

Between labeled nodes at the same timepoint: median 25.7 µm, 1st percentile
11.7 µm, minimum 2.87 µm. Because labeling is sparse, this **overestimates**
true cell spacing by roughly `f^(-1/3)` where `f` is the labeled fraction, so it
cannot be used to set `pool_kernel_um` directly.

Its real use is matching ambiguity: the metric pairs nodes within 7 µm, so GT
nodes closer than that to each other are the only places a prediction can be
paired with the wrong GT node. Median across datasets is 0, but 47 of 192 have
some, up to 44%.

Seven datasets never annotate two cells in the same frame, so they have no
within-frame neighbour distance at all and are excluded from every `nn_*`
aggregate. They carry no density or ambiguity evidence.

For a bound on `pool_kernel_um` that is not distorted by sparsity, use sister
separation at divisions instead — see below.

### Divisions

`reports/eda/divisions.csv` holds one row per ground-truth division, 151 rows
and 19 columns. Because the division term is only 0.1 of the score and this is
the entire population of events, the table is small enough to inspect
individually rather than summarise.

All 151 have exactly two children. 87 datasets contain at least one; 112 contain
none.

**Sister separation** — distance between the two daughters one frame after the
split:

| Percentile | µm |
|---|---|
| min | 2.87 |
| p1 | 4.11 |
| p5 | 5.41 |
| p50 | 10.57 |
| p90 | 14.36 |
| max | 20.30 |

85.4% of pairs separate by more than the 7 µm match gate, so for most events the
metric can resolve the two daughters as distinct nodes. Separation is also
*less* axial than isotropic — 0.18 of squared separation on z against 0.33 for
an isotropic split — meaning divisions separate preferentially in the lateral
plane, where resolution is four times better.

**This gives `pool_kernel_um` a bound that sparsity does not distort.** The
baseline NMS radius is 3.0 µm and the closest sister pair in the training set is
2.87 µm, so exactly one event would currently be merged into a single detection
and made unrecoverable. Raising the radius to 4 µm puts roughly one or two more
at risk; 5.4 µm risks about seven, which at the 0.1 division weight is on the
order of 0.005 of score. There is real headroom to raise it for detection
precision, and the cost is quantifiable rather than guessed. Around 4 µm is a
defensible cap.

**Parent-to-daughter displacement**, with daughters ordered by distance from the
parent:

| | µm |
|---|---|
| near daughter, median | 4.08 |
| far daughter, median | **7.13** |
| far daughter, p90 | 10.05 |
| far daughter, max | 13.53 |

The median far-daughter edge exceeds the **99th percentile of ordinary motion**
(6.84 µm). This is a structural problem for the baseline linker, separate from
division rarity: `SimpleNodeTransformer` scores each candidate pair from the
relative offset `(dz, dy, dx)` alongside appearance features, and it is trained
on 128,883 edges whose median displacement is 1.8 µm. The motion prior it learns
treats 7–10 µm as implausible, so the second edge of a division is
systematically scored as unlikely by construction.

That reframes the fix. Rather than an auxiliary division head trained on 151
examples, the cheaper interventions are `ilp_division_weight`, which directly
offsets the cost of accepting the long second edge; upweighting division edges
in the training loss, since 151 events against 128,883 edges contribute roughly
0.1% of gradient signal; and checking whether row-wise softmax hurts here, as a
dividing node must score two successors highly while softmax forces them to
compete.

**Local context availability.** `metrics.md` evaluates each division inside a
window of `parent → divider → children → grandchildren`:

| | |
|---|---|
| Parent annotated | 96.0% |
| Both children continue | 97.4% |
| Median grandchildren | 2.0 |
| Median annotated history before the event | 15 frames |
| Events within 5 frames of a movie boundary | 8 of 151 |

The window the metric needs is available for almost every event, so **division
recall is not capped by missing context**. The difficulty is detection and
linking, not what the metric will credit. Events sit near the middle of their
movies (median at 46% of elapsed time), so few are truncated.

### Crop provenance is not recorded

No OME `translation` transform exists on any dataset, and every GEFF axis
`offset` is null. Node coordinates are local to each crop, so whether crops from
the same recording overlap cannot be determined from metadata. A content probe
over 200 same-prefix pairs found no identical central patches, which rules out
aligned duplicates but not shifted partial overlap.

## Submission format

One CSV holding both row types, grouped by dataset:

```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6,node,1,0,32,128,128,-1,-1
1,44b6,node,2,1,33,130,125,-1,-1
2,44b6,edge,-1,-1,-1,-1,-1,1,2
```

- **Node rows**: `row_type=node`, integer voxel centroids, `source_id` and
  `target_id` set to `-1`
- **Edge rows**: `row_type=edge`, `node_id` and all coordinates set to `-1`
- `id` is a throwaway consecutive index
- `dataset` must match the folder name without `.zarr`
- Every test dataset must appear

`scripts/geffs_to_csv.py` and `scripts/csv_to_geffs.py` round-trip between this
and GEFF. Note that `csv_to_geffs` assigns fresh node ids and adds edges in CSV
row order.

## Scoring, and what the data implies about it

```
total_node_ratio = (N_pred - T_true) / T_true
adj_edge_jaccard = max(0, edge_jaccard * (1 - 0.1 * total_node_ratio))
score            = adj_edge_jaccard + 0.1 * division_jaccard
```

Aggregated per `metrics.summarise`: adjusted edge Jaccards weight-averaged by
`w_i = TP + FP + FN`, division Jaccard micro-averaged.

Five things follow from the measurements above.

**Edge Jaccard dominates.** The division term is capped at 0.10, so divisions
are at most 9% of the achievable score. One division is worth roughly 93 edges.
With only 151 divisions in the entire training set and 112 datasets containing
none, division Jaccard is also high-variance — a validation split must be
stratified by division count, or the metric will read as noise. `summarise`
drops the division term entirely when a split has zero divisions.

**Unmatched predictions are free.** An edge counts as TP or FP only if at least
one endpoint matches a GT node that has edges. Predicted nodes that match no GT
node are explicitly not false positives. The same holds for divisions: forks
with no local GT evidence are ignored. So over-detecting away from GT costs
nothing except through the node-count ratio.

**The node-count penalty is a guardrail, not a lever.** Under-predicting yields
a bonus, but it is capped at +10% and `TP + FN` is fixed at the GT edge count,
so any GT node you fail to match converts its edges directly to FN. Recall
dominates. The asymmetry is on the downside — 2× over gives 0.90, 5× gives 0.60,
11× gives **zero**. Rule: maximize recall, keep `N_pred` under about 2× `T_true`.

**Divisions are hard for a reason beyond rarity.** The median parent-to-far-
daughter edge is 7.13 µm, above the 99th percentile of ordinary motion. A linker
that learns a motion prior from 128,883 edges with a 1.8 µm median will score
that edge as implausible by construction. Local context is not the constraint —
96% of events have an annotated parent and 97% have both children continuing —
so the lever is the assignment cost and the training loss weighting, not more
context or a head trained on 151 examples.

**`T_true` is not available at test time and is only loosely predictable.** It
lives in the GEFF metadata, which the test split lacks. A log-linear fit from
stored intensity quantiles reaches R² 0.72 but 25% median relative error, and a
family constant is worse at 54%. Since a fixed threshold detects a roughly
constant *fraction* of cells, `N_pred ≈ f · T_true` holds automatically and the
multiplier is set by the single global `f` — so per-dataset calibration only pays
if `f` varies widely between datasets, which is measurable once a baseline
exists. Note also that `T_true` and annotated edge count are mildly *negatively*
correlated (Spearman −0.28), so the datasets most exposed to the penalty carry
28.8% of total edge weight against a 25% population share. Edge weighting does
not shield them.

## The EDA scripts

All read-only over the data root (`dataspec.DATA_ROOT`, overridable with
`--data-root`). None of them modify the dataset.

| Script | Purpose | Writes | Cost |
|---|---|---|---|
| `explore_data.py` | measure the dataset | `reports/eda/` | ~1 min |
| `visualize_data.py` | render figures to look at | `reports/figures/` | seconds per figure |
| `check_integrity.py` | assert the pipeline's assumptions, non-zero exit on failure | `reports/integrity/` | ~1 min, +4 s per `--deep` dataset |
| `check_t_true.py` | can `T_true` be predicted well enough to calibrate per dataset | stdout | instant |

`check_t_true.py` needs `reports/eda/datasets.csv`, so run `explore_data.py`
first. The other three are independent. The two measurement scripts are
documented below.

### `scripts/explore_data.py` — measurement

Produces the numbers this document is built from. Metadata-bound by default: it
never loads a volume unless you pass `--content-probe`, so a full pass over 203
datasets is about a minute.

Outputs:

- **`reports/eda/datasets.csv`** — one row per dataset, 108 columns. The
  per-dataset table you slice yourself.
- **`reports/eda/summary.json`** — aggregates, consistency checks, and the
  overlap verdict, grouped into named sections.
- **`reports/eda/divisions.csv`** — one row per ground-truth division, 151 rows
  by 19 columns. The whole event population, small enough to inspect directly.
- **`reports/eda/attrs_sample.json`** — raw zarr and GEFF attrs verbatim for the
  first few datasets, so anything the parser didn't understand is still visible.

What it measures, by section of `summary.json`:

- `counts` — datasets per split and family, label pairing
- `geometry` — shapes, dtypes, chunk shapes, disk and RAM footprint
- `scale_and_offsets` — voxel scale per dataset, where the OME `multiscales`
  block lives, whether `io._parse_scale` would find it, and whether any
  `translation` transform records crop provenance
- `normalization` — which quantile keys are stored, and whether the loader's
  lookup for `q=0.000`/`q=1.000` actually hits them
- `split_hygiene` — dataset names appearing in more than one split. Always runs
  and is not skippable, because a collision invalidates every test-split number
  in the report
- `ground_truth` — nodes, edges, divisions, tracklet lengths, temporal coverage,
  plus integrity assertions on degree, edge `dt`, and isolated nodes
- `division_events` — sister separation against the match gate, parent-to-daughter
  displacement, and how much local context each event has
- `node_count_target` — `T_true` distribution, the penalty multiplier table, and
  Spearman correlations between `T_true` and stored intensity quantiles
- `matching_ambiguity` — fraction of GT nodes with a same-timepoint neighbour
  inside the 7 µm match gate
- `hyperparameter_evidence` — edge displacement and nearest-neighbour distance
  percentiles in micrometres, with displacement decomposed into axial and
  lateral components
- `crop_overlap` — spatial overlap between same-family crops, or an explicit
  inconclusive verdict when no offsets are recorded
- `train_test_duplicates` — checksum comparison for same-named datasets across
  splits
- `content_probe` — with `--content-probe N`, samples N same-family pairs and
  compares a strided central patch for exact equality

Useful flags: `--limit N` for a quick pass, `--skip-duplicate-check` to avoid
the voxel reads, `--content-probe N` to test crop overlap by content.

Two caveats worth knowing. The content probe compares a *fixed* central patch,
so it detects identically-aligned duplicates but not crops that overlap while
shifted. And `frac_edges_disp_over_7um` compares displacement against the match
gate for scale only — 7 µm gates node matching, not cell motion.

### `scripts/visualize_data.py` — inspection

Renders PNGs to `reports/figures/`. Uses the Agg backend, so it never opens a
window. Reads one timepoint at a time, never the full 1.56 GB volume.

Five modes:

- **`overview`** — six panels at one timepoint: full-depth XY projection, a thin
  slab, a single z-plane, XZ, YZ, and an intensity-vs-depth profile with labeled
  node positions. GT nodes are red circles. Answers "what fraction of cells are
  labeled" by direct counting.
- **`timeline`** — slab projections across time with GT overlaid and a per-panel
  count of nodes in the slab. Shows motion and where annotation coverage starts
  and stops.
- **`norm`** — the same slab under min/max, 0.1/99.9, and 1/99 clipping, with
  raw and normalized histograms. Panel titles report what fraction of voxels
  exceed 1.0, what fraction hits the pipeline's 4.0 clamp, and — the number that
  matters — how many *annotated cells* get clamped.
- **`tracks`** — GT only, no image: XY trajectories coloured by lineage,
  temporal coverage per lineage, and the displacement histogram.
- **`zoom`** — four consecutive frames cropped around one event. Defaults to the
  first division; `--longest-edge` targets the largest displacement instead,
  which is how to eyeball a suspected annotation error.

Flags worth knowing:

- `--slab N` — z-slices per thin-slab projection. A full-depth projection
  superimposes 64 planes of tissue; `--slab 4` separates nuclei far better.
- `--proj-width N` — voxels collapsed in the XZ/YZ panels, default 64. On dense
  tissue a full-range projection saturates by construction, because every
  `(z, x)` column contains a bright nucleus at some `y`. `--proj-width 0`
  restores the full range, which is fine on sparse datasets.
- `--t-start` / `--t-stop` — window the timeline to consecutive frames. Without
  them the stride is ~11 frames, and at under 2 µm median motion per frame a
  cell shifts two nuclear diameters between panels, so identity is not traceable
  by eye.

For interactive 4D scrubbing, napari is the better tool and the repo supports it
via `io.Dataset.napari_tracks()`.

## Reproducing and validating

```bash
# measure everything above. --content-probe reads voxels and is the slow part;
# drop it to refresh the tables only.
uv run python scripts/explore_data.py --skip-duplicate-check

# validate the assumptions the pipeline relies on
uv run python scripts/check_integrity.py
uv run python scripts/check_integrity.py --deep 12

# node-count target analysis (needs datasets.csv from the first command)
uv run python scripts/check_t_true.py

# read the full summary, not just the truncated console print
uv run python -c "import json; s=json.load(open('reports/eda/summary.json')); [print(f'\n[{k}]', json.dumps(s[k], indent=2)) for k in s]"

# look at it
uv run python scripts/visualize_data.py --mode overview --dataset 6bba_05b6850b --t 50 --slab 4
uv run python scripts/visualize_data.py --mode norm --n-datasets 8
```

`check_integrity.py` exits non-zero on any FAIL, so it can gate a pipeline run.
Its most useful single check is the coordinate-convention test: it compares
intensity at annotated positions against random positions, and a ratio below 1
means the axis order is being misread.
