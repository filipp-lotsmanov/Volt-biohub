# Offline Kaggle submission chain

This is a **Notebook** competition: internet disabled, 12 h wall clock (CPU *and*
GPU), output named exactly `submission.csv`. External data and pre-trained models
are explicitly allowed, so bundling wheels and weights as Kaggle Datasets is
sanctioned.

Everything here is glue plus one CLI. The pipeline itself lives in
`src/volt_biohub/`; nothing in this directory makes a modelling decision.

```
kaggle/
  submit.py                  the inference CLI: test dir -> validated submission.csv
  submission.py              row writer + write-validate-move ordering
  check_offline.py           gate: does the path import with tracksdata absent?
  requirements-offline.txt   the derived offline package set
  notebook.py                the ~35 lines that run on Kaggle
  kernel-metadata.json       for `kaggle kernels push`
```

---

## The offline gate — run this before every upload

```bash
uv run python kaggle/check_offline.py
```

It re-execs itself in a **child interpreter**, hides `tracksdata` (plus the
EDA-only packages) behind a `sys.meta_path` finder that raises, imports the whole
offline path, and then **runs it end to end** on two synthetic image-only zarr
datasets, asserting that a validated `submission.csv` comes out. Non-zero exit
means do not build the code dataset.

Three things about its shape, each of which was wrong in an earlier version and
each of which let it pass on a tree that could not run offline:

* **Child process, not in-process.** Anything already imported was served from
  `sys.modules` without re-executing its imports, so merely pre-importing
  `volt_biohub.linking` made the gate report the path clean.
* **`sys.meta_path`, not `builtins.__import__`.** `importlib.import_module` does
  not call `builtins.__import__`. Measured on a one-line stub package: the old
  hook let `importlib.import_module` through and handed back the module. It
  stopped *tracksdata* only by accident, because tracksdata's `__init__`
  re-imports its own subpackages through `__import__`.
* **It runs, not just imports.** A lazy import inside a function body is
  invisible to an import gate — which is precisely the fix pattern used in
  `linking.py`, so an import-only gate would have been blind to the one thing it
  most needed to see. Verified by temporarily making `submit.run` call
  `linking.build_graph`: the import phase stays green and the run phase fails
  with the blocked-module traceback.

### What was blocking, and how it was fixed

`src/volt_biohub/linking.py` used to `import polars as pl` / `import tracksdata
as td` at module scope, and `submit.py` imports `link_sequence` from it.
`tracksdata` is a git dependency with 20 transitive requirements including `ilpy`
(needs SCIP), a Rust extension (`rstar-python`) and `numba` — bundling it is a
project, not a wheel. Both imports now live inside `build_graph()`, the only
function that needs them, which exists solely to hand a graph to the local
scorer. `from __future__ import annotations` is already in that file, so the
`-> td.graph.InMemoryGraph` return annotation never evaluates.

`scripts/geffs_to_csv.py` was the second blocker and is now tracksdata-free for
an independent reason: `tracksdata.from_geff` **drops the z/y/x props of a
zero-node graph**, so the writer raised `ColumnNotFoundError` on exactly the
empty `.geff` that `run_predictions(skip_failures=True)` writes for a dataset
that failed — the graceful-degradation path was a guaranteed total failure at the
end of a 12 h run. It now reads the `.geff` with plain zarr (`nodes/ids`,
`nodes/props/<k>/values`, `edges/ids`; the layout is identical for the
organizers' files and tracksdata's), so an empty graph is an empty array, and it
emits one placeholder node row for a dataset that produced none — because a
dataset absent from the CSV invalidates the whole submission.

`src/volt_biohub/scoring.py` and `io.py` still import tracksdata, and that is
correct: they read ground-truth `.geff` files, which the test set does not ship,
so they are structurally local-only. `scripts/predict_unet_transformer.py` keeps
its module-scope import too — its closure (`volt_biohub.io`,
`volt_biohub.metrics`, `evaluate`, `train_unet_transformer`) needs tracksdata in
every branch, so lazifying one line there would advertise a capability that does
not exist. `submit.py` replaces it on Kaggle.

---

## Build the three datasets

Do this on the Linux box, which has the data, the venv and outbound internet.
Auth is already wired (`KAGGLE_API_TOKEN` points at
`/home/y2a/.kaggle/access_token`). Replace `USERNAME` with the account that owns
the datasets, in the commands *and* in `kernel-metadata.json`.

### 1. `volt-wheels`

`requirements-offline.txt` explains why each package is or is not in the list.
`--platform manylinux2014_x86_64` and `--python-version 3.12` target the Kaggle
image, not this box, so the wheels are correct even though they are downloaded
here.

**3.12, not 3.11 — measured on the image 2026-09-11.** `.python-version` in this
repo pins 3.11 and used to claim it matched Kaggle; the image is now
**3.12.13**. Wheels built for 3.11 carry `cp311` ABI tags, and `pip install
--no-index` then finds no candidate and fails the whole run at minute one. Our
own source is 3.11/3.12-clean either way, so the repo pin is unaffected — but
the `--python-version` on this command is load-bearing and must track the image.
Re-probe it before each competition (see *Probing the image*).

`--no-deps` is not optional: the list in `requirements-offline.txt` is already
complete, and without the flag pip resolves zarr's `numpy>=...` and drops a numpy
wheel into the directory — which on Kaggle would silently upgrade the numpy that
the image's torch and scipy are built against. The venv has no `pip`, hence
`uv run --with pip`.

```bash
cd /home/y2a/Volt-biohub
rm -rf /tmp/volt-wheels && mkdir -p /tmp/volt-wheels
uv run --with pip python -m pip download \
    -r kaggle/requirements-offline.txt \
    -d /tmp/volt-wheels \
    --no-deps \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version 3.11

# Expect exactly 12 wheels and nothing else. An sdist means pip will try to
# build it on Kaggle, which needs a compiler and sometimes the internet; a numpy
# wheel means --no-deps was dropped.
ls /tmp/volt-wheels | grep -v '\.whl$' && echo "NON-WHEEL PRESENT — fix before uploading"
ls /tmp/volt-wheels | grep -i '^numpy' && echo "NUMPY PRESENT — remove it, see above"
# Wrong-ABI check. cp311/cp310 wheels will not install on the image's 3.12.
# `cp310-abi3` is NOT a hit: abi3 is the stable ABI and is forward-compatible,
# which is why polars_runtime_32 legitimately ships as cp310-abi3.
ls /tmp/volt-wheels | grep -E 'cp3(9|10|11)-cp3' && echo "WRONG ABI — rebuild with the right --python-version"

cat > /tmp/volt-wheels/dataset-metadata.json <<'EOF'
{"title": "volt-wheels", "id": "USERNAME/volt-wheels", "licenses": [{"name": "other"}]}
EOF
uv run kaggle datasets create -p /tmp/volt-wheels -r zip
```

### 2. `volt-code`

The repository at a known commit. Put the git SHA in the version message so a
leaderboard entry traces back to code.

```bash
SHA=$(git rev-parse --short HEAD)
rm -rf /tmp/volt-code && mkdir -p /tmp/volt-code
git archive HEAD src kaggle | tar -x -C /tmp/volt-code

cat > /tmp/volt-code/dataset-metadata.json <<'EOF'
{"title": "volt-code", "id": "USERNAME/volt-code", "licenses": [{"name": "other"}]}
EOF
uv run kaggle datasets create -p /tmp/volt-code -r zip     # first time
uv run kaggle datasets version -p /tmp/volt-code -m "$SHA" -r zip   # every push after
```

`git archive HEAD` and not `cp -r`: it takes exactly what is committed, so the
dataset cannot silently contain uncommitted local edits — which is the whole
point of recording the SHA.

### 3. `volt-weights`

The checkpoint **and its `config.json`**. The config is not optional:
architecture is not inferable from a state dict, and `submit.py` refuses to run
without it. The transformer weights in the checkpoint are unused (the geometric
linker replaced it) but harmless.

```bash
rm -rf /tmp/volt-weights && mkdir -p /tmp/volt-weights
cp -r weights/modelA_corrected/split_0 /tmp/volt-weights/

cat > /tmp/volt-weights/dataset-metadata.json <<'EOF'
{"title": "volt-weights", "id": "USERNAME/volt-weights", "licenses": [{"name": "other"}]}
EOF
uv run kaggle datasets create -p /tmp/volt-weights -r zip
```

**What Kaggle's unzip actually does — measured on the real upload 2026-09-11,
and it is not consistent.** `-r zip` sends each top-level subdirectory as
`<name>.zip`. Kaggle then extracts them, and whether the directory level
survives depends on how many there were:

| dataset | uploaded | mounted as |
|---|---|---|
| `volt-code` | `kaggle.zip`, `src.zip` | `kaggle/...`, `src/...` — level **kept** |
| `volt-weights` | `split_0.zip` | `config.json`, `edge_predictor_best.pth` at the **root** — level **dropped** |

So the `split_0/` this file used to tell you to keep does **not** exist on the
mount, and `notebook.py`'s old hardcoded path resolved to nothing. That failure
is silent by construction: `HAVE_WEIGHTS` goes False, the DoG fallback engages,
the run completes and validates, and the submission is quietly worth 0.6753
instead of 0.7366. `notebook.py` now calls `find_weights()`, which searches the
mount for `edge_predictor_best.pth` beside a `config.json`, so either layout
works and a future re-upload cannot reintroduce this.

**Verify the mount rather than trusting the upload**, whatever you change:

```bash
uv run kaggle datasets files oleksiikrasnoshtanov/volt-weights
```

---

---

---

## How code competitions actually work

Researched 2026-09-11 against Kaggle's own docs and the community write-ups
linked at the bottom, because we had been reasoning from the competition page
alone and three of our assumptions were wrong.

**The rerun is the submission.** You submit a *notebook version*, not a file.
Kaggle then privately re-executes that notebook end to end with the competition
dataset **replaced by a hidden version**, validates the execution against the
competition's constraints, extracts the submission file from its output
directory, and scores that. Consequences we had not internalised:

* **The local `submission.csv` in the repo root is not submittable and never
  was.** `kaggle competitions submit -f submission.csv` names *the file the
  kernel produces*, not a file to upload. The flag looks identical to the
  regular-competition upload flag and does something completely different.
* **The run you watch after `kernels push` is not the graded run.** It executes
  against the four visible placeholder samples. Its only jobs are to prove the
  code runs and to give you a per-video timing to extrapolate from.
* **The hidden set can be larger, smaller or otherwise different.** Ours is
  "approximately the size of the training set" — 199 videos — against 4 visible.
  So timing must always be extrapolated, never read off the visible run.

**Errored submissions still count against the daily limit.** This is deliberate:
otherwise a failing submission would be a free probe of the hidden data. With
five per day, a blind guess costs a fifth of a day's budget, which is why the
probe-kernel habit below is worth its minute. Ours cost three of them before
this was understood.

**The four failure modes, and what each actually means:**

| shown | means |
|---|---|
| Notebook Threw Exception | your code raised. The log is visible; read it. |
| Submission CSV Not Found | the rerun finished but no file was at the expected path — usually an upstream error that halted execution before the write |
| Notebook Timeout | exceeded the competition's runtime limit *on the hidden set*, which is the one you never measured directly |
| Submission Scoring Error | the file exists but is malformed: wrong columns, wrong row count, empty values, bad dtypes, invalid values |

The last one is the argument for `validate_submission()` running before every
save, which it now does — a scoring error is indistinguishable from a modelling
failure on the leaderboard, and costs a slot either way.

**You cannot see the rerun.** Its log is not exposed. So everything you want to
know about the environment has to be established by a *visible* run — hence
*Probing the image* above. Establish it once, cheaply, and write it down.

**Shipping code: three options, and why we use the one we do.**

| method | good for | why not here |
|---|---|---|
| Utility Script (attach a kernel as a module) | one file of helpers | ours is a package, `src/volt_biohub/...`, not a single module |
| **Dataset of `.py` files + `sys.path`** | **a whole repo at a known commit** | **what we do** |
| import another notebook's `/kaggle/working` | chaining train → infer | adds a second kernel to keep in sync for no gain |

The Dataset route also gives the property we most want: the version message
carries the git SHA, so a leaderboard entry traces back to a commit.

**Why a `.py` script kernel rather than a `.ipynb`.** `kaggle kernels push`
accepts a plain `.py` when `kernel_type` is `script`, and it produces the same
graded `submission.csv`. A notebook is JSON with embedded outputs, so it diffs
unreadably and cannot be reviewed in a PR. Since this file is glue by design,
the notebook format would add only noise. It still runs top to bottom if pasted
into the Kaggle editor.

**Checklist before spending a submission**

1. `uv run python kaggle/check_offline.py` — imports and runs with tracksdata hidden.
2. `kaggle datasets files <each>` — sizes match local, so you know the *version*
   you think is mounted is the one that is.
3. `kaggle kernels push` → `status` → `output`, and **read the log**: the
   detector actually chosen, the test dir actually found, the per-video timing.
4. Extrapolate that timing to the hidden set's size and compare to the cap.
5. Only then `competitions submit -k … -v …`.

Sources: [Kaggle — Code Competitions, Errors & Debugging Tips](https://www.kaggle.com/code-competition-debugging),
[Kaggle — Competitions Documentation](https://www.kaggle.com/docs/competitions),
[Kaggle — Notebooks Documentation](https://www.kaggle.com/docs/notebooks),
[Guide on Submitting Offline Kernels to a Kaggle Competition](https://medium.com/data-science/submitting-model-predictions-to-kaggle-competition-ccb64b17132e),
[Organizing Code, Experiments, and Research for Kaggle Competitions](https://towardsdatascience.com/organizing-code-experiments-and-research-for-kaggle-competitions/).

---

## Probing the image

**Do this before trusting any run, and re-do it when Kaggle updates the image.**
Three of the assumptions this file used to state as fact were wrong, and each
would have cost a full submission cycle. A throwaway script kernel with the
competition attached and no datasets settles all of them in about a minute:

```python
import sys
from pathlib import Path
print("PYTHON", sys.version)
for p in sorted(Path("/kaggle/input").iterdir()):
    print("== MOUNT", p.name)
    for q in sorted(p.rglob("*")):
        if len(q.relative_to(p).parts) <= 2:
            print("   ", q.relative_to(p))
import torch
print("CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Measured 2026-09-11:

| question | assumed | actual |
|---|---|---|
| Python | 3.11 | **3.12.13** |
| competition mount | `/kaggle/input/<slug>/test` | **`/kaggle/input/competitions/<slug>/test`** |
| GPU under `enable_gpu` | "a GPU" | **Tesla P100, sm_60** |
| preinstalled | — | torch 2.10.0+cu128, numpy 2.0.2, scipy 1.16.3, polars 1.35.2; **no zarr** |

**The GPU one is the dangerous one.** `enable_gpu: true` hands out a P100 at
compute capability **sm_60**, and the image's torch 2.10.0+cu128 ships cubins
for sm_70 upward only. `torch.cuda.is_available()` returns **True** on that
machine and then every convolution raises "no kernel image is available for
execution on the device" — an availability check is not merely insufficient
here, it is confidently wrong. Two defences, both needed:

* `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json` asks for sm_75,
  which this torch supports. Valid values are `NvidiaTeslaT4`,
  `NvidiaTeslaP100`, `Tpu1VmV38` (`kagglesdk` `machine_shape`); the CLI also
  takes `--accelerator`. Kaggle substitutes hardware when a shape is short, so
  this is a request, not a guarantee.
* `notebook.py::gpu_is_usable()` runs one real `conv3d` and falls back to DoG if
  it raises. Execute, never infer — the same rule that settled `sm_89` on the
  training box.

---

## Wait for `ready` before pushing the kernel

`kaggle datasets version` returns as soon as the **upload** finishes, but Kaggle
then processes the version asynchronously — and a kernel pushed into that window
mounts the *previous* content. That cost a run: the kernel died on
`No such file or directory: /kaggle/input/volt-code/kaggle/requirements-offline.txt`
while a probe minutes later showed the file present and the tree exactly right.
Nothing in the failure points at a race; it reads as a wrong path, which is what
makes it worth a line here.

```bash
for d in volt-wheels volt-code volt-weights; do
  until [ "$(uv run kaggle datasets status oleksiikrasnoshtanov/$d)" = "ready" ]; do
    sleep 10
  done
done
uv run kaggle kernels push -p kaggle
```

## Push and run the kernel

`notebook.py` is a **Script** kernel, not a `.ipynb`. A notebook is JSON with
embedded outputs: it diffs unreadably, cannot be reviewed in a PR, and adds
nothing here because the file is glue. `kaggle kernels push` accepts a `.py` for
`kernel_type: "script"` and produces the same graded `submission.csv`. It also
pastes straight into the Kaggle editor if you would rather click.

```bash
sed -i 's/USERNAME/<your-kaggle-username>/g' kaggle/kernel-metadata.json
uv run kaggle kernels push -p kaggle
uv run kaggle kernels status USERNAME/volt-biohub-submission
uv run kaggle kernels output USERNAME/volt-biohub-submission -p /tmp/out
```

`kernel-metadata.json` already sets `enable_internet: false` and
`enable_gpu: true`. Both matter — see the timing section.

---

## Run it locally first

The local `test/` folder is four samples byte-identical to their train
counterparts, so its *score* is meaningless. Its *format* is not: it is the only
way to exercise the writer before uploading.

```bash
# classical detector, no weights, no GPU
uv run python kaggle/submit.py \
    --test-dir /home/y2a/data/biohub/test \
    --out /tmp/subout/submission.csv

# learned detector
uv run python kaggle/submit.py \
    --test-dir /home/y2a/data/biohub/test \
    --weights weights/modelA_corrected/split_0/edge_predictor_best.pth \
    --out /tmp/subout/submission.csv
```

`--limit N` runs the first N datasets for a timing probe and deliberately leaves
the result **unvalidated**, because a partial run is genuinely missing datasets.
Never submit one.

Defaults are the best configuration measured to date — 8 µm linking gate,
division pass off, node budget 1.0× the predicted `N_true` — which scored 0.6925
on LOEO fold 0 against the classical floor's 0.6424. The gate is a *per-detector*
knob: re-sweep it if the detector changes, never carry one detector's value to
another.

### Does the CSV lose score?

It does not, and this was measured rather than assumed. Scoring the same
predictions two ways — straight from `(coords, links)` via
`linking.build_graph`, and after `submit.py` wrote them to CSV and
`csv_to_geffs.py` read them back — agrees to **0.00e+00 on all four datasets**
(adjusted edge Jaccard 0.342665 / 0.246890 / 0.710134 / 0.384507 either way).
Re-run that check whenever the writer changes; it is the only thing standing
between a coordinate-space bug and a wasted submission.

Both directions of the coordinate-space bug are covered, and this was checked
rather than assumed. The bounds check catches an *over*-scaled upsample (y=504 in
a 256-wide volume). The *under*-scaled direction — coordinates left on the 64³
isotropic grid, which stay inside the original volume and would raise nothing on
bounds alone — is caught separately by `volt_biohub.submission`'s
`suspect_isotropic_coords` check, which fires at **error** level once the pattern
appears on at least half the datasets (a coordinate-space bug is global, so that
threshold keeps a genuinely localised sparse prediction from being rejected).
Verified by writing a deliberately un-upsampled CSV: `validate_submission` returns
`ok = False` and names all four datasets with `y_max=63` against `y_extent=256`.
The round-trip score check is still worth re-running when the writer changes, but
it is no longer the only defence.

### What diffing against the oracle actually caught

One real bug, which no validator would have flagged: `csv.DictWriter` defaults to
`lineterminator="\r\n"`, and with `newline=""` that CR reaches the file. Our
first output was CRLF while `sample_submission.csv` is LF. Both parse fine
everywhere we can test, so this may well have been harmless — but it was free to
fix and the cost of being wrong is a submission slot. The header is now
byte-identical to the oracle's and the file contains zero CR bytes.

One deliberate difference remains: we emit all node rows for a dataset and then
all its edge rows, while the oracle interleaves them. That is irrelevant — `id`
is documented as a throwaway consecutive index, `csv_to_geffs.py` groups by
dataset before resolving edges, and `geffs_to_csv.py` (the reference writer)
concatenates nodes-then-edges exactly as we do.

---

## The 12 h budget

Measured on the Linux box, 100-frame videos, extrapolated to a hidden test set
"approximately the size of the training set" (199 samples):

| detector | hardware | per video | 199 videos |
|---|---|---|---|
| DoG | CPU, 4 threads | 5.9 s | **0.32 h** |
| U-Net | CPU, 8 threads | ~299 s (3.0 s/frame) | **16.5 h — over the cap** |
| U-Net | **Kaggle T4** | **13.3 s** | **0.74 h** |
| **switch, mixed** | **Kaggle T4** | **8.9 s** | **0.49 h** |

Measured on the real kernel 2026-09-11, not extrapolated from this box: the
four-sample visible test set ran DoG at 3.8–4.0 s and the U-Net at 13.3 s per
video, for 8.9 s/video mixed. **The cap is not the constraint it was feared to
be — 0.49 h against 12 h.** That is 24x headroom, which is worth knowing in the
other direction: the budget is nowhere near binding, so a *more* expensive
detector, flip TTA, or a second model are all affordable. The 12 h worry came
from CPU numbers on this box and does not survive contact with a GPU kernel.

Two conclusions:

* **The U-Net path needs a GPU.** On CPU it does not finish, and Kaggle's
  notebooks have a small single-digit vCPU count against this box's 256, so the
  real CPU number is worse than 16.5 h. `enable_gpu: true` is not optional.
* **DoG is the safe fallback.** No weights, no GPU, 20 minutes. The fallback
  lives in `notebook.py`, not in `submit.py`: `run()` uses DoG only when
  `weights=None`, and passing it a path that does not exist (or one whose
  `config.json` is missing) exits with a message and writes nothing. `notebook.py`
  therefore checks for both files and passes `None` if either is missing, so a
  half-built weights dataset degrades to a scoring submission rather than to no
  submission.

**Open, and newly discovered: node count is not reproducible across GPUs.** The
same checkpoint, config and input produced **50,561** nodes for
`44b6_0b24845f` on this box's A6000 and **34,837** on Kaggle's T4 — 31 % short
of a budget it filled locally. Not precision: `load_detector` is fp32 on both
and never calls `.half()`. The mechanism is the saturated map's *plateaus*.
`detectors._top_peaks` marks `field >= pooled`, so on a plateau where many
voxels are exactly equal to the window maximum, every one of them is a "peak" —
and how many compare exactly equal depends on the cuDNN convolution algorithm,
which differs between an A6000 and a T4. Consequences:

* **Our CV node-count calibration was measured on hardware the submission does
  not run on.** The node-count penalty is a function of `N_pred/N_true`, so a
  31 % shortfall moves the multiplier — favourably here, since under-predicting
  multiplies J *above* 1.0, but it costs edges outright at two per missed node.
* It is a second, independent argument for fixing the saturation itself
  (`det_neg_weight`, `bright_threshold`) rather than tuning around it: a
  detector whose output has real dynamic range would not have ties to break.
* Until then, treat any CV-vs-LB gap on this submission as *partly* this, and
  do not spend a day hunting a scoring bug before ruling it out.

Also still useful to record here —

```bash
CUDA_VISIBLE_DEVICES=$(volt_gpu) uv run python kaggle/submit.py \
    --test-dir /home/y2a/data/biohub/test \
    --weights weights/modelA_corrected/split_0/edge_predictor_best.pth \
    --out /tmp/subout/submission.csv --limit 2
```

— but an A6000 is much faster than a T4, so treat it as a lower bound and get the
real figure from a `--limit 2` kernel run before trusting the full one.

---

## Node budget without `N_true`

Both detectors take a node budget rather than a probability threshold, because
the metric penalises node *count*, not confidence. The budget's denominator is
`estimated_number_of_nodes`, which lives only in the ground-truth `.geff` — the
test set ships images alone.

`submit.py` predicts it from `image_statistics.quantiles`, which the test images
*do* carry: `log(N_true) = 0.7992 · log(q0.9 + 1) + 4.5725`, least squares over
the 199 training samples. R² 0.722 in log space, 25.3 % median relative error,
and an edge-weighted node multiplier of **0.9927** against **0.936** for the best
single global constant. `scripts/check_t_true.py` reproduces those numbers.

It is a weak *per-sample* predictor — on the four local placeholders it lands
between 0.50× and 1.54× of truth — but the penalty is soft in both directions
(±10 % relative), so being roughly right per sample beats being exactly wrong
everywhere.

---

---

## Submitting — the actual procedure

Submission is by **notebook**, not by CSV upload. `kaggle competitions submit
-f submission.csv` names *the output file the kernel produces*; the local CSV in
the repo root is never uploaded and cannot be submitted. Kaggle **reruns** the
notebook against the hidden test set when you submit — the run you watch after
`kernels push` only touches the four visible placeholder samples.

The CLI does all of it; the Kaggle MCP server is not needed and adds a browser
OAuth step for nothing. Auth here is the `KGAT_` token at
`$KAGGLE_API_TOKEN` (`/home/y2a/.kaggle/access_token`, on the persistent
volume).

```bash
# 1. datasets  (see "Build the three datasets")
uv run kaggle datasets version -p /tmp/volt-code -m "$(git rev-parse --short HEAD)" -r zip

# 2. confirm the new version is actually mounted — NOT `datasets status`,
#    which says "ready" about the dataset while a version is still processing
uv run kaggle datasets files oleksiikrasnoshtanov/volt-code   # sizes must match local

# 3. push, run, read the log
uv run kaggle kernels push -p kaggle
uv run kaggle kernels status oleksiikrasnoshtanov/volt-biohub-submission
uv run kaggle kernels output oleksiikrasnoshtanov/volt-biohub-submission -p /tmp/out

# 4. submit that kernel VERSION (-v), with -f naming the kernel's output file
uv run kaggle competitions submit biohub-cell-tracking-during-development \
    -k oleksiikrasnoshtanov/volt-biohub-submission -v 4 \
    -f submission.csv -m "<git sha> <what changed>"
uv run kaggle competitions submissions biohub-cell-tracking-during-development
```

Read the kernel log before submitting, every time. It prints the detector
actually chosen, the test directory actually found and the per-video timing —
the three things that have each silently been wrong once.

---

## Notes for whoever touches this next

* **`uv sync` no longer installs pandas or matplotlib.** They moved to the `eda`
  extra, since the runtime dependency list is what gets bundled as offline
  wheels. `explore_data.py`, `check_t_true.py` and `visualize_data.py` need
  `uv sync --extra eda`. CLAUDE.md's Commands section still says
  `uv sync # add --extra viz for napari` and wants `--extra eda` added too.
* **`kagglehub` and `python-dotenv` were removed, not moved.** Grepping the whole
  repo finds no import of either; `fetch_data.py` uses the `kaggle` CLI from the
  dev group.
* **`submit.load_volumes` duplicates `scoring.load_volumes`.** They must not
  drift — a normalisation mismatch means the detector measures the mismatch
  rather than the image — and the duplication exists only because `scoring.py`
  imports tracksdata. The right fix is a tracksdata-free image module in
  `volt_biohub` that both import.
* **`submit.load_detector` duplicates `UNetNodeTransformer.detect`** for the same
  reason: `train_unet_transformer.py` imports tracksdata.
* **Validation is not implemented here.** `volt_biohub.submission` is the
  authoritative validator and needs only polars and zarr, both bundled.
  `submission.write_validated` contributes the *ordering*: write to
  `.volt-staging/`, validate that, then rename into place — so a rejected
  submission never leaves a plausible-looking `submission.csv` for the notebook
  to hand in.
* **A dataset that detects nothing produces no rows and is therefore absent, not
  empty.** `submit.run` emits one placeholder node at the volume centre in that
  case, with a warning, because every test dataset must appear. It should be
  unreachable (the budget floors at one peak per frame), so the warning firing
  means something upstream is wrong. `scripts/geffs_to_csv.py` applies the same
  policy at the origin, where it has no volume shape to centre on.
