"""Kaggle submission kernel. Glue only — every decision lives in the repo.

Why a .py Script kernel and not a .ipynb: a notebook is JSON with embedded
outputs, so it diffs unreadably and cannot be reviewed in a PR, while
`kaggle kernels push` accepts a plain .py for kernel_type "script" and produces
the same graded submission.csv. Since this file is glue by design, the only thing
a notebook format would add is noise. It runs top to bottom in the Kaggle editor
too, if you would rather paste it.

Three input datasets, attached in the kernel settings (see kernel-metadata.json):
  volt-wheels    offline wheels, built by the pip download in README.md
  volt-code      this repository at a known commit
  volt-weights   the detector checkpoint AND its config.json

Set INTERNET off and, if using the U-Net detector, ACCELERATOR to a GPU: the
detector costs ~3.0 s/frame on CPU, which is ~16.5 h for a train-sized hidden
test set against a 12 h cap. Omitting the weights falls back to the classical DoG
detector, which needs no GPU (measured 5.9 s/video, ~0.32 h for 199 videos) and
is the fallback that always finishes.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/input")


def find_in_mounts(pattern: str, *, max_depth: int = 4) -> Path | None:
    """First match for *pattern*, searched shallowly across the dataset mounts.

    Nothing here hardcodes a mount path any more, and that is the product of
    three separate failures rather than caution. `kaggle datasets create -r zip`
    uploads each top-level subdirectory as `<name>.zip`, and Kaggle's extractor
    keeps the directory level for a dataset with two of them (volt-code:
    `kaggle/`, `src/`) and drops it for one with a single archive (volt-weights:
    `split_0/` gone, contents flat at the root). Meanwhile the competition does
    not mount at `<slug>/` at all but at `competitions/<slug>/`. Every one of
    those was documented wrong, and each cost a push-run-read cycle to find.

    Bounded depth, and the competition mount skipped, on purpose: it holds
    **90,776 entries** of zarr chunk files (measured), so an unbounded
    `rglob` over `/kaggle/input` spends minutes walking it before finding a
    12-file wheels directory.
    """
    for mount in sorted(q for q in ROOT.iterdir() if q.is_dir()):
        if mount.name == "competitions":
            continue
        for depth in range(max_depth):
            hits = sorted(mount.glob("/".join(["*"] * depth + [pattern])))
            if hits:
                return hits[0]
    return None


def find_weights() -> Path | None:
    """The checkpoint, wherever Kaggle's unzip put it — beside its config.json.

    The sibling config is required, not optional: architecture is not inferable
    from a state dict and `submit.load_detector` exits without it, so a `.pth`
    found alone must read as "no weights" rather than as a checkpoint that kills
    an eleven-hour run at minute one.
    """
    for mount in sorted(q for q in ROOT.iterdir() if q.is_dir()):
        if mount.name == "competitions":
            continue
        for depth in range(4):
            for pth in sorted(mount.glob(
                    "/".join(["*"] * depth + ["edge_predictor_best.pth"]))):
                if (pth.parent / "config.json").exists():
                    return pth
    return None


REQUIREMENTS = find_in_mounts("requirements-offline.txt")
_pkg = find_in_mounts("volt_biohub/metrics.py")
SRC = _pkg.parent.parent if _pkg else None
_whl = find_in_mounts("*.whl")
WHEELS = _whl.parent if _whl else None
if not (REQUIREMENTS and SRC and WHEELS):
    # Hard failure, unlike the weights: without the code and the wheels there is
    # no submission of any kind, so say exactly which mount is missing rather
    # than dying twenty lines later on an import.
    raise SystemExit(
        f"missing inputs — requirements={REQUIREMENTS} src={SRC} wheels={WHEELS}. "
        f"mounts: {sorted(q.name for q in ROOT.iterdir())}. Attach volt-code and "
        f"volt-wheels, and check `kaggle datasets files` shows the version you "
        f"expect: a version still being processed mounts as the previous one."
    )

WEIGHTS = find_weights()
HAVE_WEIGHTS = WEIGHTS is not None
OUT = Path("/kaggle/working/submission.csv")



def find_test_dir(root: Path) -> Path:
    """The competition's ``test/``, wherever Kaggle mounts it.

    Never hardcoded and never read out of ``sample_submission.csv``: that file
    names the four local placeholder datasets and they are replaced at rerun,
    so reading names from it submits an empty file.

    The previous glob was ``*/test`` — one level, on the assumption that a
    competition mounts at ``/kaggle/input/<slug>/``. Measured: it mounts at
    ``/kaggle/input/competitions/<slug>/test``, two levels down, so the glob
    matched nothing and the kernel died on ``StopIteration`` in its first
    seconds. Search a few depths and require the candidate to actually contain
    ``.zarr`` folders, so a mount layout change fails loudly here rather than
    producing an empty submission eleven hours later.
    """
    seen = [p for pat in ("*/test", "*/*/test", "*/*/*/test")
            for p in sorted(root.glob(pat)) if p.is_dir()]
    for p in seen:
        if any(p.glob("*.zarr")):
            return p
    raise SystemExit(
        f"no test/ directory containing *.zarr under {root}. "
        f"test-like dirs seen: {seen}. "
        f"mounts: {sorted(q.name for q in root.iterdir()) if root.exists() else 'none'}"
    )


def gpu_is_usable() -> bool:
    """Does a CUDA kernel actually *run* here — not just ``is_available()``.

    Measured on the real image: ``enable_gpu`` hands out a **Tesla P100**,
    compute capability **sm_60**, while the preinstalled torch 2.10.0+cu128
    ships cubins for sm_70 and up. ``torch.cuda.is_available()`` returns True
    on that machine and every convolution then dies with "no kernel image is
    available for execution on the device". An availability check is therefore
    worse than useless here: it is confidently wrong.

    ``machine_shape: NvidiaTeslaT4`` in kernel-metadata.json asks for sm_75
    instead, which this torch does support — but Kaggle substitutes hardware
    when a shape is unavailable, so the request is not a guarantee and this
    stays in. Run the smallest real conv3d and believe the result.
    """
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, 1, 4, 4, 4, device="cuda")
        torch.nn.Conv3d(1, 1, 3, padding=1).cuda()(x).sum().item()
        return True
    except Exception as exc:
        print(f"  GPU present but unusable ({type(exc).__name__}: "
              f"{str(exc)[:200]}) -> DoG", flush=True)
        return False


TEST = find_test_dir(Path("/kaggle/input"))

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={WHEELS}",
     "-r", str(REQUIREMENTS)],
    check=True,
)

sys.path[:0] = [str(SRC), str(REQUIREMENTS.parent)]

import torch  # noqa: E402
import zarr  # noqa: E402

# Fail in minute one rather than hour eleven. zarr 2.x cannot open a Zarr v3
# store at all, and the U-Net without a working GPU cannot finish inside 12 h.
print(f"zarr {zarr.__version__} · torch {torch.__version__} · "
      f"cuda {torch.cuda.is_available()} · {sys.version.split()[0]} · "
      f"test dir {TEST}", flush=True)

# The U-Net needs BOTH the checkpoint and a GPU that actually executes kernels,
# and either missing means DoG. Measured costs, extrapolated to a train-sized
# 199-video hidden test set: DoG 0.32 h, U-Net on CPU 16.5 h. The density switch
# only puts 55% of samples on the U-Net, so CPU+switch is ~9.2 h — inside the
# 12 h cap on paper, but that figure came off this box's 8 threads and Kaggle
# has a single-digit vCPU count, so it is not a margin worth betting a run on.
# Degrade to the 20-minute path that always finishes instead.
USE_UNET = HAVE_WEIGHTS and gpu_is_usable()
print(f"weights {'yes' if HAVE_WEIGHTS else 'NO'} ({WEIGHTS}) · "
      f"detector {'unet + density switch' if USE_UNET else 'DoG fallback'}",
      flush=True)
if not HAVE_WEIGHTS:
    # Not an assert: DoG really is a valid submission and a torch/weights
    # failure should still produce one. But list what is actually mounted, so
    # the log says *why* rather than leaving the next person to guess.
    print("  mounts:", {q.name: sorted(r.name for r in q.iterdir())[:8]
                        for q in ROOT.iterdir()
                        if q.is_dir() and q.name != "competitions"}, flush=True)
assert int(zarr.__version__.split(".")[0]) >= 3, "need zarr 3.x for Zarr v3 stores"

import submit  # noqa: E402

# Defaults are the best configuration measured to date. gate_um is a
# per-detector knob — do not carry it to a different detector without
# re-sweeping.
#
# switch_at is the largest single gain in the project: DoG and the U-Net tie on
# aggregate (0.6753 / 0.6789 pooled over all 199 samples) but disagree per
# sample, and the disagreement is predictable — spearman(cells/frame,
# unet - dog) = +0.700. Switching at 150 predicted cells/frame scores 0.7366
# against 0.6789 for the better single detector, 91% of the per-sample oracle's
# 0.7433. It survives the predicted density it has to run on: simulating the
# p90-quantile predictor's real 25% error over 400 draws, the *worst* draw still
# scores 0.7297.
#
# It also makes the run cheaper rather than dearer, which is not obvious: the
# U-Net runs only on the crowded samples, so 55% of a train-like test set takes
# the 1.86 s/frame path and 45% takes DoG's 0.06 s/frame. Caveat for the
# leaderboard: the 0.7366 was measured with each fold scored by the model that
# held it out (A on 6bba, D on 44b6), and a submission has one model for one
# unseen embryo. Expect less.
#
# None when the weights are missing, not 150: run() refuses a switch without a
# U-Net rather than silently degrading to plain DoG, and that guard would turn
# the DoG fallback into a crash.
submit.run(
    TEST, OUT,
    weights=WEIGHTS if USE_UNET else None,
    budget=1.0,
    gate_um=8.0,
    division_gate_um=None,
    switch_at=150.0 if USE_UNET else None,
)
