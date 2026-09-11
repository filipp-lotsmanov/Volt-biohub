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

WHEELS = Path("/kaggle/input/volt-wheels")
CODE = Path("/kaggle/input/volt-code")
WEIGHTS = Path("/kaggle/input/volt-weights/split_0/edge_predictor_best.pth")
# The U-Net path needs the checkpoint AND its config.json — architecture is not
# inferable from a state dict, and submit.load_detector exits if the config is
# missing. Checking both here is what makes the DoG fallback real: guarding on
# the .pth alone turns a half-built weights dataset into no submission at all.
HAVE_WEIGHTS = WEIGHTS.exists() and (WEIGHTS.parent / "config.json").exists()
OUT = Path("/kaggle/working/submission.csv")

# The test directory is discovered, never hardcoded and never read out of
# sample_submission.csv: that file names the four local placeholder datasets and
# they are replaced at rerun, so reading names from it submits an empty file.
TEST = next(iter(sorted(Path("/kaggle/input").glob("*/test"))))

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={WHEELS}",
     "-r", str(CODE / "kaggle" / "requirements-offline.txt")],
    check=True,
)

sys.path[:0] = [str(CODE / "src"), str(CODE / "kaggle")]

import torch  # noqa: E402
import zarr  # noqa: E402

# Fail in minute one rather than hour eleven. zarr 2.x cannot open a Zarr v3
# store at all, and no-GPU with the U-Net detector cannot finish inside 12 h.
print(f"zarr {zarr.__version__} · torch {torch.__version__} · "
      f"cuda {torch.cuda.is_available()} · test dir {TEST}", flush=True)
# `kaggle datasets create -r zip` zips subdirectories, so whether the mounted
# layout matches these paths is not something we can verify from the dev box.
# Print it: a reshaped volt-weights would otherwise fall back to DoG silently and
# quietly cost the 0.7366 -> 0.6753 gap (the switch needs the U-Net to switch to).
print(f"weights {'yes' if HAVE_WEIGHTS else 'NO -> DoG fallback'} ({WEIGHTS})",
      flush=True)
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
    weights=WEIGHTS if HAVE_WEIGHTS else None,
    budget=1.0,
    gate_um=8.0,
    division_gate_um=None,
    switch_at=150.0 if HAVE_WEIGHTS else None,
)
