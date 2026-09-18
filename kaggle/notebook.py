"""Kaggle submission kernel. Glue only — every decision lives in the repo.

Three Datasets are attached (see kernel-metadata.json): volt-wheels (offline
wheels), volt-code (this repo at a known commit), volt-weights (checkpoint +
config.json). Internet off, accelerator T4.

Why the logic is not here: this file only ever runs on Kaggle, so nothing in it
can be tested or linted in CI. Anything with a decision in it belongs in
volt-code, which is the repo. The bootstrap below is the irreducible exception —
it has to find the repo before it can import from it.

Background, measurements and the submission procedure: kaggle/README.md.
"""

import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")


def _find(pattern, max_depth=4):
    """Bootstrap-only search. kaggle_env.find_in_mounts once that is importable.

    Mount layouts are discovered, never hardcoded: every path this repo once
    hardcoded was wrong. The competition mount is skipped — ~90k zarr chunks.
    """
    for mount in sorted(p for p in INPUT.iterdir() if p.is_dir()):
        if mount.name == "competitions":
            continue
        for depth in range(max_depth):
            hits = sorted(mount.glob("/".join(["*"] * depth + [pattern])))
            if hits:
                return hits[0]
    return None


REQUIREMENTS = _find("requirements-offline.txt")
_pkg = _find("volt_biohub/metrics.py")
_whl = _find("*.whl")
if not (REQUIREMENTS and _pkg and _whl):
    raise SystemExit(
        f"missing inputs — requirements={REQUIREMENTS} src={_pkg} wheels={_whl}. "
        f"mounts: {sorted(p.name for p in INPUT.iterdir())}. Attach volt-code and "
        "volt-wheels, and check `kaggle datasets files` shows the version you "
        "expect: a version still processing mounts as the previous one."
    )

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index",
     f"--find-links={_whl.parent}", "-r", str(REQUIREMENTS)],
    check=True,
)
sys.path[:0] = [str(_pkg.parent.parent), str(REQUIREMENTS.parent)]

import kaggle_env  # noqa: E402
import submit  # noqa: E402
import zarr  # noqa: E402

# zarr 2.x cannot open a Zarr v3 store at all. Fail in minute one, not hour ten.
assert int(zarr.__version__.split(".")[0]) >= 3, "need zarr 3.x for Zarr v3 stores"

TEST = kaggle_env.find_test_dir()
WEIGHTS = kaggle_env.find_weights()
USE_UNET = WEIGHTS is not None and kaggle_env.gpu_is_usable()
print(f"test {TEST} · weights {WEIGHTS} · "
      f"detector {'unet' if USE_UNET else 'DoG fallback'}", flush=True)
if WEIGHTS is not None:
    # Say WHICH checkpoint this is. Every checkpoint of this architecture is the
    # same 8,357,783 bytes, so a size check cannot tell two runs apart and a stale
    # volt-weights version would score a different model with no visible sign.
    # config.json carries the run that produced it.
    import json as _json

    _c = _json.loads((WEIGHTS.parent / "config.json").read_text())
    print(f"  checkpoint: method={_c.get('method')} fold={_c.get('fold')} "
          f"lr={_c.get('lr')} seed={_c.get('seed')} "
          f"det_neg_weight={_c.get('det_neg_weight')}", flush=True)

# Best configuration measured through the submission path ITSELF, not through a
# parallel harness -- that mismatch is what produced CV 0.7366 against LB 0.588.
# Every number below is the full 71-sample LOEO fold, one variable at a time,
# with this same checkpoint (the public 400-epoch pack, which is legal: the rules
# explicitly allow freely available pre-trained models).
#
#   threshold 0.5, one-pass gate 8 um      0.7830   divJ 0.0110   <- scored 0.846
#   threshold 0.965                        0.7931   divJ 0.0110
#   + full post-processing                 0.8687   divJ 0.1077
#   8-view D4 TTA alone                    0.7916   divJ 0.0088
#   learned transformer edge head          0.7632   divJ 0.0178
#   post-processing at threshold 0.93      0.8649   divJ 0.0938
#   post-processing + TTA at 0.965         0.8609   divJ 0.0484
#   post-processing + TTA at 0.93          0.8593   divJ 0.0492
#
# THE POST-PROCESSING IS THE CHANGE. It is the shared post-processing cell of the
# public 0.942-0.948 notebooks, ported in their order (which is load-bearing) --
# two-pass Hungarian with velocity extrapolation, single-frame gap closing by
# inserting an interpolated node, structurally gated divisions, a short-track
# filter and line-fit smoothing. The detector was never the gap: with this exact
# checkpoint we scored 0.846 against a field median of 0.902, so everything after
# detection was.
#
# The division term is the part worth naming. divJ has sat near zero since the
# first submission, against a term worth 0.1 of the score. Our linker gated forks
# at 6.0 um while measured GT sister separation one frame after a division has a
# median of 10.4 um -- most real divisions were rejected by construction. The
# reference opens the gates (parent 9.0, sister 14.0) and defends precision
# structurally instead: mutual-nearest orphans, forward divergence, a symmetry
# test, and hard per-frame and global caps.
#
# NOT included, both on measurement rather than taste. The ILP: those notebooks
# run it and then discard its edge set wholesale, keeping only per-pair
# probabilities as a small additive bonus, so their shipped edges come from the
# two-pass Hungarian above. The learned edge head: 0.7632 against Hungarian's
# 0.7830, because abstaining leaves isolated nodes that pruning then deletes.
#
# gate_um is deliberately absent: --postprocess REPLACES the one-pass fixed-gate
# linker entirely, so passing it would be tuning a dead knob.
#
# det_threshold REPLACES the node budget, which is what took the last submission
# from 0.581 to 0.846. Four submissions scored 0.588 / 0.578 / 0.581 / 0.575
# across four structurally different pipelines whose CV node recall spanned 0.65
# to 0.98; the one thing all four shared was predict_n_true(q0.9), so every one
# emitted the same wrong node count on the hidden embryo no matter what sat
# underneath it. A threshold removes N_true from the inference path entirely and
# the detector's own confidence estimates the count better than the fit did.
#
# NO TTA, and the reason is specific rather than "it did not help". Every public
# notebook above 0.942 uses it, so a negative result needs a mechanism. Averaging
# logits over the D4 symmetries of (y, x) acts like a blur in that plane, and the
# two daughters one frame after a division are close enough to merge under it --
# so the division term halves, on both arms tested:
#
#   post-processing                        0.8687   divJ 0.1077
#   + TTA, threshold 0.965                 0.8609   divJ 0.0484
#   + TTA, threshold 0.93                  0.8593   divJ 0.0492
#
# Retuning the threshold underneath it does not recover it, which rules out "the
# operating point moved" as the whole story. It is also not free: the detector
# measured 13.3 s/video on the real T4 kernel, so eight views is ~5.9 h before
# post-processing against ~1.6 h without. Available as tta=True, and worth
# revisiting only for a detector whose daughters do not merge -- a finer grid
# (--downsample 1,2,2) is the obvious candidate.
#
# The threshold retune without TTA also lost: 0.93 gives 0.8649 against 0.965's
# 0.8687, so 0.965 is a real interior optimum and not inherited from the public
# notebooks by assumption.
submit.run(
    TEST, Path("/kaggle/working/submission.csv"),
    weights=WEIGHTS if USE_UNET else None,
    budget=1.0,
    det_threshold=0.965 if USE_UNET else None,
    postprocess=True,
    min_track_len=6,
)
