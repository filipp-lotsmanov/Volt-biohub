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
# None means one detector; the blend is a bonus, never a requirement, so a
# volt-weights version without secondary/ degrades to the 0.920 configuration
# rather than failing.
SECONDARY = kaggle_env.find_secondary_weights()
USE_UNET = WEIGHTS is not None and kaggle_env.gpu_is_usable()
print(f"test {TEST} · weights {WEIGHTS} · secondary {SECONDARY} · "
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
#
# THE CHANGE OVER THE 0.908 SUBMISSION IS THAT WE STOPPED REIMPLEMENTING THE
# INFERENCE PATH. `volt_biohub.reference_predict` is
# `scripts/predict_unet_transformer.py` copied verbatim -- itself byte-identical
# to the upstream baseline apart from docstrings -- with tracksdata removed.
# Everything after it is ours. Measured end to end through this same writer, on
# the four movies the public 0.942/0.948 notebooks validate on (their validator
# picks two held-out train movies per embryo prefix, division-preferring,
# alphabetical, and its scorer is term for term ours):
#
#   our own detect-and-link + post-processing     0.9013   adjJ 0.8888
#   reference path alone, no post-processing      0.9299   adjJ 0.9299  divJ 0.0000
#   reference path + ILP prune + our stack        0.9505   adjJ 0.9362  divJ 0.1429
#
# 0.9505 is above the 0.9417 those notebooks report for themselves, and the
# reason is the combination rather than either half: their ILP prices a fork at
# 1.2 and prunes every one of them, so it scores division Jaccard 0.0000, while
# our division pass scores 0.14 on a linker 0.04 worse. Neither side alone gets
# there.
#
# Why our rewrite lost 0.04, twice. This file's own history is the argument: on
# 2026-09-14 the same comparison read 0.129 and the conclusion recorded then was
# "stop maintaining the rewrite; ship the path that already works". We rebuilt
# the rewrite instead. Four attempts to close it from the inside -- an ILP over
# the resolved graph, real consecutive frame pairs for the trunk features, the
# reference's 3-voxel peak window, and all three at once -- moved it by at most
# 0.009, so the residual was never one identifiable bug.
#
# THE ILP RUNS, AND IT NEEDS NO SOLVER. Its earlier rejection here (0.8190, node
# recall 0.9419) was of a solver handed the RAW candidate graph, which must then
# choose the node set too; the reference resolves greedily to <=1 parent and <=2
# children first and lets it decide only what to keep. In its own code that is
# worth +0.047 of adjusted edge Jaccard, spent on precision -- edge false
# positives roughly halve while true positives move by single digits. After
# greedy resolution the graph is a forest, over which tracksdata's formulation
# is separable, so `ilp_link.prune_dp` returns the identical edge set by tree DP:
# 24,540 nodes against the solver's 24,542 on one full video, 42,186 against
# 42,184 on another, the differences being equal-objective ties. Milliseconds
# instead of ~15 s a video, and no tracksdata on the inference path.
#
# TTA IS ON, reversing a rejection this file argued at length. Two facts. The
# reference's det_tta is FLIP-xy, four views; the -0.015 we measured was the
# full eight-view D4 group, rotations included, which is a much stronger
# perturbation through an anisotropic PSF -- not the same operation. And every
# detector-side measurement we made used the public checkpoint, which trained on
# all 199 movies including our own test folds; on our honestly held-out detector
# the same eight-view TTA scores +0.0071 (0.7799 -> 0.7870) where it scored
# -0.015 there. The hidden test set is a third unseen embryo, which is the
# held-out regime, so the in-domain sign is the one not to trust.
#
# det_threshold REPLACES the node budget, which took us from 0.581 to 0.846:
# four submissions scored 0.588 / 0.578 / 0.581 / 0.575 across pipelines whose
# CV node recall spanned 0.65 to 0.98, and the one thing they shared was
# predict_n_true(q0.9). A threshold removes N_true from inference entirely.
#
# THE DUAL-SEED BLEND is the change over the 0.920 submission, and it is now
# inside the reference path (reference_predict.make_blend) rather than in
# load_detector, which that path bypasses. The public cluster blends two
# checkpoints; we had been running one of them alone, having never downloaded
# the other. Moment-matched in logit space, scale ratio clamped to [0.5, 2.0],
# with a per-frame guard that reverts to the primary if the blend loses more
# than 10% of its candidates -- because losing node recall is the one failure
# this pipeline cannot absorb.
#
# The weight is 0.15, not the 0.3 that went out as 56432361. That 0.3 was found
# under the OLD detect-and-link on fold 1; swept again on fold 0 under this
# linker the curve is unimodal with the peak at 0.15, and adjJ, divJ and node
# recall all peak there together:
#
#   w        0        0.10     0.15     0.20     0.30     0.45
#   score    0.9296   0.9347   0.9361   0.9345   0.9336   0.9310
#   adjJ     0.9236   0.9275   0.9279   0.9267   0.9259   0.9246
#
#   fold 0 (85% of weight)   0.9296 -> 0.9361   adjJ 0.9236 -> 0.9279
#                                                divJ 0.0597 -> 0.0816
#
# It gains on the edge term AND the division term while the node count barely
# moves, so it is not trading one against the other -- unlike the node-retention
# knobs, which gain 0.002 on fold 1 and nothing on fold 0 because the
# node-count multiplier sits on opposite sides of zero in the two embryos.
#
# And this is the one class of change our CV UNDERSTATES. The 0.920 submission
# transferred at 35% (CV +0.0346 -> LB +0.012) because 96% of that gain was a
# learned edge head trained on both our folds. A blend of two independently
# seeded detectors is variance reduction, which pays more on an unseen embryo
# than in domain -- the same reason 8-view TTA reads -0.015 on the contaminated
# checkpoint and +0.0071 on our honestly held-out one.
#
# NOT included: the appearance fork ranker. It is the only fork feature this
# project has found that separates (AUC 0.632 over all 151 GT divisions against
# 1,729 controls -- a dividing parent DIMS, median two-frame intensity ratio
# 0.901 against 1.002) and worth +0.0008 alone, but it is subsumed by the blend:
# 0.9336 against 0.9337 together, because both act on the division term and the
# blend has already taken it.
submit.run(
    TEST, Path("/kaggle/working/submission.csv"),
    weights=WEIGHTS if USE_UNET else None,
    secondary_weights=SECONDARY if USE_UNET else None,
    secondary_weight=0.15 if (USE_UNET and SECONDARY) else 0.0,
    budget=1.0,
    det_threshold=0.965 if USE_UNET else None,
    reference=USE_UNET,
    ilp_prune=USE_UNET,
    tta=USE_UNET,
    postprocess=True,
    min_track_len=6,
)
