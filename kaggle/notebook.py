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

# Best configuration measured through the submission path ITSELF, not through a
# parallel harness -- that mismatch is what produced CV 0.7366 against LB 0.588.
# Weighted over both LOEO folds by metric weight: U-Net 0.684, DoG 0.663.
#
# No density switch. It was worth +0.058 measured with an ORACLE node budget and a
# different model per fold, and neither holds at inference: the budget comes from a
# predictor that lands 0.50x-1.54x of truth, and it routed 1 of the 4 local samples
# to the wrong detector. Re-measured honestly the U-Net wins on BOTH folds --
# 0.6923 vs 0.6856 on 6bba and 0.6348 vs 0.5348 on 44b6 -- so there is nothing left
# to switch to.
#
# The division gate is left at its library default (6.0 um) rather than forced to
# None. Forcing it off is what made the first submission emit 0 forks in 73,130
# edges and score exactly 0 on a term worth 0.1.
submit.run(
    TEST, Path("/kaggle/working/submission.csv"),
    weights=WEIGHTS if USE_UNET else None,
    budget=1.0,
    gate_um=8.0,
)
