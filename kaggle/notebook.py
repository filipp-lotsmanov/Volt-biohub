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
      f"detector {'unet + density switch' if USE_UNET else 'DoG fallback'}", flush=True)

# Best configuration measured on LOEO CV; see CLAUDE.md for what each is worth.
# switch_at must be None without a U-Net: run() refuses a switch it cannot make,
# so passing 150 unconditionally would turn the DoG fallback into a crash.
submit.run(
    TEST, Path("/kaggle/working/submission.csv"),
    weights=WEIGHTS if USE_UNET else None,
    budget=1.0,
    gate_um=8.0,
    switch_at=150.0 if USE_UNET else None,
)
