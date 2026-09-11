"""Locate the pieces of a Kaggle run: test data, weights, a usable GPU.

Every path is discovered rather than hardcoded. That is not defensive habit —
all three mount paths this repo originally hardcoded turned out to be wrong,
each in a different way, and each cost a push-run-read cycle to find. The
measurements and the reasoning are in ``kaggle/README.md`` under *Probing the
image*; this module is just the result.

Kept out of ``submit.py`` because it is Kaggle-environment code rather than
pipeline code, and kept out of ``notebook.py`` because anything in the notebook
runs only on Kaggle, where it cannot be tested. ``tests/test_kaggle_env.py``
covers it.
"""

from __future__ import annotations

from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

COMPETITION_MOUNT = "competitions"
"""Kaggle mounts the competition at ``competitions/<slug>/``, not ``<slug>/``.

Searches skip it: it holds ~90,000 zarr chunk files, so an unbounded ``rglob``
over ``/kaggle/input`` spends minutes walking it to reach a 12-file wheels
directory.
"""


def dataset_mounts(root: Path = INPUT_ROOT) -> list[Path]:
    """Attached Dataset mounts, excluding the competition's own."""
    if not root.exists():
        return []
    return [p for p in sorted(root.iterdir())
            if p.is_dir() and p.name != COMPETITION_MOUNT]


def find_in_mounts(pattern: str, root: Path = INPUT_ROOT,
                   max_depth: int = 4) -> Path | None:
    """First match for *pattern* across the dataset mounts, bounded depth.

    Bounded because ``datasets create -r zip`` uploads each top-level
    subdirectory as ``<name>.zip`` and Kaggle's extractor is inconsistent about
    whether the directory level survives: it kept ``kaggle/`` and ``src/`` for a
    two-archive dataset and flattened ``split_0/`` for a one-archive one.
    """
    for mount in dataset_mounts(root):
        for depth in range(max_depth):
            hits = sorted(mount.glob("/".join(["*"] * depth + [pattern])))
            if hits:
                return hits[0]
    return None


def find_test_dir(root: Path = INPUT_ROOT) -> Path:
    """The competition's ``test/``, verified to actually contain ``*.zarr``.

    Never read the dataset list out of ``sample_submission.csv``: it names the
    four local placeholder datasets and they are replaced at rerun, so a
    submission built from it would be empty.

    Requiring ``*.zarr`` rather than taking the first ``test`` directory means a
    future mount change fails loudly here, in the opening seconds, instead of
    producing an empty submission at the end of the run.
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


def find_weights(root: Path = INPUT_ROOT) -> Path | None:
    """The checkpoint, beside its ``config.json``. ``None`` means run DoG.

    The sibling config is required, not optional: architecture is not inferable
    from a state dict and :func:`submit.load_detector` exits without it, so a
    lone ``.pth`` must read as "no weights" rather than as a checkpoint that
    kills a long run at minute one.
    """
    for mount in dataset_mounts(root):
        for depth in range(4):
            for pth in sorted(mount.glob(
                    "/".join(["*"] * depth + ["edge_predictor_best.pth"]))):
                if (pth.parent / "config.json").exists():
                    return pth
    return None


def gpu_is_usable() -> bool:
    """Does a CUDA kernel actually *run* here — not just ``is_available()``.

    ``enable_gpu`` can hand out a Tesla P100 (``sm_60``) while the image's torch
    ships cubins for ``sm_70`` upward. ``torch.cuda.is_available()`` returns
    True on that machine and every convolution then dies with "no kernel image
    is available for execution on the device", so an availability check is not
    merely insufficient here — it is confidently wrong.

    ``machine_shape: NvidiaTeslaT4`` asks for ``sm_75`` instead, but Kaggle
    substitutes hardware when a shape is unavailable, so the request is not a
    guarantee and this check stays. Execute, never infer.
    """
    try:
        import torch
    except ImportError:
        return False
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
