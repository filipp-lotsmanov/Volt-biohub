"""Gate: does the inference path import **and run** with the un-bundleable packages absent?

Run this before building the code dataset. It is the only mechanical check that
the notebook will not die four hours in on a ``ModuleNotFoundError``, and it is
cheap enough to run on every push.

``tracksdata`` is the package that matters. It is a git dependency, not on PyPI,
and its 20 transitive requirements include ``ilpy`` (needs SCIP), a Rust
extension (``rstar-python``) and ``numba`` — so bundling it is a project, not a
wheel. CLAUDE.md's rule is therefore to keep it out of the inference path and use
it only locally for ground-truth parsing and scoring, which costs nothing because
``.geff`` is plain Zarr v3.

The other blocked names are packages the repo declares but which the *inference*
path has no business needing. Blocking them keeps the offline wheel list honest
in one direction only: it proves nothing *forbidden* is reachable. It cannot
prove the wheel list is **complete**, because the gate runs inside the dev venv,
where every third-party package is installed — an import of something that is
present here but missing from ``requirements-offline.txt`` (``tqdm``, say) passes
this gate and then fails on Kaggle. Measured, not theorised: adding a module-scope
``import tqdm`` to ``submit.py`` leaves this gate at exit 0.

Completeness is only establishable in a clean environment, so verify it there
before an upload that changes the offline path's imports::

    uv venv --python 3.11 /tmp/offline_env
    PY=/tmp/offline_env/bin/python
    uv pip install --python $PY --no-deps -r kaggle/requirements-offline.txt
    uv pip install --python $PY numpy scipy torch   # the base image's, not bundled
    PYTHONPATH=src:kaggle $PY kaggle/submit.py --test-dir <dir> --out /tmp/s/submission.csv

That venv contains no tracksdata at all, which is also the only check that does
not depend on this file's blocker being correct.

**Two design choices, both of which were wrong in the first version and made the
gate capable of passing on a tree that could not import offline at all.**

*It runs in a child process.* The blocking used to be done in-process by swapping
``builtins.__import__`` and purging ``sys.modules``. That is a workaround for a
problem a subprocess does not have: any module imported before the gate ran was
returned from cache without re-executing its imports, so pre-importing
``volt_biohub.linking`` was enough to make the gate report the whole path clean.
A fresh interpreter has nothing cached and needs no purge.

*It blocks via ``sys.meta_path``, not ``builtins.__import__``.*
``importlib.import_module`` and ``importlib.util.find_spec`` do not call
``builtins.__import__``, so a lazy ``importlib.import_module(...)`` — exactly the
shape a well-meaning "lazy import" fix takes — walks straight past a patched
``__import__``. Measured with a one-line stub package: the old hook let
``importlib.import_module`` through and returned the module, while the meta-path
finder blocks all three of ``import X``, ``importlib.import_module`` and
``importlib.util.find_spec``. The old hook did happen to stop *tracksdata*, but
only indirectly, because tracksdata's own ``__init__`` re-imports its
subpackages through ``__import__``; that is an accident of one package's
internals, not a property of the gate. A meta-path finder sits below every
import mechanism there is.

*And it runs the path, not just imports it.* An import gate cannot see a
tracksdata call inside a function body, which is the whole point of a lazy
import. So after the imports the gate synthesises two image-only test datasets
and drives ``submit.run`` end to end, asserting that a validated
``submission.csv`` comes out. That exercises detect -> link -> write rows ->
validate -> rename, which is every line the notebook executes except the U-Net
forward pass (the DoG detector is used so the gate needs no weights and no GPU).
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

HARD_BLOCK = ("tracksdata",)
"""Not installable offline at any effort worth spending. A failure here is fatal."""

SOFT_BLOCK = ("geff", "dask", "pandas", "matplotlib", "kagglehub", "dotenv", "kaggle")
"""Installable, but the inference path should not want them.

``geff`` and ``dask`` are only for reading ground truth and whole-video lazy
loads; ``pandas``/``matplotlib`` are EDA; ``kagglehub``/``dotenv`` are used by
nothing.  A hit here is not fatal but it is a wheel we would otherwise have to
bundle for no reason.

``polars`` is deliberately *not* blocked. ``volt_biohub.submission`` — the
authoritative submission validator, which the offline path calls on every run —
needs it, so it is on the bundle list regardless of anything else.
"""

MODULES = (
    "volt_biohub.detectors",
    "volt_biohub.linking",
    "volt_biohub.models",
    "volt_biohub.gpu",
    "volt_biohub.submission",
    "submission",
    "submit",
)
"""The import closure of the offline path, most fundamental first.

``submit`` last, because it pulls the rest in and its failure is uninformative
until the individual modules have been placed.
"""

REPO = Path(__file__).resolve().parent.parent

# Two frames is the minimum that produces a linked edge; (1, 4, 4) downsampling
# turns (8, 32, 32) into the 8**3 isotropic grid the real pipeline runs at 64**3.
_SHAPE = (3, 8, 32, 32)
_SCALE = (1.0, 1.625, 0.40625, 0.40625)


class _Blocker:
    """A ``sys.meta_path`` finder that makes the named top-level packages missing.

    Raising from ``find_spec`` rather than returning ``None`` is deliberate: a
    ``None`` would let a later finder resolve the module, and the point is that
    nothing may. The message names this file so a failure is traceable to the
    gate rather than mistaken for a broken environment.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = frozenset(names)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in self.names:
            raise ModuleNotFoundError(
                f"No module named {fullname!r} — blocked by kaggle/check_offline.py "
                "because it cannot be installed in the offline notebook",
                name=top,
            )
        return None


def _write_zarr(path: Path) -> None:
    """A minimal but structurally faithful competition sample, images only.

    Faithful in the three ways the offline path depends on: the group attrs carry
    ``multiscales`` (voxel scale) and ``image_statistics.quantiles`` — including
    ``0.9``, which is what the node-budget predictor reads — and there is no
    ``.geff`` beside it, because the hidden test set ships images alone.
    """
    import numpy as np
    import zarr

    rng = np.random.default_rng(0)
    vol = rng.integers(20, 60, size=_SHAPE, dtype=np.uint16)
    # A few bright blobs, drifting one voxel per frame, so there is something to
    # detect and something to link.
    for t in range(_SHAPE[0]):
        for z, y, x in [(2, 8, 8), (4, 20, 20), (6, 12, 26)]:
            vol[t, z, y - 1 + t: y + 2 + t, x - 1: x + 2] = 2000

    group = zarr.open_group(str(path), mode="w")
    group.create_array("0", data=vol, chunks=(1, *_SHAPE[1:]))
    group.attrs.update({
        "multiscales": [{
            "datasets": [{"coordinateTransformations": [
                {"type": "scale", "scale": list(_SCALE)},
            ]}],
        }],
        "image_statistics": {"quantiles": {"0.001": 20.0, "0.9": 100.0, "0.999": 2000.0}},
    })


def _end_to_end(work: Path) -> None:
    """Drive ``submit.run`` on synthetic data and assert a valid submission.csv.

    Raises on anything wrong. The DoG detector is used (no ``weights``) so this
    needs neither a checkpoint nor a GPU; the U-Net branch differs only in where
    the detection map comes from, and its extra dependency — torch — is imported
    by ``volt_biohub.models`` during the import phase above.
    """
    import submit

    from volt_biohub.submission import SUBMISSION_FILENAME, validate_submission

    test_dir = work / "test"
    test_dir.mkdir(parents=True)
    for name in ("emb_bbbb", "emb_aaaa"):  # written out of order on purpose
        _write_zarr(test_dir / f"{name}.zarr")

    out = work / "out" / SUBMISSION_FILENAME
    submit.run(test_dir, out)

    if not out.exists():
        raise AssertionError(f"submit.run returned without writing {out}")
    # submit.run already validates before renaming into place, so this is the
    # cheap independent confirmation that what landed is what was validated.
    report = validate_submission(out, test_dir)
    if not report.ok:
        raise AssertionError(f"the submission it wrote does not validate:\n{report}")
    print(f"  {report}".replace("\n", "\n  "))


def gate(strict: bool = True) -> int:
    """Block, import every module of the offline path, then run it end to end.

    *strict* also blocks :data:`SOFT_BLOCK`. Returns the number of hard failures.
    Meant to run in a fresh interpreter — see the module docstring.
    """
    # Mirrors notebook.py's sys.path setup, so the gate resolves modules the same
    # way the notebook will rather than relying on the editable install.
    sys.path[:0] = [str(REPO / "src"), str(Path(__file__).parent)]

    names = HARD_BLOCK + (SOFT_BLOCK if strict else ())
    print(f"blocking: {', '.join(names)}\n")
    sys.meta_path.insert(0, _Blocker(names))

    hard_failures = 0
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            print(f"  ok    {mod}")
        except ModuleNotFoundError as exc:
            missing = exc.name or "?"
            fatal = missing in HARD_BLOCK
            hard_failures += fatal
            print(f"  {'FAIL ' if fatal else 'warn '} {mod}: needs {missing}")
            if fatal:
                print(f"        {exc.args[0] if exc.args else ''}"[:400])
        except Exception as exc:
            hard_failures += 1
            print(f"  FAIL  {mod}: {type(exc).__name__}: {exc}")

    if hard_failures:
        print(f"\n{hard_failures} module(s) cannot be imported offline. "
              "Do NOT build the code dataset until this is zero.")
        return hard_failures

    print("\nrunning the path end to end on synthetic image-only datasets:")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _end_to_end(Path(tmp))
        except Exception as exc:
            # A lazy tracksdata import inside a function body is invisible to the
            # import phase and lands here, which is exactly why this phase exists.
            import traceback
            traceback.print_exc()
            print(f"\nFAIL  the offline path imports but does not run: "
                  f"{type(exc).__name__}: {exc}")
            return 1

    print("\noffline path imports AND produces a valid submission.csv.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--no-strict", action="store_true",
                   help="Block only tracksdata, not the EDA-only packages.")
    p.add_argument("--in-process", action="store_true",
                   help=argparse.SUPPRESS)  # internal: the child does the work
    args = p.parse_args(argv)

    if args.in_process:
        return 1 if gate(strict=not args.no_strict) else 0

    # Re-exec in a child so nothing this interpreter has already imported can
    # satisfy an import the gate is trying to forbid. Inherits stdout/stderr, so
    # the output is identical to running it directly.
    cmd = [sys.executable, str(Path(__file__).resolve()), "--in-process"]
    if args.no_strict:
        cmd.append("--no-strict")
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
