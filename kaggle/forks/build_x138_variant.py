#!/usr/bin/env python
"""Build a private variant of the public x138 notebook with our changes on top.

The frontier public notebook (anvithpothula/biohub-x138, public LB ~0.953) is
someone else's code, so it is not committed here. This script is the
reproducible part: it takes the pulled notebook, refuses anything but the exact
snapshot the patches were written against, applies named patches as plain
string replacements, and writes a kernel directory ready for
``kaggle kernels push``.

    uv run kaggle kernels pull anvithpothula/biohub-x138 -p /tmp/x138 -m
    uv run python kaggle/forks/build_x138_variant.py /tmp/x138/biohub-x138.ipynb \
        /tmp/k_variant --slug volt-x138-soup2sec --secondary-soup pubfull_soup2.pth

Every patch fails loudly when its anchor is missing or ambiguous. A patch that
silently does not apply produces the unmodified notebook, which scores exactly
the public number and looks like a null result -- the failure this repo has
already paid for with a missing ``secondary/`` directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

X138_SHA256 = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"
SOUPS_DATASET = "oleksiikrasnoshtanov/volt-x138-soups"
# md5 of every file the soups dataset carries. The notebook re-checks the one it
# loads: all soups of this architecture are 8,355,927 bytes, so a size cannot
# tell them apart (kaggle/README.md, 2026-09-22).
SOUP_MD5 = {
    "pubfull_soup2.pth": "3af2701dea2d949b9e9f5a09ac2aff02",  # pubfull 381 + 400
    "pubfull_soup4.pth": "467180abb65639615ebb3f787717c87d",  # + two SWA epochs
    "pack50_soup2.pth": "97e5d6eb6a6cd72416d122a7f7351ed8",   # pack50 best + 402
    "pack50_soup3.pth": "7fcade11c29ebcf7ffc6981e8ad656ca",   # + epoch 350
}

BASE_METADATA = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
        "anvithpothula/biohub-v1284-head-s075",
    ],
    "competition_sources": ["biohub-cell-tracking-during-development"],
    "kernel_sources": [],
    "model_sources": [],
    # The image x138 itself pins; torch/numpy drift is a real risk for a
    # notebook that patches another script by string replacement.
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:"
                    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "docker_image_pinning_type": "original",
    "machine_shape": "NvidiaTeslaT4",
}


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"patch {what!r}: anchor found {n} times, expected exactly 1")
    return src.replace(old, new)


def _soup_loader(var: str, name: str) -> str:
    """Code that finds one soup on the mount, checks its md5 and binds *var*."""
    md5 = SOUP_MD5[name]
    owner, slug = SOUPS_DATASET.split("/")
    return (
        f"import hashlib as _volt_hashlib\n"
        f"_volt_candidates = [Path('/kaggle/input/{slug}/{name}'),\n"
        f"                    Path('/kaggle/input/datasets/{owner}/{slug}/{name}')]\n"
        f"{var} = next((p for p in _volt_candidates if p.is_file()), None)\n"
        f"if {var} is None:\n"
        f"    raise FileNotFoundError(('volt soup not mounted', [str(p) for p in _volt_candidates]))\n"
        f"if _volt_hashlib.md5({var}.read_bytes()).hexdigest() != '{md5}':\n"
        f"    raise RuntimeError('volt soup {name} md5 mismatch')\n"
        f"if not ({var}.parent / 'config.json').is_file():\n"
        f"    raise FileNotFoundError('config.json must sit beside the soup')\n"
    )


def patch_secondary_soup(src: str, name: str) -> str:
    """Swap the secondary (pubfull, 80 % of the detection map) for a soup.

    The pinned-hash checks above the anchor still run against the original
    file, so the stock integrity guarantees hold; only what is *loaded* changes,
    and the receipt is rewritten to describe it rather than the original.
    """
    anchor = 'os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(SECONDARY_WEIGHTS_PATH)\n'
    new = anchor + (
        "# ---- volt patch: secondary = same-run weight soup ----\n"
        + _soup_loader("_volt_soup", name)
        + "SECONDARY_WEIGHTS_PATH = _volt_soup\n"
        "_secondary_actual_sha256 = _sha256_file(_volt_soup)\n"
        'os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(_volt_soup)\n'
        f"print('VOLT secondary soup: {name}', _volt_soup, flush=True)\n"
    )
    return _replace_once(src, anchor, new, "secondary-soup")


def patch_env(src: str, key: str, value: str) -> str:
    """Override one of cell 0's ``os.environ[...] = ...`` settings."""
    import re
    pat = re.compile(r'^os\.environ\[["\']' + re.escape(key) + r'["\']\]\s*=\s*.*$', re.M)
    hits = pat.findall(src)
    if len(hits) != 1:
        raise SystemExit(f"--set {key}: {len(hits)} assignments in the cell, expected 1")
    return pat.sub(f'os.environ["{key}"] = "{value}"  # volt patch', src)


def patch_v1284_head(src: str, name: str, md5: str) -> str:
    """Load the coordinate head from our dataset instead of the author's.

    The architecture is fixed by the notebook's own module (224 -> 32 -> 3), so
    only the weights and the input normalisation change. Replacing the rglob
    also drops a full walk of /kaggle/input, which the notebook's own comments
    price at minutes on the graded rerun.
    """
    anchor = "_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt'))\n"
    owner, slug = SOUPS_DATASET.split("/")
    new = (
        "# ---- volt patch: our coordinate head ----\n"
        "import hashlib as _volt_hashlib\n"
        f"_volt_hc = [Path('/kaggle/input/{slug}/{name}'),\n"
        f"            Path('/kaggle/input/datasets/{owner}/{slug}/{name}')]\n"
        "_volt_head = next((p for p in _volt_hc if p.is_file()), None)\n"
        "if _volt_head is None:\n"
        "    raise FileNotFoundError(('volt head not mounted', [str(p) for p in _volt_hc]))\n"
        f"if _volt_hashlib.md5(_volt_head.read_bytes()).hexdigest() != '{md5}':\n"
        f"    raise RuntimeError('volt head {name} md5 mismatch')\n"
        "_myhead = [_volt_head]\n"
        f"print('VOLT coordinate head: {name}', _volt_head, flush=True)\n"
    )
    return _replace_once(src, anchor, new, "v1284-head")


def patch_add_env(src: str, key: str, value: str) -> str:
    """Add an ``os.environ`` setting cell 0 does not make (the notebook reads it
    later with a default, e.g. BIOHUB_OUTPUT_MOTION_RELINK). Refuses keys cell 0
    already assigns -- those go through --set, so there is one owner per key."""
    if f'os.environ["{key}"]' in src or f"os.environ['{key}']" in src:
        raise SystemExit(f"--add {key}: cell 0 already sets it; use --set")
    anchor = 'print("BIOHUB_PRESET:", BIOHUB_PRESET)\n'
    return _replace_once(src, anchor, f'os.environ["{key}"] = "{value}"  # volt patch\n' + anchor,
                         f"add-env {key}")


def patch_v1284_mode(src: str, mode: str) -> str:
    anchor = "os.environ['V1284_MODE']='candidate'\n"
    return _replace_once(src, anchor, f"os.environ['V1284_MODE']={mode!r}  # volt patch\n",
                         "v1284-mode")


def build(nb_path: Path, out_dir: Path, slug: str, *, secondary_soup: str | None,
          v1284_mode: str | None, env: dict[str, str] | None = None,
          v1284_head: tuple[str, str] | None = None,
          add_env: dict[str, str] | None = None) -> Path:
    raw = nb_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != X138_SHA256:
        raise SystemExit(f"{nb_path} sha256 {sha} is not the snapshot these patches target "
                         f"({X138_SHA256}); re-derive the anchors before building")
    nb = json.loads(raw)
    patches = []
    if secondary_soup:
        patches.append(('os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(SECONDARY_WEIGHTS_PATH)\n',
                        lambda s: patch_secondary_soup(s, secondary_soup)))
    if v1284_mode:
        patches.append(("os.environ['V1284_MODE']='candidate'\n",
                        lambda s: patch_v1284_mode(s, v1284_mode)))
    if v1284_head:
        patches.append(("_myhead = sorted(Path('/kaggle/input').rglob(",
                        lambda s: patch_v1284_head(s, *v1284_head)))
    for key, value in (env or {}).items():
        patches.append((f'os.environ["{key}"]', lambda s, k=key, v=value: patch_env(s, k, v)))
    for key, value in (add_env or {}).items():
        patches.append(('print("BIOHUB_PRESET:", BIOHUB_PRESET)\n',
                        lambda s, k=key, v=value: patch_add_env(s, k, v)))
    if not patches:
        raise SystemExit("no patch requested; that is the public notebook unchanged")
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for anchor, apply in patches:
        hits = [c for c in code_cells if anchor in "".join(c["source"])]
        if len(hits) != 1:
            raise SystemExit(f"anchor {anchor!r} is in {len(hits)} cells, expected 1")
        hits[0]["source"] = apply("".join(hits[0]["source"]))
    patched = len(patches)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{slug}.ipynb"
    (out_dir / code_file).write_text(json.dumps(nb, indent=1))
    meta = {"id": f"oleksiikrasnoshtanov/{slug}", "title": slug, "code_file": code_file,
            **BASE_METADATA}
    if secondary_soup or v1284_head:
        meta["dataset_sources"] = [*meta["dataset_sources"], SOUPS_DATASET]
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir / code_file} ({patched} cell(s) patched) and kernel-metadata.json")
    return out_dir / code_file


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook", type=Path, help="the pulled public x138 .ipynb")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--slug", required=True, help="kernel slug under our account")
    p.add_argument("--secondary-soup", choices=sorted(SOUP_MD5), default=None)
    p.add_argument("--v1284-mode", choices=("zero", "candidate"), default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one of cell 0's os.environ settings, e.g. "
                        "BIOHUB_MOTION_RELINK_LEARNED_BONUS=15.0")
    p.add_argument("--add", action="append", default=[], metavar="KEY=VALUE",
                   help="add an os.environ setting cell 0 does not make, e.g. "
                        "BIOHUB_OUTPUT_MOTION_RELINK=0")
    p.add_argument("--v1284-head", default=None, metavar="FILE=MD5",
                   help="a head file in the soups dataset, with its md5")
    a = p.parse_args(argv)
    env = dict(kv.split("=", 1) for kv in a.set)
    head = tuple(a.v1284_head.split("=", 1)) if a.v1284_head else None
    add = dict(kv.split("=", 1) for kv in a.add)
    build(a.notebook, a.out_dir, a.slug, secondary_soup=a.secondary_soup,
          v1284_mode=a.v1284_mode, env=env, v1284_head=head, add_env=add)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
