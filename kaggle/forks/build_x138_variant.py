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
import os
from pathlib import Path

X138_SHA256 = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"
# The public notebook went 404 on 2026-09-27. Our own build of it, `volt-x138-ilpdis15`
# (submitted as 56539713), differs from it by exactly one setting -- ILP disappearance cost
# 1.5 instead of 2.0, in cell 0 and in cell 1's configuration guard -- so it is the other
# accepted base, and the builder puts x138's 2.0 back before applying anything else.
ILPDIS15_SHA256 = "a9b1679cdd1dca77eeb78bfc50657c2f3efb7bc56319eee2281d2ba66f3453cf"
ILPDIS15_REVERT = [
    ('os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.5"  # volt patch\n',
     'os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "2.0"  # volt: x138 value restored\n'),
    ('    "BIOHUB_ILP_DISAPPEARANCE_WEIGHT": 1.5,\n',
     '    "BIOHUB_ILP_DISAPPEARANCE_WEIGHT": 2.0,\n'),
]
SOUPS_DATASET = "oleksiikrasnoshtanov/volt-x138-soups"
# The kernel is pushed under the account of whoever holds the token.
FALLBACK_OWNER = "oleksiikrasnoshtanov"
DEFAULT_OWNER = os.environ.get("KAGGLE_USERNAME", "").strip() or FALLBACK_OWNER
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


# x138's "8-view" D4 TTA takes rot90(x, 1).transpose(-1, -2) as its eighth view, which is
# flip-x again, so flip-x carries 2/8 of the average and the anti-transpose is missing. The
# same expression is repeated at every TTA site (primary and secondary detection, both
# feature averages, DeepCenter), partly inside patch strings with escaped newlines. The fix
# uses rot90(x.transpose(-1, -2), 2) for the view, which is the anti-transpose, an
# involution, so its inverse is the same expression. Nothing in the upstream script
# contains these rot90 terms, so rewriting every occurrence consistently keeps each patch's
# anchor matching the text the earlier patch wrote.
_D4_RULES = [
    # (pattern, replacement, expected count over the whole notebook)
    (r"(rot90)\(([\w\[\]]+), 1, dims=\(-2, -1\)\)\.transpose\(-1, -2\)",
     r"\1(\2.transpose(-1, -2), 2, dims=(-2, -1))", 4),
    (r"(rot90)\(\\n(\s*)imgs, 1, dims=\(-2, -1\)\\n(\s*)\)\.transpose\(-1, -2\)",
     r"\1(\\n\2imgs.transpose(-1, -2), 2, dims=(-2, -1)\\n\3)", 3),
    (r"(rot90)\(([\w\[\]\(\)]+)\.transpose\(-1, -2\), -1, dims=\(-2, -1\)\)",
     r"\1(\2.transpose(-1, -2), 2, dims=(-2, -1))", 5),
    (r"(secondary_det_at\[f\]\.transpose\(-1, -2\),\\n\s*)-1,",
     r"\g<1>2,", 3),
    (r"(_secondary_u_at\.transpose\(-1, -2\)), -1, dims=\(-2, -1\)",
     r"\1, 2, dims=(-2, -1)", 1),
]


def patch_d4(cells: list[dict]) -> None:
    import re
    for pat, rep, want in _D4_RULES:
        rx = re.compile(pat)
        got = sum(len(rx.findall("".join(c["source"]))) for c in cells)
        if got != want:
            raise SystemExit(f"--fix-d4: {pat!r} matched {got} times, expected {want}")
        for c in cells:
            c["source"] = rx.sub(rep, "".join(c["source"]))
    left = sum("".join(c["source"]).count(".transpose(-1, -2), -1, dims=(-2, -1))") for c in cells)
    if left:
        raise SystemExit(f"--fix-d4: {left} old inverse(s) survived")


# The division diagnosis. Every division number this repo has was measured on OUR
# pipeline; x138's division pass adds a DeepCenter veto we never had, which rejected
# 482 of 557 geometric candidates on the visible movies, and nobody knows what it
# throws away. This wraps `add_safe_divisions_postlink` -- it does not edit it -- and
# dumps, per movie: the graph the pass receives, every candidate pair with the value
# of EVERY gate (no short-circuit, so a refused pair still shows which other gates it
# would have failed), and the forks the pass actually added. The wrapper hands the
# original function its arguments untouched and returns its result untouched, and the
# DeepCenter scores are computed with caches of their own, so the submission is
# byte-identical to the unpatched build. `scripts/x138_division_diagnosis.py` reads
# the dumps. Diagnosis only: it costs a second DeepCenter pass, so never submit it.
_DIVDIAG_ANCHOR = "def filter_short_track_components("
_DIVDIAG_CODE = r'''# ---- volt patch: division diagnosis. READ-ONLY: the submission is unchanged. Do not submit. ----
import csv as _volt_csv
import inspect as _volt_inspect

_VOLT_DIVDIAG_DIR = os.environ.get("VOLT_DIVDIAG_DIR", __OUT_DIR__).strip()
# Wider than every shipped gate, so a refused true daughter is still logged with the value
# that refused it. The 2026-09-22 reject log saw true parent distances up to 11.83 um and
# sister distances up to 13.98 um.
_VOLT_DIVDIAG_PAR_UM = 14.0
_VOLT_DIVDIAG_SIS_UM = 18.0
_VOLT_DIVDIAG_EXIST_UM = 14.0
_VOLT_DIVDIAG_COLUMNS = [
    "t", "source_id", "child_id", "cand_id", "d_exist", "d_par", "d_sis", "nn_rank",
    "child_succ", "cand_succ", "div_margin", "symmetry", "dc_score", "rank_score",
    "pass_exist", "pass_par", "pass_sis", "pass_nn", "pass_div", "pass_dc", "pass_sym",
]
_volt_asd_original = add_safe_divisions_postlink
print(f"VOLT DIVISION DIAGNOSIS BUILD -> {_VOLT_DIVDIAG_DIR} (diagnosis only; do not submit)",
      flush=True)


def _volt_divdiag_dump(nodes_by_id, edges, dataset, deepcenter_bundle):
    out_dir = Path(_VOLT_DIVDIAG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(nodes_by_id)
    np.savez_compressed(
        out_dir / f"{dataset}_graph.npz",
        node_id=np.asarray(ids, dtype=np.int64),
        t=np.asarray([int(nodes_by_id[i]["t"]) for i in ids], dtype=np.int64),
        zyx=np.asarray([node_point(nodes_by_id[i]) for i in ids], dtype=np.float64).reshape(-1, 3),
        edge_src=np.asarray([int(e["source_id"]) for e in edges], dtype=np.int64),
        edge_tgt=np.asarray([int(e["target_id"]) for e in edges], dtype=np.int64),
    )
    children, has_parent, by_t = {}, set(), {}
    for e in edges:
        s, d = int(e["source_id"]), int(e["target_id"])
        children.setdefault(s, []).append(d)
        has_parent.add(d)
    for i in ids:
        by_t.setdefault(int(nodes_by_id[i]["t"]), []).append(i)

    # Caches of our own: the real pass never sees a frame or heatmap this loads.
    frame_cache, heat_cache, dc_memo = {}, {}, {}
    veto_on = bool(DEEPCENTER_SAFE_DIV_VETO and USE_DEEPCENTER_VETO)

    def dc_score(cid):
        if cid not in dc_memo:
            node = nodes_by_id[cid]
            s = deepcenter_score_point(dataset, int(node["t"]), node_point(node),
                                       deepcenter_bundle, frame_cache, heat_cache)
            dc_memo[cid] = float("nan") if s is None else float(s)
        return dc_memo[cid]

    rows = []
    for t in sorted(by_t):
        orphans = [i for i in by_t.get(t + 1, []) if i not in has_parent]
        if not orphans:
            continue
        orphan_um = np.stack([_position_um(nodes_by_id[i]) for i in orphans])
        tree = cKDTree(orphan_um)
        for s_id in by_t[t]:
            kids = children.get(s_id, [])
            if len(kids) != 1:
                continue
            c1 = kids[0]
            child = nodes_by_id.get(c1)
            if child is None or int(child["t"]) != t + 1:
                continue
            source = nodes_by_id[s_id]
            d_exist = edge_distance_um(source, child)
            if d_exist > _VOLT_DIVDIAG_EXIST_UM:
                continue
            near = tree.query_ball_point(_position_um(source), _VOLT_DIVDIAG_PAR_UM)
            if not near:
                continue
            # Rank of every orphan by distance to the existing child: rank 0 is the single
            # candidate the mutual-nearest rule proposes.
            to_child = np.linalg.norm(orphan_um - _position_um(child), axis=1)
            rank = np.empty(len(orphans), dtype=np.int64)
            rank[np.argsort(to_child, kind="stable")] = np.arange(len(orphans))
            c1_succ = children.get(c1, [])
            for k in near:
                q_id = orphans[k]
                cand = nodes_by_id[q_id]
                d_par = edge_distance_um(source, cand)
                d_sis = edge_distance_um(child, cand)
                if d_par > _VOLT_DIVDIAG_PAR_UM or d_sis > _VOLT_DIVDIAG_SIS_UM:
                    continue
                q_succ = children.get(q_id, [])
                margin, div_ok = float("nan"), False
                if len(c1_succ) == 1 and len(q_succ) == 1:
                    g1, g2 = nodes_by_id.get(c1_succ[0]), nodes_by_id.get(q_succ[0])
                    if (g1 is not None and g2 is not None
                            and int(g1["t"]) == t + 2 and int(g2["t"]) == t + 2):
                        margin = edge_distance_um(g1, g2) - d_sis
                        div_ok = margin >= SAFE_DIV_DIVERGE_UM
                sym = abs(d_exist - d_par) / max((d_exist + d_par) / 2.0, 1e-6)
                dc = dc_score(q_id) if veto_on else float("nan")
                rows.append([
                    t, s_id, c1, q_id, d_exist, d_par, d_sis, int(rank[k]),
                    len(c1_succ), len(q_succ), margin, sym, dc, d_par + 0.15 * d_sis,
                    int(d_exist <= SAFE_DIV_EXISTING_CHILD_MAX_UM),
                    int(d_par <= SAFE_DIV_MAX_UM),
                    int(d_sis <= SAFE_DIV_SISTER_MAX_UM),
                    int((not SAFE_DIV_REQUIRE_MUTUAL_NN) or rank[k] == 0),
                    int((not SAFE_DIV_REQUIRE_DIVERGENCE) or div_ok),
                    # A missing score is accepted, exactly as deepcenter_accept_repair_point does.
                    int((not veto_on) or not np.isfinite(dc) or dc >= DEEPCENTER_SAFE_DIV_THRESHOLD),
                    int(SAFE_DIV_SISTER_SYMMETRY_TAU <= 0.0 or sym <= SAFE_DIV_SISTER_SYMMETRY_TAU),
                ])
    with open(out_dir / f"{dataset}_cands.csv", "w", newline="") as fh:
        w = _volt_csv.writer(fh)
        w.writerow(_VOLT_DIVDIAG_COLUMNS)
        w.writerows(rows)
    print(f"  [{dataset}] VOLT division diagnosis: {len(ids)} nodes, {len(rows)} candidate pairs",
          flush=True)


def _volt_divdiag_dump_added(out_edges, dataset):
    added = [(int(e["source_id"]), int(e["target_id"])) for e in out_edges if e.get("safe_division")]
    with open(Path(_VOLT_DIVDIAG_DIR) / f"{dataset}_added.csv", "w", newline="") as fh:
        w = _volt_csv.writer(fh)
        w.writerow(["source_id", "target_id"])
        w.writerows(added)


def add_safe_divisions_postlink(*args, **kwargs):
    bound = _volt_inspect.signature(_volt_asd_original).bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    dataset = a.get("dataset")
    if dataset:
        try:
            _volt_divdiag_dump(a["nodes_by_id"], a["edges"], dataset, a.get("deepcenter_bundle"))
        except Exception as exc:  # the diagnosis must never cost the run
            print(f"  [{dataset}] VOLT division diagnosis FAILED (output unaffected): "
                  f"{type(exc).__name__}: {exc}", flush=True)
    result = _volt_asd_original(*args, **kwargs)
    if dataset:
        try:
            _volt_divdiag_dump_added(result, dataset)
        except Exception as exc:
            print(f"  [{dataset}] VOLT division diagnosis FAILED on the added forks: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    return result


'''


def patch_division_diagnosis(src: str, out_dir: str) -> str:
    """Insert the read-only division logger in front of the next function in cell 5."""
    code = _DIVDIAG_CODE.replace("__OUT_DIR__", repr(out_dir))
    return _replace_once(src, _DIVDIAG_ANCHOR, code + _DIVDIAG_ANCHOR, "division-diagnosis")


# Run x138 on TRAIN movies, which have ground truth, instead of the test placeholders.
# With the old local harness gone, this is how the division diagnosis gets a labelled
# run: Kaggle mounts the full train split under the competition input, so the kernel
# links the chosen movies into a directory of its own and points TEST_DIR there.
# Everything downstream (prediction, post-processing, the final dataset check) reads
# TEST_DIR, and x138's validator -- the one cell that reads TRAIN_DIR -- is disabled
# (BIOHUB_VALIDATOR_ENABLE=0), so the configuration is the submitted one. The links live
# under /tmp, not /kaggle/working, so Kaggle never saves the movies as kernel output.
# NEVER SUBMIT such a build: the graded rerun would grade train movies. The patch
# refuses to run when the test directory holds an embryo that train does not, which is
# what the hidden set is.
_TRAIN_AS_TEST_ANCHOR = 'TEST_DIR = COMP_DIR / "test"\n'
_TRAIN_AS_TEST_CODE = r'''# ---- volt patch: TRAIN movies as the test set. Diagnosis only -- NEVER SUBMIT. ----
_VOLT_TRAIN_STEMS = __STEMS__
_volt_train_dir = COMP_DIR / "train"
_volt_real_test = COMP_DIR / "test"
_volt_prefixes = lambda d: ({p.name.split("_", 1)[0] for p in d.iterdir() if p.name.endswith(".zarr")}
                            if d.exists() else set())
_volt_unseen = _volt_prefixes(_volt_real_test) - _volt_prefixes(_volt_train_dir)
if _volt_unseen:
    raise RuntimeError(f"volt train-as-test build is running on a test set with embryos {sorted(_volt_unseen)} "
                       "that train does not have: this is a graded rerun. This build must never be submitted.")
_volt_missing = [s for s in _VOLT_TRAIN_STEMS if not (_volt_train_dir / f"{s}.zarr").is_dir()]
if _volt_missing:
    raise FileNotFoundError(f"volt train-as-test: {len(_volt_missing)} movie(s) not under {_volt_train_dir}, "
                            f"e.g. {_volt_missing[:5]}")
TEST_DIR = Path(os.environ.get("VOLT_TRAIN_AS_TEST_DIR", "/tmp/volt_train_as_test"))
if TEST_DIR.exists():
    shutil.rmtree(TEST_DIR)
TEST_DIR.mkdir(parents=True)
for _volt_stem in _VOLT_TRAIN_STEMS:
    (TEST_DIR / f"{_volt_stem}.zarr").symlink_to(_volt_train_dir / f"{_volt_stem}.zarr",
                                                 target_is_directory=True)
print(f"VOLT TRAIN-AS-TEST: {len(_VOLT_TRAIN_STEMS)} train movies from {_volt_train_dir} -> {TEST_DIR} "
      "(diagnosis only; NEVER SUBMIT)", flush=True)
'''


def read_stems(path: Path) -> list[str]:
    """One movie name per line; blank lines and ``#`` comments ignored; ``.zarr`` stripped."""
    import re
    stems = []
    for line in path.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        s = s.removesuffix(".zarr")
        if not re.fullmatch(r"[A-Za-z0-9]+_[A-Za-z0-9]+", s):
            raise SystemExit(f"{path}: {s!r} is not a movie name like 44b6_0113de3b")
        stems.append(s)
    if not stems:
        raise SystemExit(f"{path}: no movie names")
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise SystemExit(f"{path}: duplicate movie names {dup[:5]}")
    return stems


def patch_train_as_test(src: str, stems: list[str]) -> str:
    """Point x138's TEST_DIR at a directory of links to the given train movies."""
    code = _TRAIN_AS_TEST_CODE.replace("__STEMS__", repr(list(stems)))
    return _replace_once(src, _TRAIN_AS_TEST_ANCHOR, code, "train-as-test")


def kernel_metadata(slug: str, owner: str, extra_datasets: list[str]) -> dict:
    meta = {"id": f"{owner}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", **BASE_METADATA}
    meta["dataset_sources"] = [*meta["dataset_sources"], *extra_datasets]
    return meta


def patch_v1284_mode(src: str, mode: str) -> str:
    anchor = "os.environ['V1284_MODE']='candidate'\n"
    return _replace_once(src, anchor, f"os.environ['V1284_MODE']={mode!r}  # volt patch\n",
                         "v1284-mode")


def resolve_notebook(path: Path) -> Path:
    """The notebook file itself; a directory holding exactly one .ipynb also works."""
    if path.is_dir():
        found = sorted(path.glob("*.ipynb"))
        if len(found) != 1:
            raise SystemExit(f"{path} is a directory with {len(found)} .ipynb files; "
                             "pass the notebook file itself")
        return found[0]
    if not path.is_file():
        raise SystemExit(f"{path}: no such notebook")
    return path


def build(nb_path: Path, out_dir: Path, slug: str, *, secondary_soup: str | None,
          v1284_mode: str | None, env: dict[str, str] | None = None,
          v1284_head: tuple[str, str] | None = None,
          add_env: dict[str, str] | None = None,
          replace: list[tuple[str, str]] | None = None, fix_d4: bool = False,
          division_diagnosis: str | None = None, train_as_test: list[str] | None = None,
          owner: str = DEFAULT_OWNER) -> Path:
    nb_path = resolve_notebook(nb_path)
    raw = nb_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha == X138_SHA256:
        base = []
    elif sha == ILPDIS15_SHA256:
        base = [(old, lambda s, o=old, n=new: _replace_once(s, o, n, f"ilpdis15 revert {o!r}"))
                for old, new in ILPDIS15_REVERT]
        print(f"base: {nb_path.name} (volt-x138-ilpdis15); ILP disappearance restored to "
              "x138's 2.0 in cell 0 and cell 1")
    else:
        raise SystemExit(f"{nb_path} sha256 {sha} is neither the public x138 snapshot "
                         f"({X138_SHA256}) nor volt-x138-ilpdis15 ({ILPDIS15_SHA256}); "
                         "re-derive the anchors before building")
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
    for old, new in replace or []:
        patches.append((old, lambda s, o=old, n=new: _replace_once(s, o, n, f"replace {o!r}")))
    if division_diagnosis:
        patches.append((_DIVDIAG_ANCHOR,
                        lambda s: patch_division_diagnosis(s, division_diagnosis)))
    if train_as_test:
        patches.append((_TRAIN_AS_TEST_ANCHOR, lambda s: patch_train_as_test(s, train_as_test)))
    if not patches and not fix_d4:
        raise SystemExit("no patch requested; that is the public notebook unchanged")
    patches = base + patches
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if fix_d4:
        patch_d4(code_cells)
    for anchor, apply in patches:
        hits = [c for c in code_cells if anchor in "".join(c["source"])]
        if len(hits) != 1:
            raise SystemExit(f"anchor {anchor!r} is in {len(hits)} cells, expected 1")
        hits[0]["source"] = apply("".join(hits[0]["source"]))
    patched = len(patches) - len(base) + int(fix_d4)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = kernel_metadata(slug, owner, [SOUPS_DATASET] if (secondary_soup or v1284_head) else [])
    code_file = meta["code_file"]
    (out_dir / code_file).write_text(json.dumps(nb, indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir / code_file} ({patched} cell(s) patched) and kernel-metadata.json "
          f"for kernel {meta['id']}")
    if train_as_test:
        print(f"TRAIN-AS-TEST build on {len(train_as_test)} train movies: push and run it, "
              "download its output, and NEVER SUBMIT it.")
    return out_dir / code_file


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook", type=Path, help="the pulled public x138 .ipynb")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--slug", required=True, help="kernel slug under --owner's account")
    p.add_argument("--owner", default=DEFAULT_OWNER,
                   help="Kaggle account the kernel is pushed to; must be the account of the "
                        f"token you push with (default: $KAGGLE_USERNAME, else {FALLBACK_OWNER})")
    p.add_argument("--secondary-soup", choices=sorted(SOUP_MD5), default=None)
    p.add_argument("--v1284-mode", choices=("zero", "candidate"), default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one of cell 0's os.environ settings, e.g. "
                        "BIOHUB_MOTION_RELINK_LEARNED_BONUS=15.0")
    p.add_argument("--add", action="append", default=[], metavar="KEY=VALUE",
                   help="add an os.environ setting cell 0 does not make, e.g. "
                        "BIOHUB_OUTPUT_MOTION_RELINK=0")
    p.add_argument("--replace", action="append", default=[], metavar="OLD=>NEW",
                   help="exact text replacement anywhere in the notebook; the text must "
                        "occur exactly once in exactly one cell (for settings outside "
                        "cell 0 and for cell 1's configuration guard)")
    p.add_argument("--fix-d4", action="store_true",
                   help="replace the duplicated flip-x view with the anti-transpose at "
                        "every TTA site, making it the full 8-element D4 average")
    p.add_argument("--v1284-head", default=None, metavar="FILE=MD5",
                   help="a head file in the soups dataset, with its md5")
    p.add_argument("--division-diagnosis", default=None, metavar="DIR",
                   help="log every division candidate with all gate values, the graph the "
                        "division pass receives, and the forks it adds, into DIR (overridable "
                        "at run time with VOLT_DIVDIAG_DIR). Read-only: the CSV is unchanged. "
                        "For diagnosis runs (with --train-as-test on Kaggle), not for "
                        "submission. Analyse with "
                        "scripts/x138_division_diagnosis.py")
    p.add_argument("--train-as-test", type=Path, default=None, metavar="FILE",
                   help="run on these TRAIN movies (one name per line) instead of the test "
                        "placeholders, so the output can be scored against ground truth. "
                        "Diagnosis only: never submit such a build")
    a = p.parse_args(argv)
    env = dict(kv.split("=", 1) for kv in a.set)
    head = tuple(a.v1284_head.split("=", 1)) if a.v1284_head else None
    add = dict(kv.split("=", 1) for kv in a.add)
    rep = [tuple(r.split("=>", 1)) for r in a.replace]
    build(a.notebook, a.out_dir, a.slug, secondary_soup=a.secondary_soup,
          v1284_mode=a.v1284_mode, env=env, v1284_head=head, add_env=add, replace=rep,
          fix_d4=a.fix_d4, division_diagnosis=a.division_diagnosis,
          train_as_test=read_stems(a.train_as_test) if a.train_as_test else None,
          owner=a.owner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
