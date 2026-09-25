#!/usr/bin/env python
"""Train a sub-voxel coordinate head compatible with the public x138 notebook.

x138 (anvithpothula/biohub-x138, public LB ~0.953) moves every detection by up
to 2 um with a tiny MLP over frozen U-Net features: the 32-channel primary
features at the peak voxel and the six neighbour-minus-centre differences,
224 inputs -> 32 -> 3, squashed onto the 2 um ball. Its author trained it on
4,136 detection/GT pairs from 20 movies and notes that a 4-movie version was
*worse* than no head, i.e. the head is data-limited. All 199 train movies
carry GT, so this trains the same architecture on everything.

Inputs are the notebook's own ``V1284_MODE=capture`` dumps -- one npz per movie
and frame with ``coords`` (n, 4) = [t, z, y, x] on the 1.625 um grid and
``features`` (n, 224) -- so the training inputs are exactly what the deployed
head sees, including the notebook's 8-view feature TTA.

Pairs are mutual nearest neighbours between detections and GT nodes of the
same frame within ``--radius-um``. GT is sparse, so most detections have no
target; that is fine for a regressor, and unlike a detector there is no
negative class to get wrong.

    uv run python scripts/train_coord_head.py --capture-dir <dir> --out-dir <dir>

Writes ``head_all.pt`` plus one head per held-out embryo (``head_no44b6.pt``
trained without 44b6, etc.) for leave-one-embryo-out evaluation, each in the
format the notebook loads: ``{"state_dict", "mean", "scale"}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from dataspec import DATASET_PATH

GRID_UM = 1.625                                  # the 1,4,4-strided grid is isotropic
GT_UM = np.array([1.625, 0.40625, 0.40625])      # original anisotropic voxels
BOUND_UM = 2.0                                   # x138's bounded() radius


def make_head(hidden: int = 32) -> torch.nn.Module:
    head = torch.nn.Sequential(torch.nn.Linear(224, hidden), torch.nn.SiLU(),
                               torch.nn.Linear(hidden, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Identical to the notebook's: radial squash onto the open 2 um ball."""
    delta = head(x)
    return BOUND_UM * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def gt_by_frame(data_dir: Path, name: str) -> dict[int, np.ndarray]:
    g = zarr.open_group(str(data_dir / f"{name}.geff"), mode="r")
    t = np.asarray(g["nodes/props/t/values"]).astype(int)
    zyx = np.stack([np.asarray(g[f"nodes/props/{a}/values"]) for a in "zyx"], 1)
    um = zyx.astype(np.float64) * GT_UM
    return {int(k): um[t == k] for k in np.unique(t)}


def pairs_for_movie(cap: Path, gt: dict[int, np.ndarray], radius_um: float):
    xs, ys, ds = [], [], []
    for f in sorted(cap.glob("*.npz")):
        d = np.load(f)
        coords, feats = d["coords"], d["features"]
        if not len(coords):
            continue
        t = int(coords[0, 0])
        g = gt.get(t)
        if g is None or not len(g):
            continue
        det_um = coords[:, 1:].astype(np.float64) * GRID_UM
        # Mutual nearest within the radius: a GT node claims one detection and
        # vice versa, so two detections of one nucleus cannot both be pulled
        # onto it and a detection between two nuclei is not trained on either.
        d_g, j_g = cKDTree(det_um).query(g, distance_upper_bound=radius_um)
        _, j_d = cKDTree(g).query(det_um, distance_upper_bound=radius_um)
        for gi, (dist, dj) in enumerate(zip(d_g, j_g, strict=True)):
            if np.isfinite(dist) and j_d[dj] == gi:
                xs.append(feats[dj])
                ys.append(g[gi] - det_um[dj])
                ds.append(dist)
    if not xs:
        return None
    return np.stack(xs), np.stack(ys).astype(np.float32), np.asarray(ds)


def train_head(x: np.ndarray, y: np.ndarray, *, hidden: int, epochs: int, lr: float,
               wd: float, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    mean = x.mean(0)
    scale = x.std(0) + 1e-6
    xt = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    # Targets beyond the ball cannot be reached; clip radially so the loss does
    # not chase them with ever-larger raw outputs.
    n = np.linalg.norm(y, axis=1, keepdims=True)
    yc = y * np.minimum(1.0, (0.95 * BOUND_UM) / np.maximum(n, 1e-9))
    yt = torch.as_tensor(yc, dtype=torch.float32, device=device)
    head = make_head(hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 1024
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.mse_loss(bounded(head, xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return {"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "mean": torch.as_tensor(mean, dtype=torch.float32),
            "scale": torch.as_tensor(scale, dtype=torch.float32)}


@torch.no_grad()
def apply_head(saved: dict, x: np.ndarray, hidden: int) -> np.ndarray:
    head = make_head(hidden)
    head.load_state_dict(saved["state_dict"])
    xt = (torch.as_tensor(x, dtype=torch.float32) - saved["mean"]) / saved["scale"]
    return bounded(head, xt).numpy()


def residual(y: np.ndarray, shift: np.ndarray | None = None) -> np.ndarray:
    return np.linalg.norm(y - (0.0 if shift is None else shift), axis=1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path(DATASET_PATH))
    p.add_argument("--radius-um", type=float, default=4.0)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-head", type=Path, default=None,
                   help="x138's own v1284_head.pt, scored on the same pairs")
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    per_movie = {}
    for cap in sorted(d for d in a.capture_dir.iterdir() if d.is_dir()):
        res = pairs_for_movie(cap, gt_by_frame(a.data_dir, cap.name), a.radius_um)
        if res is not None:
            per_movie[cap.name] = res
    names = sorted(per_movie)
    emb = np.array([n.split("_")[0] for n in names])
    total = sum(len(per_movie[n][0]) for n in names)
    print(f"{len(names)} movies, {total} pairs (radius {a.radius_um} um)", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    kw = {"hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "wd": a.wd,
          "seed": a.seed, "device": device}
    report: dict = {"config": {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in vars(a).items()},
                    "movies": len(names), "pairs": total, "folds": {}}
    ref = (torch.load(a.reference_head, map_location="cpu", weights_only=True)
           if a.reference_head else None)

    def stack(sel):
        return (np.concatenate([per_movie[n][0] for n in sel]),
                np.concatenate([per_movie[n][1] for n in sel]))

    for held in sorted(set(emb)):
        tr = [n for n, e in zip(names, emb, strict=True) if e != held]
        te = [n for n, e in zip(names, emb, strict=True) if e == held]
        xtr, ytr = stack(tr)
        xte, yte = stack(te)
        saved = train_head(xtr, ytr, **kw)
        torch.save(saved, a.out_dir / f"head_no{held}.pt")
        base = residual(yte)
        ours = residual(yte, apply_head(saved, xte, a.hidden))
        row = {"train_pairs": len(xtr), "test_pairs": len(xte),
               "mean_um_before": float(base.mean()), "mean_um_ours": float(ours.mean()),
               "median_um_before": float(np.median(base)),
               "median_um_ours": float(np.median(ours)),
               "frac_closer_ours": float((ours < base).mean())}
        # Shrinkage: does pulling the move toward zero generalise better across
        # embryos? A regressor fit on one embryo is expected to overreach on
        # another; if so the best shrink here is below 1.
        pred = apply_head(saved, xte, a.hidden)
        row["mean_um_by_shrink"] = {s: float(residual(yte, s * pred).mean())
                                    for s in (0.25, 0.5, 0.75, 1.0)}
        if ref is not None:
            theirs = residual(yte, apply_head(ref, xte, 32))
            row["mean_um_theirs"] = float(theirs.mean())
            row["median_um_theirs"] = float(np.median(theirs))
        # Per-movie consistency, the gate the x138 author used ("16 of 20").
        gains = []
        for n in te:
            xm, ym = per_movie[n][0], per_movie[n][1]
            gains.append(residual(ym).mean() - residual(ym, apply_head(saved, xm, a.hidden)).mean())
        row["movies_improved"] = f"{int(np.sum(np.array(gains) > 0))}/{len(gains)}"
        report["folds"][held] = row
        print(held, json.dumps(row), flush=True)

    x_all, y_all = stack(names)
    torch.save(train_head(x_all, y_all, **kw), a.out_dir / "head_all.pt")
    (a.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {a.out_dir}/head_all.pt, head_no*.pt, report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
