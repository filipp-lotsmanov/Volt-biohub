"""Does a dividing nucleus *look* different, and can that rank fork candidates?

The division term is 10 % of the score and we earn 0.007 of it. `CLAUDE.md`
records that geometric features (sister separation, symmetry, daughter
distances) do not separate real forks from false ones at the quartiles, and
names the one axis that study never touched: **appearance**. Mitosis condenses
chromatin, so a nucleus about to divide should be brighter and more compact
than the same cell two frames earlier -- a signal no geometric gate can see.

This measures that on ground truth before any of it is built into the ranker:
for every GT division, the parent's intensity profile against control nodes
from the same videos and frames, normalised exactly the way training
normalises (quantiles 0.001/0.999, lower clamp, no upper clamp).

Run: uv run python scripts/division_appearance.py [--embryo 6bba] [--max-samples N]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import zarr

SCALE_UM = np.array([1.625, 0.40625, 0.40625])


def _norm_frame(arr: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    out = (arr.astype(np.float32) - q_lo) / max(q_hi - q_lo, 1e-6)
    return np.clip(out, 0.0, None)


def _ball(vol: np.ndarray, zyx: np.ndarray, radius_um: float) -> np.ndarray:
    """Voxel values within *radius_um* of a centroid, in original voxel space."""
    half = np.ceil(radius_um / SCALE_UM).astype(int)
    lo = np.maximum(zyx - half, 0)
    hi = np.minimum(zyx + half + 1, vol.shape)
    if np.any(lo >= hi):
        return np.empty(0, dtype=np.float32)
    sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    grid = np.stack(np.meshgrid(
        *[np.arange(a, b) for a, b in zip(lo, hi, strict=True)], indexing="ij"), axis=-1)
    d = np.linalg.norm((grid - zyx) * SCALE_UM, axis=-1)
    return sub[d <= radius_um].ravel()


def _profile(vol: np.ndarray, zyx: np.ndarray) -> dict[str, float]:
    inner = _ball(vol, zyx, 3.0)
    outer = _ball(vol, zyx, 7.0)
    if inner.size == 0 or outer.size == 0:
        return {}
    shell = outer[outer.size - inner.size:] if outer.size > inner.size else outer
    return {
        "peak": float(vol[tuple(zyx)]),
        "mean3": float(inner.mean()),
        "p90_3": float(np.percentile(inner, 90)),
        # Compactness: how concentrated the signal is inside 3 um relative to
        # the 7 um neighbourhood. Condensed chromatin should raise it.
        "concentration": float(inner.mean() / (outer.mean() + 1e-6)),
        "shell_mean": float(shell.mean()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path(os.environ["VOLT_DATA_DIR"]))
    p.add_argument("--embryo", default=None, help="restrict to one embryo prefix")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--controls-per-division", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("reports/division_appearance.csv"))
    a = p.parse_args(argv)

    stems = sorted(q.name[:-5] for q in a.data_dir.iterdir() if q.name.endswith(".zarr"))
    if a.embryo:
        stems = [s for s in stems if s.startswith(a.embryo)]

    rng = np.random.default_rng(0)
    rows: list[dict] = []
    n_done = 0
    for stem in stems:
        geff = a.data_dir / f"{stem}.geff"
        if not geff.exists():
            continue
        g = zarr.open_group(str(geff), mode="r")
        edges = np.asarray(g["edges/ids"])
        if edges.size == 0:
            continue
        src, cnt = np.unique(edges[:, 0], return_counts=True)
        forks = src[cnt >= 2]
        if len(forks) == 0:
            continue

        ids = np.asarray(g["nodes/ids"])
        t = np.asarray(g["nodes/props/t/values"])
        zz = np.asarray(g["nodes/props/z/values"])
        yy = np.asarray(g["nodes/props/y/values"])
        xx = np.asarray(g["nodes/props/x/values"])
        idx = {int(i): k for k, i in enumerate(ids)}
        pred = {int(b): int(s) for s, b in edges}

        img = zarr.open_group(str(a.data_dir / f"{stem}.zarr"), mode="r")
        q = img.attrs["image_statistics"]["quantiles"]
        q_lo, q_hi = float(q["0.001"]), float(q["0.999"])
        arr = img["0"]

        # Controls: nodes with exactly one child, sampled from the same frames
        # as the divisions so any time-of-movie trend cancels.
        one_child = {int(s) for s, c in zip(src, cnt, strict=True) if c == 1}
        frames_needed = set()
        for f in forks:
            k = idx[int(f)]
            frames_needed.update(range(max(0, int(t[k]) - 2), min(arr.shape[0], int(t[k]) + 2)))
        cache: dict[int, np.ndarray] = {}

        # Bound as defaults rather than captured: both closures are defined
        # inside the per-sample loop, so a later iteration would silently
        # rebind them (B023). They are in fact only called within their own
        # iteration, but binding here makes that structural.
        def frame(ti: int, *, cache=cache, arr=arr,
                  q_lo=q_lo, q_hi=q_hi) -> np.ndarray:
            if ti not in cache:
                cache[ti] = _norm_frame(np.asarray(arr[ti]), q_lo, q_hi)
            return cache[ti]

        def record(node_id: int, kind: str, *, idx=idx, t=t, zz=zz, yy=yy,
                   xx=xx, pred=pred, stem=stem, frame=frame) -> None:
            k = idx[node_id]
            ti = int(t[k])
            here = np.array([int(zz[k]), int(yy[k]), int(xx[k])])
            prof = _profile(frame(ti), here)
            if not prof:
                return
            row = {"stem": stem, "kind": kind, "node": node_id, "t": ti, **prof}
            # Self-referenced rise: the same lineage two frames earlier. A
            # ratio removes the 11x per-sample brightness spread that would
            # otherwise swamp a within-cell change.
            back = node_id
            for _ in range(2):
                back = pred.get(back, None)
                if back is None:
                    break
            if back is not None and back in idx and int(t[idx[back]]) == ti - 2:
                kb = idx[back]
                pb = _profile(frame(ti - 2), np.array([int(zz[kb]), int(yy[kb]), int(xx[kb])]))
                if pb:
                    for key in ("peak", "mean3", "p90_3", "concentration"):
                        row[f"rise_{key}"] = prof[key] / (pb[key] + 1e-6)
            rows.append(row)

        for f in forks:
            record(int(f), "division")
        pool = [n for n in one_child if n in idx and int(t[idx[n]]) in frames_needed]
        if pool:
            take = min(len(pool), a.controls_per_division * len(forks))
            for n in rng.choice(pool, size=take, replace=False):
                record(int(n), "control")

        n_done += 1
        if a.max_samples and n_done >= a.max_samples:
            break

    if not rows:
        print("no divisions found")
        return 1

    import csv
    a.out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    div = [r for r in rows if r["kind"] == "division"]
    ctl = [r for r in rows if r["kind"] == "control"]
    print(f"{len(div)} divisions, {len(ctl)} controls, {n_done} samples -> {a.out}")
    print(f"{'feature':16s} {'div p25/p50/p75':26s} {'control p25/p50/p75':26s} AUC")
    for key in ("peak", "mean3", "p90_3", "concentration",
                "rise_peak", "rise_mean3", "rise_p90_3", "rise_concentration"):
        dv = np.array([r[key] for r in div if key in r], dtype=float)
        cv = np.array([r[key] for r in ctl if key in r], dtype=float)
        if len(dv) < 5 or len(cv) < 5:
            continue
        # AUC via the Mann-Whitney U identity: P(random division > random
        # control). 0.5 is no signal; the geometric features in CLAUDE.md all
        # sit there.
        allv = np.concatenate([dv, cv])
        order = allv.argsort()
        ranks = np.empty(len(allv))
        ranks[order] = np.arange(1, len(allv) + 1)
        u = ranks[:len(dv)].sum() - len(dv) * (len(dv) + 1) / 2
        auc = u / (len(dv) * len(cv))
        d25, d50, d75 = np.percentile(dv, [25, 50, 75])
        c25, c50, c75 = np.percentile(cv, [25, 50, 75])
        print(f"{key:16s} {d25:7.3f}/{d50:7.3f}/{d75:7.3f}    "
              f"{c25:7.3f}/{c50:7.3f}/{c75:7.3f}    {auc:.3f}  (n={len(dv)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
