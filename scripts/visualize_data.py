"""Visual inspection for the biohub cell-tracking data.

Renders PNGs to --out. Uses the Agg backend, so it never opens a window and
works the same from a plain terminal.

Modes:
    overview  orthogonal MIPs at one timepoint with ground-truth nodes overlaid
    timeline  XY MIP across several timepoints, ground truth overlaid
    norm      effect of the normalization clip range, plus intensity histograms
    tracks    ground-truth trajectories, temporal coverage, division markers
    zoom      consecutive frames cropped around a division or a specific node
    all       overview + timeline + tracks for the chosen dataset, plus norm

Examples:
    uv run python scripts/visualize_data.py --mode all
    uv run python scripts/visualize_data.py --mode overview --dataset 6bba_05b6850b --t 50
    uv run python scripts/visualize_data.py --mode zoom --dataset 6bba_f20478e9 --longest-edge
    uv run python scripts/visualize_data.py --mode norm --n-datasets 6
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

# scripts/ is not a package; import the shared path config the same way the
# other scripts do.
sys.path.insert(0, str(Path(__file__).parent))
from dataspec import DATA_ROOT

SCALE_ZYX = (1.625, 0.40625, 0.40625)
# Anisotropy factor: one z voxel spans this many y or x voxels physically.
Z_ASPECT = SCALE_ZYX[0] / SCALE_ZYX[1]

# Display-only intensity clip. Deliberately not min/max, which renders most
# frames near-black because of bright outlier voxels.
DISPLAY_CLIP = (0.01, 0.99)


# ------------------------------------------------------------------- loading


def open_image(zarr_path: Path):
    group = zarr.open_group(zarr_path, mode="r")
    level = sorted(group.array_keys())[0]
    return group[level], dict(group.attrs)


def stored_quantiles(attrs: dict) -> dict[float, float]:
    q = attrs.get("image_statistics", {})
    q = q.get("quantiles", {}) if isinstance(q, dict) else {}
    out = {}
    for k, v in q.items():
        try:
            out[float(k)] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def clip_range(attrs: dict, frame: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Prefer the quantiles stored in the zarr attrs; fall back to the frame."""
    q = stored_quantiles(attrs)
    vlo = q.get(lo)
    vhi = q.get(hi)
    if vlo is None or vhi is None:
        vlo, vhi = np.quantile(frame.ravel()[::37], [lo, hi])
    return float(vlo), float(vhi)


def read_graph(geff_path: Path) -> dict:
    """Minimal geff reader: node ids, t/z/y/x props, and edge pairs."""
    g = zarr.open_group(geff_path, mode="r")
    node_ids = np.asarray(g["nodes/ids"][:])
    props = {
        name: np.asarray(g[f"nodes/props/{name}/values"][:])
        for name in ("t", "z", "y", "x")
    }
    edges = np.asarray(g["edges/ids"][:])
    edges = edges.reshape(-1, 2) if edges.size else np.empty((0, 2), dtype=np.int64)
    return {"ids": node_ids, **props, "edges": edges}


def nodes_at_t(graph: dict, t: int) -> dict:
    m = graph["t"] == t
    return {k: graph[k][m] for k in ("ids", "t", "z", "y", "x")}


def division_nodes(graph: dict) -> list[int]:
    out_deg = defaultdict(int)
    for s, _ in graph["edges"]:
        out_deg[int(s)] += 1
    return [n for n, d in out_deg.items() if d >= 2]


def node_row(graph: dict, node_id: int) -> dict:
    i = int(np.where(graph["ids"] == node_id)[0][0])
    return {k: graph[k][i] for k in ("t", "z", "y", "x")}


# ------------------------------------------------------------------- helpers


def show_gray(ax, img, attrs, frame_for_fallback, title, aspect=1.0):
    vlo, vhi = clip_range(attrs, frame_for_fallback, *DISPLAY_CLIP)
    ax.imshow(img, cmap="gray", vmin=vlo, vmax=vhi, aspect=aspect, origin="upper",
              interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def scalebar(ax, length_um, px_per_um, label=None, y_frac=0.94, color="yellow"):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = length_um * px_per_um
    x0 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y0 = ylim[0] + (1 - y_frac) * (ylim[1] - ylim[0]) if ylim[0] > ylim[1] else \
        ylim[0] + y_frac * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + width], [y0, y0], color=color, lw=2.5, solid_capstyle="butt")
    ax.text(x0, y0, label or f"{length_um:g} um", color=color, fontsize=8,
            va="bottom", ha="left")


# --------------------------------------------------------------------- modes


def mode_overview(name, zarr_path, geff_path, t, out: Path, slab: int, proj_width: int):
    """Three orthogonal maximum-intensity projections with ground truth overlaid.

    The XZ and YZ panels are aspect-corrected so one z voxel occupies four y or
    x voxels on screen, which is what the physical voxel size implies. Without
    that correction the depth axis looks four times shorter than it is.

    Those two panels collapse a band of y (or x) rather than the whole range.
    In dense tissue a full-range projection saturates by construction: every
    (z, x) column contains some bright nucleus at some y, so the panel goes
    flat white and shows nothing. Pass --proj-width 0 for the full range.
    """
    arr, attrs = open_image(zarr_path)
    T, Z, Y, X = arr.shape
    t = int(np.clip(t, 0, T - 1))
    vol = np.asarray(arr[t])

    graph = read_graph(geff_path) if geff_path else None
    nd = nodes_at_t(graph, t) if graph else None

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.22, wspace=0.12)

    # Full-depth XY projection.
    ax = fig.add_subplot(gs[0, 0])
    show_gray(ax, vol.max(0), attrs, vol, f"XY max over all {Z} z-slices")
    if nd is not None and len(nd["x"]):
        ax.scatter(nd["x"], nd["y"], s=60, facecolors="none", edgecolors="#ff3b30", lw=1.2)
    scalebar(ax, 20, 1 / SCALE_ZYX[2])

    # Thin-slab XY projection: a full-depth MIP superimposes ~64 z-planes of
    # tissue, so a slab is closer to what is separable in one view.
    zc = Z // 2
    z0, z1 = max(0, zc - slab // 2), min(Z, zc + slab // 2 + 1)
    ax = fig.add_subplot(gs[0, 1])
    show_gray(ax, vol[z0:z1].max(0), attrs, vol, f"XY max over z={z0}..{z1 - 1} ({slab} slices)")
    if nd is not None:
        m = (nd["z"] >= z0) & (nd["z"] < z1)
        if m.sum():
            ax.scatter(nd["x"][m], nd["y"][m], s=60, facecolors="none",
                       edgecolors="#ff3b30", lw=1.2)

    # Single plane: what the detector sees at one depth.
    ax = fig.add_subplot(gs[0, 2])
    show_gray(ax, vol[zc], attrs, vol, f"single z-plane z={zc}")
    if nd is not None:
        m = np.abs(nd["z"] - zc) <= 1
        if m.sum():
            ax.scatter(nd["x"][m], nd["y"][m], s=60, facecolors="none",
                       edgecolors="#ff3b30", lw=1.2)

    # XZ: max over a band of y. Rows are z, columns are x.
    yc, xc = Y // 2, X // 2
    if proj_width > 0:
        ya, yb = max(0, yc - proj_width // 2), min(Y, yc + proj_width // 2)
        xa, xb = max(0, xc - proj_width // 2), min(X, xc + proj_width // 2)
    else:
        ya, yb, xa, xb = 0, Y, 0, X

    ax = fig.add_subplot(gs[1, 0])
    show_gray(ax, vol[:, ya:yb, :].max(1), attrs, vol,
              f"XZ max over y={ya}..{yb - 1} (aspect corrected)", aspect=Z_ASPECT)
    if nd is not None:
        m = (nd["y"] >= ya) & (nd["y"] < yb)
        if m.sum():
            ax.scatter(nd["x"][m], nd["z"][m], s=60, facecolors="none",
                       edgecolors="#ff3b30", lw=1.2)
    ax.set_ylabel("z", fontsize=8)

    # YZ: max over a band of x. Rows are z, columns are y.
    ax = fig.add_subplot(gs[1, 1])
    show_gray(ax, vol[:, :, xa:xb].max(2), attrs, vol,
              f"YZ max over x={xa}..{xb - 1} (aspect corrected)", aspect=Z_ASPECT)
    if nd is not None:
        m = (nd["x"] >= xa) & (nd["x"] < xb)
        if m.sum():
            ax.scatter(nd["y"][m], nd["z"][m], s=60, facecolors="none",
                       edgecolors="#ff3b30", lw=1.2)
    ax.set_ylabel("z", fontsize=8)

    # Depth histogram of intensity and of labelled nodes.
    ax = fig.add_subplot(gs[1, 2])
    prof = vol.reshape(Z, -1).mean(1)
    ax.plot(prof, np.arange(Z), color="#378ADD", lw=1.2, label="mean intensity")
    ax.invert_yaxis()
    ax.set_xlabel("mean intensity", fontsize=8)
    ax.set_ylabel("z", fontsize=8)
    ax.tick_params(labelsize=7)
    if nd is not None and len(nd["z"]):
        ax2 = ax.twiny()
        ax2.hist(nd["z"], bins=min(20, Z), orientation="horizontal",
                 color="#ff3b30", alpha=0.35)
        ax2.set_xlabel("labelled nodes", fontsize=8, color="#ff3b30")
        ax2.tick_params(labelsize=7, colors="#ff3b30")
    ax.set_title("signal vs depth", fontsize=9)

    n_lab = len(nd["x"]) if nd is not None else 0
    fig.suptitle(
        f"{name}   t={t}   shape={arr.shape}   "
        f"{n_lab} labelled nodes at this timepoint   "
        f"physical {Z * SCALE_ZYX[0]:.0f} x {Y * SCALE_ZYX[1]:.0f} x {X * SCALE_ZYX[2]:.0f} um",
        fontsize=11,
    )
    p = out / f"overview_{name}_t{t}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def mode_timeline(name, zarr_path, geff_path, out: Path, n_frames: int, slab: int,
                  t_start=None, t_stop=None):
    """XY slab projections across time, to see motion and label coverage.

    Without --t-start/--t-stop this samples the whole movie, giving a stride of
    ~11 frames at T=100. Median motion is under 2 um per frame, so a cell moves
    roughly two nuclear diameters between panels and identity is not traceable
    by eye. Pass a narrow window to get consecutive frames instead.
    """
    arr, attrs = open_image(zarr_path)
    T, Z, Y, X = arr.shape
    lo = 0 if t_start is None else int(np.clip(t_start, 0, T - 1))
    hi = T - 1 if t_stop is None else int(np.clip(t_stop, lo, T - 1))
    # Round, not truncate: linspace returns values like 2.9999999999999996,
    # which astype(int) turns into 2, colliding with the previous frame.
    ts = np.unique(np.round(np.linspace(lo, hi, n_frames)).astype(int))

    graph = read_graph(geff_path) if geff_path else None
    divs = set(division_nodes(graph)) if graph else set()

    zc = Z // 2
    z0, z1 = max(0, zc - slab // 2), min(Z, zc + slab // 2 + 1)

    ncol = min(len(ts), 5)
    nrow = int(np.ceil(len(ts) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.6 * nrow),
                             squeeze=False, constrained_layout=True)

    for i, t in enumerate(ts):
        ax = axes[i // ncol][i % ncol]
        vol_slab = np.asarray(arr[t, z0:z1]).max(0)
        n_here = 0
        if graph is not None:
            nd = nodes_at_t(graph, int(t))
            m = (nd["z"] >= z0) & (nd["z"] < z1)
            n_here = int(m.sum())
            if n_here:
                ax.scatter(nd["x"][m], nd["y"][m], s=55, facecolors="none",
                           edgecolors="#ff3b30", lw=1.1)
            dm = m & np.isin(nd["ids"], list(divs))
            if dm.sum():
                ax.scatter(nd["x"][dm], nd["y"][dm], s=170, marker="s",
                           facecolors="none", edgecolors="#EF9F27", lw=1.8)
        show_gray(ax, vol_slab, attrs, vol_slab, f"t={t}   {n_here} GT in slab")
    for j in range(len(ts), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles = [
        Line2D([], [], marker="o", ls="", mfc="none", mec="#ff3b30", label="GT node"),
        Line2D([], [], marker="s", ls="", mfc="none", mec="#EF9F27", label="division"),
    ]
    # "outside" placement is what constrained_layout reserves room for; a plain
    # "lower center" legend is drawn on top of the last panel instead.
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               fontsize=9, frameon=False)
    stride = int(ts[1] - ts[0]) if len(ts) > 1 else 0
    fig.suptitle(
        f"{name}   XY max over z={z0}..{z1 - 1}   "
        f"t={ts[0]}..{ts[-1]}, stride {stride}",
        fontsize=11,
    )
    # constrained_layout reserves room for each row's title. tight_layout plus
    # bbox_inches="tight" does not: savefig recomputes the bounding box and
    # discards the padding, so lower-row titles land under the images above.
    p = out / f"timeline_{name}_t{ts[0]}-{ts[-1]}.png"
    fig.savefig(p, dpi=105)
    plt.close(fig)
    return p


def mode_norm(records, out: Path, n_datasets: int, t: int, slab: int):
    """Compare normalization clip ranges the way the real loader applies them.

    io._process_on_gpu computes (x - q_lo) / (q_hi - q_lo), then clamps to
    [0, 4] -- not [0, 1]. Voxels above the upper clip point therefore survive
    as values between 1 and 4 and remain distinguishable to the network. Only
    values above 4.0 are genuinely destroyed.

    Matplotlib can only render [0, 1], so the image panels necessarily clip at
    1.0 and look saturated where the data is not. The panel titles and the
    normalized-value histogram carry the real numbers, including what happens
    at labelled cell positions specifically.
    """
    PIPELINE_CLAMP = 4.0

    picks = []
    for prefix in sorted({r["prefix"] for r in records}):
        same = [r for r in records if r["prefix"] == prefix]
        step = max(1, len(same) // max(1, n_datasets // 2))
        picks.extend(same[::step][: n_datasets // 2])
    picks = picks[:n_datasets]

    ranges = [(0.0, 1.0, "min-max (current)", "#E24B4A"),
              (0.001, 0.999, "0.1 / 99.9 pct", "#1D9E75"),
              (0.01, 0.99, "1 / 99 pct", "#7F77DD")]

    fig, axes = plt.subplots(len(picks), 5, figsize=(20, 3.6 * len(picks)), squeeze=False)

    for i, rec in enumerate(picks):
        arr, attrs = open_image(rec["zarr_path"])
        T, Z, Y, X = arr.shape
        tt = min(t, T - 1)
        zc = Z // 2
        z0, z1 = max(0, zc - slab // 2), min(Z, zc + slab // 2 + 1)
        vol = np.asarray(arr[tt]).astype(np.float32)
        img = vol[z0:z1].max(0)
        q = stored_quantiles(attrs)

        # Intensity sampled at labelled cell centres, which is the population
        # that actually matters for whether clamping costs anything.
        gt_vals = None
        if rec["geff_path"]:
            g = read_graph(rec["geff_path"])
            nd = nodes_at_t(g, tt)
            if len(nd["z"]):
                gt_vals = vol[nd["z"].astype(int), nd["y"].astype(int), nd["x"].astype(int)]

        norm_hists = []
        for j, (lo, hi, label, colour) in enumerate(ranges):
            vlo = q.get(lo, float(vol.min()))
            vhi = q.get(hi, float(vol.max()))
            span = max(vhi - vlo, 1e-6)

            # Exactly what the loader does, clamp included.
            norm_vol = np.clip((vol - vlo) / span, 0.0, PIPELINE_CLAMP)
            norm_img = np.clip((img - vlo) / span, 0.0, PIPELINE_CLAMP)
            norm_hists.append((norm_vol.ravel()[::53], label, colour))

            frac_over_1 = float((norm_vol > 1.0).mean())
            frac_clamped = float((norm_vol >= PIPELINE_CLAMP).mean())

            gt_note = ""
            if gt_vals is not None and len(gt_vals):
                gtn = np.clip((gt_vals - vlo) / span, 0.0, PIPELINE_CLAMP)
                gt_note = (f"\nGT cells: median {np.median(gtn):.2f}, "
                           f"{float((gtn >= PIPELINE_CLAMP).mean()) * 100:.0f}% clamped")

            ax = axes[i][j]
            # vmax=1 is a display limit only; the data extends to 4.
            ax.imshow(np.clip(norm_img, 0, 1), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(
                f"{label}   clip {vlo:.0f}-{vhi:.0f}\n"
                f"{frac_over_1 * 100:.1f}% of voxels >1.0 (kept by net, "
                f"clipped by this display)\n"
                f"{frac_clamped * 100:.2f}% >=4.0 (lost){gt_note}",
                fontsize=7.5,
            )
            if j == 0:
                ax.set_ylabel(rec["name"], fontsize=9)

        # Normalized values, which is what the network actually receives.
        ax = axes[i][3]
        for vals, label, colour in norm_hists:
            ax.hist(vals, bins=140, range=(0, PIPELINE_CLAMP), histtype="step",
                    color=colour, lw=1.1, log=True, label=label)
        ax.axvline(1.0, color="#444441", ls="--", lw=1.0)
        ax.axvline(PIPELINE_CLAMP, color="#E24B4A", ls="-", lw=1.2)
        ax.text(1.02, 0.96, "display limit", transform=ax.get_xaxis_transform(),
                fontsize=6.5, rotation=90, va="top", color="#444441")
        ax.text(PIPELINE_CLAMP - 0.12, 0.96, "pipeline clamp",
                transform=ax.get_xaxis_transform(), fontsize=6.5, rotation=90,
                va="top", ha="right", color="#E24B4A")
        if gt_vals is not None and len(gt_vals):
            for lo, hi, label, colour in ranges:
                vlo, vhi = q.get(lo, 0.0), q.get(hi, 1.0)
                gtn = np.clip((gt_vals - vlo) / max(vhi - vlo, 1e-6), 0, PIPELINE_CLAMP)
                ax.plot(gtn, np.full_like(gtn, 1.5), "|", ms=7, color=colour, alpha=0.9)
        ax.set_title("normalized values (what the net sees)", fontsize=8)
        ax.set_xlabel("normalized intensity", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6.5, loc="upper right")

        # Raw intensity with the three clip points marked.
        ax = axes[i][4]
        ax.hist(vol.ravel()[::53], bins=140, color="#B4B2A9", log=True)
        for lo, hi, label, colour in ranges:
            for v, style in ((q.get(lo), ":"), (q.get(hi), "-")):
                if v is not None:
                    ax.axvline(v, ls=style, lw=1.0, color=colour)
        if gt_vals is not None and len(gt_vals):
            ax.plot(gt_vals, np.full_like(gt_vals, 2.0), "|", ms=7,
                    color="#ff3b30", alpha=0.9)
        ax.set_title("raw intensity, red ticks = GT cells", fontsize=8)
        ax.set_xlabel("raw value", fontsize=7)
        ax.tick_params(labelsize=7)

    handles = [Line2D([], [], color=c, label=l) for _, _, l, c in ranges]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        "Normalization clip range. Image panels clip at 1.0 because matplotlib "
        "must; the pipeline clamps at 4.0, so titles and histograms carry the "
        "real numbers.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    p = out / "normalization_comparison.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return p


def mode_tracks(name, geff_path, out: Path):
    """Ground-truth structure only: trajectories, temporal coverage, motion."""
    graph = read_graph(geff_path)
    edges = graph["edges"]
    ids = graph["ids"]
    idx = {int(n): i for i, n in enumerate(ids)}
    divs = set(division_nodes(graph))

    # Lineage membership by union-find over the undirected graph.
    parent = {int(n): int(n) for n in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, tg in edges:
        s, tg = int(s), int(tg)
        rs, rt = find(s), find(tg)
        if rs != rt:
            parent[rs] = rt
    lineage = {int(n): find(int(n)) for n in ids}
    roots = sorted(set(lineage.values()))
    root_order = {r: i for i, r in enumerate(roots)}

    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # XY trajectories.
    ax = axes[0]
    for s, tg in edges:
        i, j = idx[int(s)], idx[int(tg)]
        c = cmap(root_order[lineage[int(s)]] % 20)
        ax.plot([graph["x"][i], graph["x"][j]], [graph["y"][i], graph["y"][j]],
                color=c, lw=0.9, alpha=0.85)
    if divs:
        di = [idx[d] for d in divs]
        ax.scatter(graph["x"][di], graph["y"][di], s=70, marker="s",
                   facecolors="none", edgecolors="k", lw=1.4, zorder=5, label="division")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"XY trajectories, {len(roots)} lineages", fontsize=10)
    ax.set_xlabel("x (voxels)"); ax.set_ylabel("y (voxels)")
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Temporal coverage per lineage.
    ax = axes[1]
    for r in roots:
        members = [int(n) for n in ids if lineage[int(n)] == r]
        tt = np.sort(graph["t"][[idx[m] for m in members]])
        y = root_order[r]
        ax.plot(tt, np.full_like(tt, y, dtype=float), "|", ms=4,
                color=cmap(y % 20))
    if divs:
        for d in divs:
            ax.plot(graph["t"][idx[d]], root_order[lineage[d]], "s", ms=5,
                    mfc="none", mec="k", mew=1.2)
    ax.set_title("temporal coverage per lineage", fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("lineage index")

    # Frame-to-frame displacement in physical units.
    ax = axes[2]
    scale = np.asarray(SCALE_ZYX)
    disp = []
    for s, tg in edges:
        i, j = idx[int(s)], idx[int(tg)]
        d = np.array([graph["z"][j] - graph["z"][i],
                      graph["y"][j] - graph["y"][i],
                      graph["x"][j] - graph["x"][i]], dtype=float) * scale
        disp.append(np.linalg.norm(d))
    disp = np.asarray(disp)
    if disp.size:
        ax.hist(disp, bins=50, color="#5DCAA5", edgecolor="none")
        ax.axvline(7.0, color="#E24B4A", ls="--", lw=1.3,
                   label="7 um (node-match gate, for scale only)")
        ax.axvline(np.percentile(disp, 99), color="#534AB7", ls=":", lw=1.3,
                   label=f"p99 = {np.percentile(disp, 99):.1f} um")
        ax.legend(fontsize=8)
    ax.set_title("frame-to-frame displacement", fontsize=10)
    ax.set_xlabel("um"); ax.set_ylabel("edges")

    fig.suptitle(
        f"{name}   {len(ids)} nodes   {len(edges)} edges   {len(divs)} divisions",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = out / f"tracks_{name}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def mode_zoom(name, zarr_path, geff_path, out: Path, node_id, longest_edge, half, slab):
    """Consecutive frames cropped around one event.

    Picks the first division if no node is given, or the longest edge in the
    dataset with --longest-edge, which is how to eyeball a suspected
    annotation error.
    """
    arr, attrs = open_image(zarr_path)
    T, Z, Y, X = arr.shape
    graph = read_graph(geff_path)
    idx = {int(n): i for i, n in enumerate(graph["ids"])}
    scale = np.asarray(SCALE_ZYX)
    label = ""

    if longest_edge:
        best, best_d = None, -1.0
        for s, tg in graph["edges"]:
            i, j = idx[int(s)], idx[int(tg)]
            d = np.linalg.norm(np.array([graph["z"][j] - graph["z"][i],
                                         graph["y"][j] - graph["y"][i],
                                         graph["x"][j] - graph["x"][i]]) * scale)
            if d > best_d:
                best_d, best = d, (int(s), int(tg))
        node_id = best[0]
        label = f"longest edge {best[0]}->{best[1]}, {best_d:.1f} um"
    elif node_id is None:
        divs = division_nodes(graph)
        if not divs:
            return None
        node_id = divs[0]
        label = f"division at node {node_id}"

    row = node_row(graph, node_id)
    t0, zc, yc, xc = int(row["t"]), int(row["z"]), int(row["y"]), int(row["x"])
    ts = [t for t in range(t0 - 1, t0 + 3) if 0 <= t < T]
    y0, y1 = max(0, yc - half), min(Y, yc + half)
    x0, x1 = max(0, xc - half), min(X, xc + half)
    z0, z1 = max(0, zc - slab // 2), min(Z, zc + slab // 2 + 1)

    children = defaultdict(list)
    for s, tg in graph["edges"]:
        children[int(s)].append(int(tg))

    fig, axes = plt.subplots(1, len(ts), figsize=(3.4 * len(ts), 3.8), squeeze=False)
    for k, t in enumerate(ts):
        ax = axes[0][k]
        crop = np.asarray(arr[t, z0:z1, y0:y1, x0:x1]).max(0)
        show_gray(ax, crop, attrs, crop, f"t={t}" + (" (event)" if t == t0 else ""))
        nd = nodes_at_t(graph, t)
        m = (nd["y"] >= y0) & (nd["y"] < y1) & (nd["x"] >= x0) & (nd["x"] < x1)
        if m.sum():
            ax.scatter(nd["x"][m] - x0, nd["y"][m] - y0, s=90, facecolors="none",
                       edgecolors="#ff3b30", lw=1.3)
        marks = [node_id] if t == t0 else (children[node_id] if t == t0 + 1 else [])
        for nid in marks:
            if nid in idx:
                i = idx[nid]
                if y0 <= graph["y"][i] < y1 and x0 <= graph["x"][i] < x1:
                    ax.scatter(graph["x"][i] - x0, graph["y"][i] - y0, s=210,
                               marker="s", facecolors="none",
                               edgecolors="#EF9F27", lw=2.0)

    fig.suptitle(f"{name}   {label}   crop y={y0}:{y1} x={x0}:{x1} z={z0}:{z1}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = out / f"zoom_{name}_n{node_id}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


# -------------------------------------------------------------------- driver


def find_records(train_dir: Path) -> list[dict]:
    out = []
    for z in sorted(train_dir.glob("*.zarr")):
        g = train_dir / f"{z.stem}.geff"
        out.append({
            "name": z.stem,
            "prefix": z.stem.split("_")[0],
            "zarr_path": z,
            "geff_path": g if g.exists() else None,
        })
    return out


def pick_with_divisions(records) -> dict | None:
    """Prefer a dataset that actually contains a division, since most do not."""
    for r in records:
        if r["geff_path"] and division_nodes(read_graph(r["geff_path"])):
            return r
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Directory containing train/ and test/. Default: "
                         "dataspec.DATA_ROOT, which honours $VOLT_DATA_DIR.")
    ap.add_argument("--out", default="reports/figures", type=Path)
    ap.add_argument("--mode", default="all",
                    choices=["overview", "timeline", "norm", "tracks", "zoom", "all"])
    ap.add_argument("--dataset", default=None, help="dataset name, default picks one with a division")
    ap.add_argument("--t", type=int, default=50)
    ap.add_argument("--slab", type=int, default=8, help="z-slices per thin-slab projection")
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument("--t-start", type=int, default=None,
                    help="first timepoint for --mode timeline")
    ap.add_argument("--t-stop", type=int, default=None,
                    help="last timepoint for --mode timeline; narrow this for consecutive frames")
    ap.add_argument("--proj-width", type=int, default=64,
                    help="voxels collapsed in the XZ/YZ panels; 0 = full range (saturates on dense tissue)")
    ap.add_argument("--n-datasets", type=int, default=4, help="datasets for --mode norm")
    ap.add_argument("--node-id", type=int, default=None)
    ap.add_argument("--longest-edge", action="store_true",
                    help="zoom on the largest displacement edge instead of a division")
    ap.add_argument("--half", type=int, default=48, help="zoom crop half-width in voxels")
    args = ap.parse_args(argv)

    train = (args.data_root or DATA_ROOT) / "train"
    if not train.is_dir():
        print(f"not found: {train.resolve()}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    records = find_records(train)
    if not records:
        print(f"no datasets under {train}")
        return 1

    if args.dataset:
        sel = next((r for r in records if r["name"] == args.dataset), None)
        if sel is None:
            print(f"dataset not found: {args.dataset}")
            return 1
    else:
        sel = pick_with_divisions(records) or records[0]
        print(f"selected {sel['name']}")

    written = []
    if args.mode in ("overview", "all"):
        written.append(mode_overview(sel["name"], sel["zarr_path"], sel["geff_path"],
                                     args.t, args.out, args.slab, args.proj_width))
    if args.mode in ("timeline", "all"):
        written.append(mode_timeline(sel["name"], sel["zarr_path"], sel["geff_path"],
                                     args.out, args.n_frames, args.slab,
                                     args.t_start, args.t_stop))
    if args.mode in ("tracks", "all") and sel["geff_path"]:
        written.append(mode_tracks(sel["name"], sel["geff_path"], args.out))
    if args.mode in ("norm", "all"):
        written.append(mode_norm(records, args.out, args.n_datasets, args.t, args.slab))
    if args.mode == "zoom" or (args.mode == "all" and sel["geff_path"]):
        p = mode_zoom(sel["name"], sel["zarr_path"], sel["geff_path"], args.out,
                      args.node_id, args.longest_edge, args.half, args.slab)
        if p:
            written.append(p)
        elif args.mode == "zoom":
            print("no division found in this dataset; pass --node-id or --longest-edge")

    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
