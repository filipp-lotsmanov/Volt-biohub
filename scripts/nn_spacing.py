"""Nearest-neighbour spacing per dataset, from GT centroids."""
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from scipy.spatial import cKDTree

from dataspec import DATASET_PATH
from volt_biohub.scoring import ORIGINAL_SCALE

FLOOR_UM = 6.5
K = td.DEFAULT_ATTR_KEYS
rows = []

for geff in sorted(Path(DATASET_PATH).glob("*.geff")):
    g = td.graph.IndexedRXGraph.from_geff(geff)
    g = g[0] if isinstance(g, tuple) else g
    df = g.node_attrs(attr_keys=[K.T, "z", "y", "x"])
    nn = []
    for _, frame in df.group_by(K.T):
        if frame.height < 2:
            continue
        pts = frame.select("z", "y", "x").to_numpy() * np.array(ORIGINAL_SCALE)
        d, _ = cKDTree(pts).query(pts, k=2)
        nn.append(d[:, 1])
    if not nn:
        continue
    nn = np.concatenate(nn)
    rows.append({
        "name": geff.stem,
        "nn_median_um": float(np.median(nn)),
        "nn_p10_um": float(np.percentile(nn, 10)),
        "frac_below_floor": float((nn < FLOOR_UM).mean()),
        "n_nodes": int(nn.size),
    })

out = Path("reports/nn_spacing.csv")
pl.DataFrame(rows).write_csv(out)
print(f"{len(rows)} datasets -> {out}")