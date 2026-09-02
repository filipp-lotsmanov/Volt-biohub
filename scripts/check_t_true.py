""" T_true (estimated_number_of_nodes) sets the node-count penalty in the adjusted
edge Jaccard, but it is absent from the test split. This script asks two things:

1. Can T_true be predicted from zarr metadata alone, well enough to calibrate
   det_threshold per dataset at inference time?
2. If a single global threshold produced the same node count everywhere, how
   much penalty would that incur once samples are edge-weighted the way
   metrics.summarise weights them?

Run:
    uv run python scripts/check_t_true.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("reports/eda/datasets.csv")
ALPHA = 0.1  # metrics.ADJUSTMENT_ALPHA


def multiplier(n_pred, t_true, alpha: float = ALPHA):
    return np.maximum(0.0, 1.0 - alpha * (n_pred - t_true) / t_true)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(CSV)
    t = d[(d.split == "train") & (d.t_true > 0)].reset_index(drop=True)
    q = pd.DataFrame([json.loads(s) for s in t.quantile_values]).reset_index(drop=True)
    q = q.apply(pd.to_numeric, errors="coerce")
    return t, q


def fit_report(name: str, X: np.ndarray, y_log: np.ndarray, t_true: np.ndarray) -> dict:
    """Least-squares fit in log space, reported as relative error on the raw scale.

    Log space because T_true spans 21x; absolute errors there would be dominated
    by the largest datasets, and what matters is the ratio N_pred / T_true.
    """
    beta, *_ = np.linalg.lstsq(X, y_log, rcond=None)
    resid = y_log - X @ beta
    r2 = 1.0 - (resid**2).sum() / ((y_log - y_log.mean()) ** 2).sum()
    pred = np.exp(X @ beta)
    rel = np.abs(pred - t_true) / t_true
    return {
        "name": name,
        "n_features": X.shape[1],
        "r2_log": r2,
        "median_rel_err": float(np.median(rel)),
        "p90_rel_err": float(np.percentile(rel, 90)),
        "frac_within_25pct": float((rel <= 0.25).mean()),
        "beta": beta,
        "pred": pred,
    }


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"not found: {CSV.resolve()} -- run explore_data.py first")

    t, q = load()
    t_true = t.t_true.to_numpy(dtype=float)
    y = np.log(t_true)
    ones = np.ones(len(t))
    is_44b6 = (t.prefix == "44b6").astype(float).to_numpy()

    def lg(col):
        return np.log(q[col].to_numpy(dtype=float) + 1.0)

    print(f"n = {len(t)} training datasets")
    print(f"T_true: min {t_true.min():,.0f}  median {np.median(t_true):,.0f}  "
          f"max {t_true.max():,.0f}  spread {t_true.max()/t_true.min():.1f}x\n")

    print("=== predicting T_true ===")
    models = [
        ("family constant only", np.column_stack([is_44b6, ones])),
        ("p90 only", np.column_stack([lg("0.9"), ones])),
        ("p10 + p90", np.column_stack([lg("0.1"), lg("0.9"), ones])),
        ("p10 + p90 + family", np.column_stack([lg("0.1"), lg("0.9"), is_44b6, ones])),
        ("p10 + p90 + p99 + family",
         np.column_stack([lg("0.1"), lg("0.9"), lg("0.99"), is_44b6, ones])),
        ("all quantiles + family",
         np.column_stack([lg("0.001"), lg("0.01"), lg("0.1"), lg("0.9"),
                          lg("0.99"), lg("0.999"), is_44b6, ones])),
    ]
    results = [fit_report(n, X, y, t_true) for n, X in models]
    print(f"{'model':30} {'k':>2} {'R2':>6} {'med err':>8} {'p90 err':>8} {'<=25%':>7}")
    for r in results:
        print(f"{r['name']:30} {r['n_features']:2d} {r['r2_log']:6.3f} "
              f"{r['median_rel_err']:7.1%} {r['p90_rel_err']:7.1%} "
              f"{r['frac_within_25pct']:6.1%}")

    # Family median as a non-parametric baseline, no fitting involved.
    fam_med = t.groupby("prefix").t_true.transform("median").to_numpy(dtype=float)
    rel_fam = np.abs(fam_med - t_true) / t_true
    print(f"\n{'family median (no fit)':30} {'-':>2} {'-':>6} "
          f"{np.median(rel_fam):7.1%} {np.percentile(rel_fam, 90):7.1%} "
          f"{(rel_fam <= 0.25).mean():6.1%}")

    # Weighting: metrics.summarise weight-averages adj_edge_jaccard by
    # w_i = TP_i + FP_i + FN_i. TP + FN equals the GT edge count, so n_edges is
    # the usable proxy here -- it is a lower bound on w_i.
    w = t.n_edges.to_numpy(dtype=float)

    print("\n=== cost of one global node count ===")
    print("If a single det_threshold produced the same N_pred on every dataset:")
    print(f"{'N_pred':>12} {'mean mult':>10} {'wtd mult':>9} {'min':>6} "
          f"{'#<0.9':>6} {'#zeroed':>8}")
    for target in [t_true.min(), 5000, 10000, np.median(t_true), 30000,
                   50000, t_true.max()]:
        m = multiplier(target, t_true)
        print(f"{target:12,.0f} {m.mean():10.3f} {(m*w).sum()/w.sum():9.3f} "
              f"{m.min():6.3f} {(m < 0.9).sum():6d} {(m == 0).sum():8d}")

    print("\n=== cost of a per-dataset predicted target ===")
    best = max(results, key=lambda r: r["frac_within_25pct"])
    for label, pred in [("family median", fam_med), (best["name"], best["pred"])]:
        m = multiplier(pred, t_true)
        print(f"{label:28} mean mult {m.mean():.3f}  "
              f"weighted {(m*w).sum()/w.sum():.3f}  min {m.min():.3f}  "
              f"zeroed {(m == 0).sum()}")

    print("\n=== does edge weighting shield the sparse datasets? ===")
    rho = float(t[["t_true", "n_edges"]].corr(method="spearman").iloc[0, 1])
    print(f"spearman(T_true, n_edges) = {rho:.3f}")
    lo = t_true <= np.percentile(t_true, 25)
    share_lo = w[lo].sum() / w.sum()
    print(f"bottom quartile of T_true holds {share_lo:.1%} of total edge weight")
    # Derive the conclusion from the numbers rather than asserting a direction.
    # Sensitivity to the penalty scales as 1/T_true, so low-T_true datasets are
    # the exposed ones; the question is whether they also carry weight.
    if rho < 0:
        print("T_true and annotated edge count move in opposite directions, so the")
        print("datasets most exposed to the penalty (small T_true) are the ones")
        print("carrying the most edge weight. Edge weighting does not shield them.")
    else:
        print("T_true and annotated edge count move together, so the datasets most")
        print("exposed to the penalty (small T_true) also carry the least weight.")
    if share_lo > 0.25:
        print(f"The exposed quartile carries {share_lo:.1%} of the weight, above its")
        print("25% share of datasets: per-dataset N_pred calibration matters most there.")


if __name__ == "__main__":
    main()
