"""Does the fork RANKING bind, and can anything separate the pool it ranks?

`add_safe_divisions` admits a candidate only after four structural gates, then
orders the survivors by `d_par + 0.15*d_sis` and truncates at a per-frame cap.
CLAUDE.md closed the division term by elimination -- every gate tested, none
binds -- and landed on "the forks are admissible but out-ranked". That
conclusion rests on two measurements which cannot both be right:

* loosening the divergence gate lost *monotonically* (divJ 0.0816 -> 0.0521),
  which can only happen if false forks outrank true ones under a binding cap;
* scaling both caps by 1.25x measured neutral (divJ 0.0822 -> 0.0800), which
  says the cap barely binds at all.

Nobody has looked at the pool directly. `VOLT_DIV_DUMP=<csv>` makes the pass
record every admitted candidate with its features, its rank and whether the cap
let it through; this script labels those rows against GT and answers three
questions with numbers instead of inference:

1. **Does the cap bind?** How often does the frame hold more candidates than
   the cap admits, and what share of candidates is truncated.
2. **Where do the true forks rank?** If they sit near the top and are admitted,
   the ranking is not the constraint and the remaining loss is elsewhere.
3. **Can any feature separate?** AUC per feature over the *admitted pool* --
   which is the only population a better ranker could reorder. The 2026-09-12
   study that found geometric features non-separating measured a different
   population (every proposal behind a 6 um gate, n=3 real forks).

Usage:
    VOLT_DIV_DUMP=$T/divpool_f0.csv uv run python scripts/score_submission.py ...
    uv run python scripts/division_candidate_audit.py \
        --dump $T/divpool_f0.csv --log $T/logs/<that run>.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataspec import DATASET_PATH
from volt_biohub.scoring import ORIGINAL_SCALE, load_gt

TOL_UM = 7.0          # the metric's own node-matching tolerance
FRAME_TOL = 1         # the division metric's +/-1 timepoint tolerance


def video_order(log_path: Path) -> list[str]:
    """Video names in the order the scoring run processed them.

    The dump's `call` counter increments once per `add_safe_divisions` call and
    the pass runs once per video, so call k is the k-th name here. Reading the
    log avoids plumbing a name through `run_all` for a diagnostic.
    """
    pat = re.compile(r"^\s*\[\s*\d+/\s*\d+\]\s+(\S+)")
    names = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = pat.match(line)
        if m:
            names.append(m.group(1))
    return names


def gt_divisions(name: str, data_dir: Path) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """(t, divider_um, daughter1_um, daughter2_um) for every GT division."""
    gt = load_gt(data_dir, name)
    df = gt.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    ids = df["node_id"].to_list()
    t = df["t"].to_numpy().astype(np.int64)
    zyx = np.stack([df["z"].to_numpy(), df["y"].to_numpy(), df["x"].to_numpy()], axis=1)
    um = zyx.astype(np.float64) * np.asarray(ORIGINAL_SCALE)
    by_id = {i: (int(t[k]), um[k]) for k, i in enumerate(ids)}

    out = []
    for node in ids:
        kids = list(gt.successors(node))
        if len(kids) != 2:
            continue
        if node not in by_id or any(c not in by_id for c in kids):
            continue
        tt, p = by_id[node]
        out.append((tt, p, by_id[kids[0]][1], by_id[kids[1]][1]))
    return out


def label(rows: pl.DataFrame, divs, iso_um: float) -> np.ndarray:
    """1 if a candidate reconstructs a GT division, else 0.

    Strict on purpose: the parent must match the GT divider AND the two
    daughters must match the GT daughters in either pairing, all within the
    metric's own 7 um tolerance. A parent-only match would count a candidate
    that attaches the wrong second daughter to a real divider, which is exactly
    the error a better ranker is supposed to avoid making.
    """
    if not len(rows) or not divs:
        return np.zeros(len(rows), dtype=np.int8)
    p = rows.select(["pz", "py", "px"]).to_numpy() * iso_um
    c = rows.select(["cz", "cy", "cx"]).to_numpy() * iso_um
    h = rows.select(["hz", "hy", "hx"]).to_numpy() * iso_um
    t = rows["t"].to_numpy()

    y = np.zeros(len(rows), dtype=np.int8)
    for tt, gp, g1, g2 in divs:
        near_t = np.abs(t - tt) <= FRAME_TOL
        if not near_t.any():
            continue
        dp = np.linalg.norm(p - gp, axis=1)
        ok = near_t & (dp <= TOL_UM)
        if not ok.any():
            continue
        d_c1 = np.linalg.norm(c - g1, axis=1)
        d_c2 = np.linalg.norm(c - g2, axis=1)
        d_h1 = np.linalg.norm(h - g1, axis=1)
        d_h2 = np.linalg.norm(h - g2, axis=1)
        pair_a = (d_c1 <= TOL_UM) & (d_h2 <= TOL_UM)
        pair_b = (d_c2 <= TOL_UM) & (d_h1 <= TOL_UM)
        y[ok & (pair_a | pair_b)] = 1
    return y


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC, higher score = more likely positive. NaN-safe.

    Ties take mid-ranks. Without that a constant feature reports a confident
    0.0 or 1.0 rather than the 0.5 it deserves, because argsort breaks ties by
    position -- which is exactly what the first version of this function did to
    `has_grandchildren`, a column the gates force to a single value.
    """
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    pos, neg = s[yy == 1], s[yy == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--reject", type=Path, default=None,
                    help="CSV from VOLT_DIV_REJECT: every orphan inside the "
                         "sister gate with the first gate that refused it. This "
                         "is what attributes the GT divisions that never reach "
                         "the pool at all -- the dump alone records survivors.")
    ap.add_argument("--data-dir", type=Path, default=DATASET_PATH)
    ap.add_argument("--iso-um", type=float, default=1.625)
    args = ap.parse_args()

    df = pl.read_csv(args.dump)
    names = video_order(args.log)
    print(f"{len(df)} admitted candidates over {df['call'].n_unique()} videos "
          f"({len(names)} in the log)")

    # --- 1. does the cap bind ---
    per_frame = df.group_by(["call", "t"]).agg(
        pl.col("n_props").first(), pl.col("frame_cap").first(),
        pl.col("accepted").sum().alias("n_acc"))
    over = per_frame.filter(pl.col("n_props") > pl.col("frame_cap"))
    tot_props = int(per_frame["n_props"].sum())
    tot_acc = int(per_frame["n_acc"].sum())
    print("\n--- 1. does the per-frame cap bind ---")
    print(f"frames with candidates      {per_frame.height}")
    print(f"frames where n_props > cap  {over.height} "
          f"({100 * over.height / max(1, per_frame.height):.1f} %)")
    print(f"candidates admitted         {tot_props}")
    print(f"candidates accepted         {tot_acc} "
          f"({100 * tot_acc / max(1, tot_props):.1f} %)")
    print(f"median n_props / frame_cap  {per_frame['n_props'].median()} / "
          f"{per_frame['frame_cap'].median()}")

    # --- 2/3. label against GT ---
    labels = []
    n_gt = 0
    for call, sub in df.group_by("call", maintain_order=True):
        idx = int(call[0]) - 1
        if idx < 0 or idx >= len(names):
            labels.append(np.zeros(len(sub), dtype=np.int8))
            continue
        try:
            divs = gt_divisions(names[idx], args.data_dir)
        except Exception as exc:                      # a missing geff is not fatal here
            print(f"  ! {names[idx]}: {exc}")
            divs = []
        n_gt += len(divs)
        labels.append(label(sub, divs, args.iso_um))
    df = df.sort("call").with_columns(pl.Series("y", np.concatenate(labels)))

    # The headroom of the whole proposal-limited line. If most GT divisions are
    # already represented in the pool, widening candidate generation cannot pay
    # however free the false ones are -- so this ratio, not the cap statistics,
    # is what bounds the line. Counted only over videos that produced at least
    # one candidate, since a video contributing no rows is invisible here.
    n_pos = int(df["y"].sum())
    print("\n--- headroom: GT divisions vs the pool ---")
    print(f"GT divisions in the audited videos   {n_gt}")
    print(f"represented by an admitted candidate {n_pos} "
          f"({100 * n_pos / max(1, n_gt):.1f} %)")
    print(f"MISSING from the pool entirely       {max(0, n_gt - n_pos)}")
    print(f"\n--- 2. where do the {n_pos} TRUE forks sit in the ranking ---")
    if n_pos:
        pos = df.filter(pl.col("y") == 1)
        print(f"accepted by the cap   {int(pos['accepted'].sum())} / {n_pos}")
        print(f"rank    p25/p50/p75   {pos['rank'].quantile(0.25)} / "
              f"{pos['rank'].median()} / {pos['rank'].quantile(0.75)}")
        print(f"frame_cap p50         {pos['frame_cap'].median()}")
        print(f"n_props   p50         {pos['n_props'].median()}")
        print("\nrejected true forks by reason:")
        rej = pos.filter(pl.col("accepted") == 0)
        print(f"  rank >= frame_cap   {rej.filter(pl.col('rank') >= pl.col('frame_cap')).height}")
        print(f"  other (global cap / claimed elsewhere)  "
              f"{rej.filter(pl.col('rank') < pl.col('frame_cap')).height}")

    print("\n--- 3. can any feature separate the ADMITTED pool ---")
    y = df["y"].to_numpy()
    feats = {
        "shipped ranking (-score)": -df["rank_score"].to_numpy(),
        "-d_par": -df["d_par"].to_numpy(),
        "-d_sis": -df["d_sis"].to_numpy(),
        "-d_exist": -df["d_exist"].to_numpy(),
        "-symmetry": -df["symmetry"].to_numpy(),
        "diverge": df["diverge"].to_numpy(),
        "has_grandchildren": df["has_grandchildren"].to_numpy().astype(float),
        "-n_props (frame crowding)": -df["n_props"].to_numpy().astype(float),
    }
    for k, v in feats.items():
        print(f"  AUC {auc(v.astype(float), y):.3f}   {k}")

    out = args.dump.with_suffix(".labelled.csv")
    df.write_csv(out)
    print(f"\nlabelled pool -> {out}")

    if args.reject is not None and args.reject.exists():
        # "nan" has to be declared as null, or polars types learned_prob as str
        # and every comparison on it fails at the first quantile.
        rj = pl.read_csv(args.reject, null_values=["nan"])
        rlabels = []
        for call, sub in rj.group_by("call", maintain_order=True):
            idx = int(call[0]) - 1
            divs = (gt_divisions(names[idx], args.data_dir)
                    if 0 <= idx < len(names) else [])
            rlabels.append(label(sub, divs, args.iso_um))
        rj = rj.sort("call").with_columns(pl.Series("y", np.concatenate(rlabels)))
        true_rows = rj.filter(pl.col("y") == 1)
        print(f"\n--- 4. what refuses the TRUE second daughter "
              f"({true_rows.height} considered) ---")
        if true_rows.height:
            for r in (true_rows.group_by("reason")
                      .agg(pl.len().alias("n"), pl.col("k_rank").median())
                      .sort("n", descending=True).iter_rows(named=True)):
                print(f"  {r['n']:4d}  {r['reason']:<18} "
                      f"median k_rank {r['k_rank']}")
            if "learned_prob" in rj.columns:
                # The one signal geometry does not already contain. It has to
                # separate the ~12 true daughters the d_par and symmetry gates
                # refuse from the thousands of false candidates those same
                # gates are holding back -- that is the whole test.
                blocked = true_rows.filter(pl.col("reason").is_in(["d_par", "symmetry"]))
                false_blocked = rj.filter(
                    (pl.col("y") == 0) & pl.col("reason").is_in(["d_par", "symmetry"]))
                print("\n  learned_prob (edge head, discarded by the greedy pass):")
                for nm, d in (("true, gate-refused", blocked),
                              ("true, would pass", true_rows.filter(pl.col("reason") == "pass")),
                              ("false, gate-refused", false_blocked)):
                    v = d["learned_prob"].drop_nulls()
                    if v.len():
                        print(f"    {nm:<22} n={d.height:5d}  recorded={v.len():5d}  "
                              f"p50 {v.median():.4f}  p90 {v.quantile(0.9):.4f}")
                    else:
                        print(f"    {nm:<22} n={d.height:5d}  recorded=0")
                both = pl.concat([blocked, false_blocked])
                a = auc(both["learned_prob"].fill_null(0.0)
                        .to_numpy().astype(float), both["y"].to_numpy())
                print(f"    AUC true-vs-false among GATE-REFUSED: {a:.3f}   "
                      f"(NaN -> 0.0, i.e. outside the recorded top-5)")

            reachable = true_rows.filter(pl.col("reason") == "pass")
            print(f"\n  would PASS every gate: {reachable.height}")
            if reachable.height:
                ks = reachable["k_rank"]
                print(f"  their k_rank p50/p90/max: {ks.median()} / "
                      f"{ks.quantile(0.9)} / {ks.max()}   "
                      f"<- n_cand must exceed this to reach them")
        rout = args.reject.with_suffix(".labelled.csv")
        rj.write_csv(rout)
        print(f"\nlabelled rejections -> {rout}")


if __name__ == "__main__":
    main()
