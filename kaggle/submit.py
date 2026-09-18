"""Offline inference: a directory of test ``.zarr`` videos -> a validated ``submission.csv``.

This is the whole submission chain. The notebook is glue that calls ``main()``.

Why this script exists rather than ``scripts/predict_unet_transformer.py``:

* that script imports tracksdata at module scope, through ``volt_biohub.io``,
  ``volt_biohub.metrics``, ``evaluate`` and ``train_unet_transformer`` — so it
  cannot be imported at all in the offline notebook (its ``--test-dir`` mode does
  now enumerate a directory, but that is a local probe, not the graded path);
* it emits one ``.geff`` per video and then needs ``geffs_to_csv.py`` to flatten
  them; that writer is tracksdata-free now, but every other module in that
  script's closure needs tracksdata in some branch, and none of them are of any
  use to the notebook;
* it runs the learned edge transformer, which currently emits zero edges and is
  measured worse than plain Hungarian assignment on micrometre distance (see the
  module docstring of ``volt_biohub.detectors``, and the LOEO numbers below).

So the offline path is: enumerate the directory, detect per frame, link
geometrically, write rows. The best configuration measured to date is the
default here — U-Net detector, 8 um gate, division pass off, budget 1.0x the
predicted node count — which scored 0.6925 on LOEO fold 0 against the classical
floor's 0.6424. Omitting ``--weights`` falls back to the classical DoG detector,
which needs nothing else and always produces a valid file.

Node budget at inference. Both detectors take a node budget rather than a
threshold, and the budget's denominator ``estimated_number_of_nodes`` exists only
in the ground-truth ``.geff`` — the test set ships images alone. It is however
predictable from ``image_statistics.quantiles``, which the test images *do*
carry: see :func:`predict_n_true`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import zarr

from volt_biohub.detectors import (
    budget_for,
    detect_dog,
    detect_from_map,
    predict_n_true,
)
from volt_biohub.gpu import resolve_device

# volt_biohub.linking is importable offline: its only tracksdata/polars users are
# inside build_graph(), which assembles a graph for the local scorer and is never
# called from here. check_offline.py is the gate that keeps that true — it imports
# this whole path, and runs it, with tracksdata hidden behind a meta_path blocker.
from volt_biohub.linking import DIVISION_GATE_UM, MOTION_GATE_UM, link_sequence

DOWNSAMPLE = (1, 4, 4)
"""The 1,4,4 grid is exactly isotropic: 0.40625 * 4 == 1.625, the z spacing.

Must match the checkpoint's own ``downsample`` — a mismatch changes what a
micrometre means to every gate downstream, so :func:`load_detector` checks it.
"""

ISO_UM = 1.625

DEFAULT_SEPARATION_UM = 4.0
"""Peak suppression radius. **Was 2*sigma = 5.86 um, and that cost 0.08 of score on
both submissions we have made.**

Every tuned CV number in this repo was measured at 4.0; this default was 5.86; and
`scripts/score_submission.py` -- the harness written specifically to catch exactly this
kind of CV/submission divergence -- did not pass the parameter, so it inherited 5.86 too
and agreed with the leaderboard for the wrong reason. Measured with the public
400-epoch weights on 8 fold-1 videos through the submission path:

| separation | score | edge J | node recall |
|------------|-------|--------|-------------|
| 5.86       | 0.7233 | 0.6996 | 0.9151 |
| **4.0**    | **0.8041** | 0.7851 | **0.9762** |
| 3.0        | 0.7751 | 0.7593 | 0.9835 |
| 2.0        | 0.7751 | 0.7593 | 0.9835 |

Below 4.0 node recall keeps rising and the score falls -- metric fact 3, more matched
nodes exposing more of the linker's mistakes to scoring. 3.0 and 2.0 are identical
because `_peak_window` takes `ceil(sep/1.625)`, so both give a 3-voxel window; sweep
distinct windows, not round numbers.

The 2.87 um sister-separation floor still applies as a hard lower bound.
"""



def _load_submission_module():
    """Import the sibling ``submission.py`` by path, under a distinctive name.

    Not ``import submission``: this directory is named ``kaggle``, which is also
    the Kaggle API package, and on the notebook ``/kaggle`` is the working root —
    so whether a bare or a package-relative import resolves to our file depends on
    ``sys.path`` order we do not control. Loading by absolute path removes the
    ambiguity, and it keeps ``kaggle/`` from having to be a package.
    """
    import importlib.util
    import sys

    name = "volt_submission"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("submission.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_sub = _load_submission_module()
dataset_rows = _sub.dataset_rows
write_validated = _sub.write_validated


def resolve_test_dir(explicit: Path | None = None) -> Path:
    """Find the directory of test ``.zarr`` folders.

    Order: an explicit ``--test-dir``, then ``$VOLT_TEST_DIR``, then the Kaggle
    competition mount, then ``test/`` beside ``$VOLT_DATA_DIR``. Never reads the
    dataset list out of ``sample_submission.csv``: those four names are local
    placeholders and are replaced at notebook rerun.
    """
    if explicit is not None:
        return explicit
    if "VOLT_TEST_DIR" in os.environ:
        return Path(os.environ["VOLT_TEST_DIR"])
    for mount in sorted(Path("/kaggle/input").glob("*/test")):
        return mount
    if "VOLT_DATA_DIR" in os.environ:
        return Path(os.environ["VOLT_DATA_DIR"]).parent / "test"
    raise SystemExit(
        "cannot find the test directory: pass --test-dir, or set VOLT_TEST_DIR"
    )


def list_datasets(test_dir: Path) -> list[str]:
    """Every ``*.zarr`` in *test_dir*, sorted, without the extension."""
    names = sorted(p.stem for p in test_dir.glob("*.zarr"))
    if not names:
        raise SystemExit(f"no *.zarr found in {test_dir}")
    return names


def load_volumes(zarr_path: Path) -> tuple[np.ndarray, float]:
    """Normalised frames on the isotropic grid, plus the stored 0.9 quantile.

    Normalisation must match training exactly or the detector measures the
    mismatch rather than the image. Both ``train_unet_transformer.py`` and
    ``predict_unet_transformer.py`` bypass ``io.open_dataset(normalize=True)``
    (whose min/max default compresses real signal) and normalise from the stored
    0.001/0.999 quantiles with a lower clamp and no upper clamp. This reproduces
    that. It duplicates ``volt_biohub.scoring.load_volumes``, which cannot be
    imported here because that module pulls in tracksdata; the two must not
    drift, and the right home is a tracksdata-free ``volt_biohub`` image module.
    """
    g = zarr.open_group(str(zarr_path), mode="r")
    stats = g.attrs.get("image_statistics")
    if not stats or "quantiles" not in stats:
        # train and predict both raise here. All 203 downloaded samples carry
        # quantiles, so this is latent — but a hard crash mid-notebook is the
        # worst outcome, so fall back to the volume's own percentiles instead.
        arr = np.asarray(g["0"][0]).astype(np.float32)
        lo, hi = np.percentile(arr, [0.1, 99.9])
        print(f"    WARNING: no image_statistics; using per-volume percentiles "
              f"{lo:.1f}/{hi:.1f}", flush=True)
        q90 = float(np.percentile(arr, 90))
    else:
        q = stats["quantiles"]
        lo, hi = float(q["0.001"]), float(q["0.999"])
        q90 = float(q["0.9"])

    dz, dy, dx = DOWNSAMPLE
    raw = np.asarray(g["0"][:, ::dz, ::dy, ::dx]).astype(np.float32)
    return np.clip((raw - lo) / (hi - lo + 1e-6), 0, None), q90


def load_detector(weights: Path, device, tta: bool = False):
    """Rebuild just the detection half of the trained model.

    The checkpoint is a joint ``UNetNodeTransformer`` state dict: ``unet.*``,
    ``detect_head.*`` and ``transformer.*``. Only the first two are loaded, both
    because the geometric linker replaces the transformer and because leaving it
    out keeps the weights dataset small. ``scripts/train_unet_transformer.py``
    cannot be imported here to reuse its ``detect()`` — it imports tracksdata —
    so the four lines of head-plus-forward are reproduced.
    """
    import torch
    import torch.nn as nn

    from volt_biohub.models import TemporalUNet3D

    config_path = weights.parent / "config.json"
    if not config_path.exists():
        raise SystemExit(
            f"no config.json beside {weights}. The weights dataset must carry the "
            "config next to the checkpoint — architecture is not inferable from a "
            "state dict alone."
        )
    config = json.loads(config_path.read_text())
    if tuple(config.get("downsample", DOWNSAMPLE)) != DOWNSAMPLE:
        raise SystemExit(
            f"checkpoint trained at downsample {config['downsample']} but this "
            f"script assumes {DOWNSAMPLE}; every micrometre gate downstream would "
            "be wrong."
        )

    out_ch = config["unet_out_channels"]
    unet = TemporalUNet3D(in_channels=1, out_channels=out_ch, layers=config["unet_layers"])
    head = nn.Conv3d(out_ch, 1, kernel_size=1)

    state = torch.load(weights, map_location="cpu", weights_only=True)
    unet.load_state_dict({k[len("unet."):]: v for k, v in state.items()
                          if k.startswith("unet.")})
    head.load_state_dict({k[len("detect_head."):]: v for k, v in state.items()
                          if k.startswith("detect_head.")})
    unet.to(device).eval()
    head.to(device).eval()

    def _forward(t: torch.Tensor, want_features: bool):
        pair = torch.stack([t, t], dim=0).unsqueeze(0).unsqueeze(2)
        feats = unet(pair)
        logits = head(feats[0, 0:1])[0, 0].float()
        return (logits, feats[0, 0]) if want_features else (logits, None)

    def detect_map(frame: np.ndarray, want_features: bool = False):
        """Detection **logits** for one frame, at the input resolution.

        Logits, not a sigmoid: a sigmoid flattens the ranking the peak finder
        needs. The frame is duplicated into a fake pair because the U-Net is
        temporal and wants two frames — measured to cost nothing (node recall
        0.9967 from the real pair against 0.9973 from the fake one, because the
        temporal attention degenerates cleanly on two identical tokens).

        *want_features* additionally returns the 32-channel trunk features the
        learned edge head indexes at detected coordinates. They are returned per
        frame rather than per video on purpose: 32 channels x 64^3 x 100 frames
        is 3.4 GB, while two frames is 67 MB.

        **TTA is averaged in logit space over the D4 symmetries of the (y, x)
        plane only.** Averaging sigmoids would compress exactly the ranking the
        peak finder reads, and z is excluded because the microscope's PSF is
        elongated in z — a z flip is out of distribution even though the
        downsampled grid is geometrically isotropic. Rotations need y == x, which
        holds for every sample measured (64 x 64 after the 1,4,4 stride), so a
        differently shaped hidden sample falls back to the two flips rather than
        raising. Features are returned from the identity view: they feed the edge
        head's coordinate indexing, which no symmetry average is defined for.
        """
        with torch.no_grad():
            t = torch.from_numpy(frame).to(device)
            if not tta:
                logits, feats = _forward(t, want_features)
                logits = logits.cpu().numpy()
                return (logits, feats) if want_features else logits

            square = t.shape[-1] == t.shape[-2]
            acc = None
            feats = None
            for k in (0, 1, 2, 3) if square else (0,):
                rot = torch.rot90(t, k, dims=(-2, -1)) if k else t
                for flip in (False, True):
                    view = torch.flip(rot, dims=(-1,)) if flip else rot
                    out, f = _forward(view, want_features and k == 0 and not flip)
                    if f is not None:
                        feats = f
                    if flip:
                        out = torch.flip(out, dims=(-1,))
                    if k:
                        out = torch.rot90(out, -k, dims=(-2, -1))
                    acc = out if acc is None else acc + out
            n_views = 8 if square else 2
            logits = (acc / n_views).cpu().numpy()
            return (logits, feats) if want_features else logits

    return detect_map


def detect_video(
    volumes: np.ndarray,
    n_per_frame: int,
    separation_um: float,
    detect_map=None,
    refine: bool = False,
    threshold: float | None = None,
) -> list[np.ndarray]:
    """Peaks per frame: from the learned head when given one, else from DoG.

    *threshold* switches the learned path off the node budget — see
    ``detectors.detect_from_map``. The budget makes detection purely
    budget-limited, and the predictor behind it is a fit on two embryos applied to
    an unseen third.
    """
    if detect_map is None:
        return [
            detect_dog(v, n_per_frame, ISO_UM, min_separation_um=separation_um,
                       refine=refine)
            for v in volumes
        ]
    return [
        detect_from_map(detect_map(v), n_per_frame, ISO_UM, separation_um,
                        refine=refine, threshold=threshold)
        for v in volumes
    ]


def detect_and_link_learned(
    volumes: np.ndarray,
    detect_map,
    edge_head,
    n_per_frame: int,
    separation_um: float,
    downsample: tuple[int, int, int],
    refine: bool = False,
    threshold: float | None = None,
    edge_threshold: float = 0.5,
) -> tuple[list[np.ndarray], list[list[tuple[int, int, float]]]]:
    """Detect and link in one pass, keeping only two frames of features live.

    Interleaved rather than detect-then-link because the edge head needs the
    trunk features at both endpoints of every pair, and holding a whole video's
    features costs 3.4 GB against 67 MB for a sliding pair.
    """
    from volt_biohub.edge_head import link_pair

    window_shape = (2, *volumes.shape[1:])
    coords_per_t: list[np.ndarray] = []
    links_per_t: list[list[tuple[int, int, float]]] = []
    prev_feat = None

    for t, frame in enumerate(volumes):
        logits, feat = detect_map(frame, want_features=True)
        coords = detect_from_map(logits, n_per_frame, ISO_UM, separation_um,
                                refine=refine, threshold=threshold)
        coords_per_t.append(coords)
        if t > 0:
            links_per_t.append(link_pair(
                edge_head, prev_feat, feat, coords_per_t[t - 1], coords,
                t_src=0, window_shape=window_shape, downsample=downsample,
                threshold=edge_threshold,
            ))
        prev_feat = feat
    return coords_per_t, links_per_t


def run(
    test_dir: Path,
    out_path: Path,
    *,
    weights: Path | None = None,
    budget: float = 1.0,
    separation_um: float = DEFAULT_SEPARATION_UM,
    gate_um: float = MOTION_GATE_UM,
    division_gate_um: float | None = DIVISION_GATE_UM,
    refine: bool = False,
    limit: int | None = None,
    switch_at: float | None = None,
    det_threshold: float | None = None,
    learned_edges: bool = False,
    edge_threshold: float = 0.5,
    postprocess: bool = False,
    min_track_len: int = 6,
    tta: bool = False,
) -> Path:
    """Predict every video in *test_dir* and write a validated submission."""
    names = list_datasets(test_dir)
    if limit:
        names = names[:limit]

    detect_map = None
    edge_head = None
    if weights is not None:
        device = resolve_device()
        print(f"detector: unet on {device} ({weights})", flush=True)
        detect_map = load_detector(weights, device, tta=tta)
        if learned_edges:
            import json as _json

            from volt_biohub.edge_head import load_edge_head
            cfg = _json.loads((weights.parent / "config.json").read_text())
            edge_head = load_edge_head(weights, cfg, device)
            # None means the checkpoint has no transformer.* weights, so fall
            # back to the geometric linker rather than dying at hour ten.
            which = ("learned edge head" if edge_head is not None
                     else "geometric (no transformer.* in checkpoint)")
            print(f"linker: {which}", flush=True)
    else:
        print("detector: dog (no weights)", flush=True)

    if switch_at is not None and detect_map is None:
        # A switch with no learned detector would silently be plain DoG, which
        # is the one failure mode that looks like success in the log.
        raise ValueError("--switch-at needs --weights: the switch picks between "
                         "the U-Net and DoG, and without weights there is no U-Net")
    if switch_at is not None:
        print(f"density switch: U-Net at >= {switch_at:.0f} predicted cells/frame, "
              f"else DoG", flush=True)
    print(f"{len(names)} datasets · budget {budget}x predicted N_true · "
          f"sep {separation_um:.2f} um · gate {gate_um} um · "
          f"div gate {'off' if division_gate_um is None else division_gate_um}",
          flush=True)

    rows: list[dict] = []
    t0 = time.monotonic()
    for i, name in enumerate(names, 1):
        t1 = time.monotonic()
        volumes, q90 = load_volumes(test_dir / f"{name}.zarr")
        n_true = predict_n_true(q90)
        n_per_frame = budget_for(n_true, len(volumes), budget)

        # Measured over all 199 training samples: the U-Net wins on the CROWDED
        # samples and DoG on the sparse ones, spearman(cells/frame, unet-dog) =
        # +0.700. Switching at 150 scores 0.7366 weighted against 0.6789 for the
        # better single detector, taking 91% of the per-sample oracle's 0.7433.
        # The switch runs on PREDICTED density (N_true is absent at test time);
        # simulating that predictor's real 25% error over 400 draws, the worst
        # draw still scored 0.7297, and the response is flat from 125 to 175.
        use_map = detect_map
        picked = "unet" if detect_map is not None else "dog"
        if switch_at is not None:
            crowded = (n_true / max(len(volumes), 1)) >= switch_at
            use_map = detect_map if crowded else None
            picked = "unet" if crowded else "dog"
        learned_links = None
        if edge_head is not None and use_map is not None:
            coords, learned_links = detect_and_link_learned(
                volumes, use_map, edge_head, n_per_frame, separation_um,
                DOWNSAMPLE, refine=refine, threshold=det_threshold,
                edge_threshold=edge_threshold,
            )
        else:
            coords = detect_video(volumes, n_per_frame, separation_um, use_map,
                                  refine, threshold=det_threshold)
        if not any(len(c) for c in coords):
            # Every test dataset must be *present* in the submission, and a
            # dataset that detected nothing emits no rows at all — absent, not
            # empty. One placeholder node at the volume centre keeps it present;
            # sample_submission.csv's own filler rows sit at the centre too. This
            # should be unreachable (the budget floors at 1 peak per frame), so
            # reaching it means something is wrong upstream — hence the warning.
            print("    WARNING: no detections; emitting one placeholder node",
                  flush=True)
            centre = np.array([[s // 2 for s in volumes.shape[1:]]], dtype=np.int64)
            coords = [centre] + [np.empty((0, 3), dtype=np.int64)] * (len(volumes) - 1)

        if postprocess:
            # Replaces the one-pass fixed-gate linker entirely: two-pass Hungarian
            # with velocity extrapolation, single-frame gap closing, structurally
            # gated divisions, short-track filtering and line-fit smoothing, in
            # the order the reference uses (see volt_biohub.postprocess.run_all).
            from volt_biohub.postprocess import run_all
            coords, links = run_all(
                coords, ISO_UM,
                divisions=division_gate_um is not None,
                min_track_len=min_track_len,
                shape=tuple(volumes.shape[1:]),
            )
        elif learned_links is not None:
            links = learned_links
        else:
            links = link_sequence(
                coords, (ISO_UM, ISO_UM, ISO_UM),
                gate_um=gate_um, division_gate_um=division_gate_um)
        rows.extend(dataset_rows(name, coords, links, upsample=DOWNSAMPLE))

        n_nodes = sum(len(c) for c in coords)
        n_edges = sum(len(lk) for lk in links)
        dt = time.monotonic() - t1
        eta = (time.monotonic() - t0) / i * (len(names) - i)
        print(f"  [{i:3d}/{len(names)}] {name:16s} {picked:4s} {n_nodes:7d} nodes "
              f"{n_edges:7d} edges  N_true_hat={n_true:8.0f}  {dt:5.1f}s  "
              f"eta {eta / 60:5.1f}m", flush=True)

    # Validation reads the test zarrs for per-dataset volume bounds, so it needs
    # the directory rather than a shape argument.
    report = write_validated(rows, out_path, test_dir, validate=limit is None)
    # The report names the staging file it actually validated, so name the
    # destination separately rather than letting the two look inconsistent.
    print(f"\nwrote {out_path}\n{report}", flush=True)

    dt = time.monotonic() - t0
    print(f"{dt / 60:.1f} min total, {dt / max(len(names), 1):.1f} s/video "
          f"-> {dt / max(len(names), 1) * 199 / 3600:.2f} h for a train-sized test set "
          f"(12 h cap)", flush=True)
    return Path(out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--test-dir", type=Path, default=None,
                   help="Directory of test *.zarr folders. Auto-detected on Kaggle.")
    p.add_argument("--out", type=Path, default=Path("submission.csv"),
                   help="Must be named exactly submission.csv to be graded.")
    p.add_argument("--weights", type=Path, default=None,
                   help="Checkpoint for the U-Net detector. Omitted -> DoG.")
    p.add_argument("--budget", type=float, default=1.0,
                   help="Node count as a multiple of the predicted N_true.")
    p.add_argument("--separation-um", type=float, default=DEFAULT_SEPARATION_UM,
                   help="Minimum peak separation; hard floor 2.87 um (sister pairs).")
    p.add_argument("--gate-um", type=float, default=MOTION_GATE_UM,
                   help="Linking gate. Per-detector: swept, not derived.")
    p.add_argument("--division-gate-um", type=float, default=DIVISION_GATE_UM,
                   help="Second-child gate. Off by default: measured net loss ~0.025.")
    p.add_argument("--refine", action="store_true",
                   help="Response-weighted centroid, for the saturated map's plateaus.")
    p.add_argument("--det-threshold", type=float, default=None,
                   help="Sigmoid threshold for the learned detector. Given, it REPLACES "
                        "the node budget -- which is what caps node recall on samples "
                        "where predict_n_true under-shoots (0.50x and 0.54x on two of "
                        "the four test samples). The reference uses 0.5.")
    p.add_argument("--learned-edges", action="store_true",
                   help="Use the checkpoint's trained transformer edge head instead of "
                        "Hungarian assignment. The public checkpoint carries 580,353 "
                        "such parameters that load_detector was discarding; holding "
                        "detections fixed, this is worth +0.073.")
    p.add_argument("--edge-threshold", type=float, default=0.5,
                   help="Minimum softmax-over-sources probability for a learned edge.")
    p.add_argument("--postprocess", action="store_true",
                   help="Run the full post-detection stack from the public 0.942-0.948 "
                        "notebooks: two-pass Hungarian with velocity, gap closing, "
                        "structurally gated divisions, short-track filtering, line-fit "
                        "smoothing. Replaces --gate-um linking.")
    p.add_argument("--min-track-len", type=int, default=6,
                   help="Drop connected components smaller than this many NODES "
                        "(fork-containing components exempt). Reference uses 6.")
    p.add_argument("--tta", action="store_true",
                   help="Average detection logits over the 8 D4 symmetries of the "
                        "(y, x) plane. Every public notebook above 0.942 uses it; "
                        "8x the detector cost is affordable against the 12 h cap.")
    p.add_argument("--switch-at", type=float, default=None,
                   help="Predicted cells/frame above which to use the U-Net instead of "
                        "DoG (150 is the measured optimum; flat 125-175). Needs "
                        "--weights. Worth +0.058 weighted over the better single "
                        "detector on CV.")
    p.add_argument("--limit", type=int, default=None,
                   help="First N datasets only. A timing probe: the result is "
                        "deliberately left unvalidated, since a partial run is "
                        "missing datasets. Never use it for a submission.")
    args = p.parse_args(argv)

    run(
        resolve_test_dir(args.test_dir), args.out,
        weights=args.weights, budget=args.budget,
        separation_um=args.separation_um, gate_um=args.gate_um,
        division_gate_um=args.division_gate_um, refine=args.refine,
        limit=args.limit, switch_at=args.switch_at,
        det_threshold=args.det_threshold,
        learned_edges=args.learned_edges, edge_threshold=args.edge_threshold,
        postprocess=args.postprocess, min_track_len=args.min_track_len,
        tta=args.tta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
