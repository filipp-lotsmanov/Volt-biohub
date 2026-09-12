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
    DEFAULT_SIGMA_UM,
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

DEFAULT_SEPARATION_UM = 2.0 * DEFAULT_SIGMA_UM
"""~5.86 um. Hard floor is 2.87 um, the closest observed division sister pair."""



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


def load_detector(weights: Path, device):
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

    def detect_map(frame: np.ndarray) -> np.ndarray:
        """Detection **logits** for one frame, at the input resolution.

        Logits, not a sigmoid: the map is saturated (median sigmoid 1.0000 at GT
        centroids, ~6,000 voxels/frame above 0.99), so a sigmoid flattens exactly
        the ranking the peak finder needs. The frame is duplicated into a fake
        pair because the U-Net is temporal and wants two frames.
        """
        with torch.no_grad():
            t = torch.from_numpy(frame).to(device)
            pair = torch.stack([t, t], dim=0).unsqueeze(0).unsqueeze(2)
            feats = unet(pair)
            return head(feats[0, 0:1])[0, 0].float().cpu().numpy()

    return detect_map


def detect_video(
    volumes: np.ndarray,
    n_per_frame: int,
    separation_um: float,
    detect_map=None,
    refine: bool = False,
) -> list[np.ndarray]:
    """Peaks per frame: from the learned head when given one, else from DoG."""
    if detect_map is None:
        return [
            detect_dog(v, n_per_frame, ISO_UM, min_separation_um=separation_um,
                       refine=refine)
            for v in volumes
        ]
    return [
        detect_from_map(detect_map(v), n_per_frame, ISO_UM, separation_um, refine=refine)
        for v in volumes
    ]


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
) -> Path:
    """Predict every video in *test_dir* and write a validated submission."""
    names = list_datasets(test_dir)
    if limit:
        names = names[:limit]

    detect_map = None
    if weights is not None:
        device = resolve_device()
        print(f"detector: unet on {device} ({weights})", flush=True)
        detect_map = load_detector(weights, device)
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
        coords = detect_video(volumes, n_per_frame, separation_um, use_map, refine)
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

        links = link_sequence(coords, (ISO_UM, ISO_UM, ISO_UM),
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
