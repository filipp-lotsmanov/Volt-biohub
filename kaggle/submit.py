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
import math
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



def volume_meta(zarr_path: Path) -> tuple[int, tuple[int, int, int], float]:
    """Frame count, downsampled spatial shape and the stored 0.9 quantile.

    What :func:`load_volumes` returns minus the pixels, for the paths that do
    not need them. Reading the array is not free: one video is 839 MB
    uncompressed, and decoding it twice per dataset is real time on a notebook
    with a handful of vCPUs.
    """
    g = zarr.open_group(str(zarr_path), mode="r")
    shape = tuple(g["0"].shape)
    stats = g.attrs.get("image_statistics") or {}
    q = stats.get("quantiles") or {}
    if "0.9" in q:
        q90 = float(q["0.9"])
    else:
        q90 = float(np.percentile(np.asarray(g["0"][0]).astype(np.float32), 90))
    dz, dy, dx = DOWNSAMPLE
    return (shape[0],
            (-(-shape[1] // dz), -(-shape[2] // dy), -(-shape[3] // dx)),
            q90)


def load_detector(weights: Path, device, tta: bool = False,
                  feature_tta: bool = False,
                  secondary: Path | None = None,
                  secondary_weight: float = 0.0,
                  retention: float = 0.90,
                  det_threshold_for_guard: float | None = None,
                  tta_views: int = 8):
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

    # A second, independently-trained detector. The public cluster blends two and
    # we had been running one: `…seed314159-v1` is their SECONDARY seed and
    # `…support-pack-50ep-v1` their primary. Measured separately on our full fold
    # the two are a dead heat (0.8760 vs 0.8754) but they disagree in their
    # *errors* — divJ 0.1034 vs 0.0896, node ratio −0.071 vs −0.086 — which is the
    # case where blending pays rather than averaging two copies of one mistake.
    unet2 = head2 = None
    if secondary is not None and secondary_weight > 0.0:
        cfg2 = json.loads((secondary.parent / "config.json").read_text())
        if tuple(cfg2.get("downsample", DOWNSAMPLE)) != DOWNSAMPLE:
            raise SystemExit(
                f"secondary checkpoint trained at downsample {cfg2['downsample']}, "
                f"not {DOWNSAMPLE}; every micrometre gate downstream would be wrong."
            )
        unet2 = TemporalUNet3D(in_channels=1, out_channels=cfg2["unet_out_channels"],
                               layers=cfg2["unet_layers"])
        head2 = nn.Conv3d(cfg2["unet_out_channels"], 1, kernel_size=1)
        st2 = torch.load(secondary, map_location="cpu", weights_only=True)
        unet2.load_state_dict({k[len("unet."):]: v for k, v in st2.items()
                               if k.startswith("unet.")})
        head2.load_state_dict({k[len("detect_head."):]: v for k, v in st2.items()
                               if k.startswith("detect_head.")})
        unet2.to(device).eval()
        head2.to(device).eval()

    def _blend(primary_logits, frame_t):
        """Moment-matched logit blend with a per-frame retention guard.

        Two detectors trained separately have no reason to share a logit scale, so
        a raw average would let whichever runs hotter dominate the peak finder.
        Align the secondary to the primary's mean and standard deviation first —
        the reference clamps the scale ratio to [0.5, 2.0], which stops a nearly
        flat frame from being amplified into noise.

        The guard is the part that makes this safe rather than merely plausible:
        if the blended map yields fewer than *retention* of the primary's
        candidates on this frame, keep the primary. A blend that loses detections
        costs node recall, and node recall is the term we can least afford — it is
        what the ILP cost us (0.9419 against 0.9830) when it went wrong.
        """
        out2 = head2(unet2(torch.stack([frame_t, frame_t], dim=0)
                           .unsqueeze(0).unsqueeze(2))[0, 0:1])[0, 0].float()
        p_mean, p_std = primary_logits.mean(), primary_logits.std()
        s_mean, s_std = out2.mean(), out2.std()
        ratio = torch.clamp(p_std / (s_std + 1e-6), 0.5, 2.0)
        aligned = (out2 - s_mean) * ratio + p_mean
        blended = (1.0 - secondary_weight) * primary_logits + secondary_weight * aligned
        if det_threshold_for_guard is not None:
            cut = math.log(det_threshold_for_guard / (1.0 - det_threshold_for_guard))
            n_p = int((primary_logits > cut).sum())
            n_b = int((blended > cut).sum())
            if n_p > 0 and n_b < retention * n_p:
                return primary_logits
        return blended

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
        raising.

        **Feature TTA is a separate switch, and the separation is the point.**
        An earlier version of this docstring claimed features could not be
        symmetry-averaged "because no symmetry average is defined for coordinate
        indexing". That was simply wrong: a feature map is (C, Z, Y, X) and its
        spatial axes invert exactly like the logits. Keeping the two flags apart
        matters because detection TTA *loses* here (0.8609 against 0.8687) by
        blurring two daughters into a single peak, while the public chain's
        largest single step (0.941 -> 0.946) is averaging the features the EDGE
        predictor consumes. With ``feature_tta`` on and ``tta`` off, detections
        stay sharp and only the edge probabilities are smoothed -- so the failure
        that killed detection TTA structurally cannot occur.
        """
        avg_feats = feature_tta and want_features
        with torch.no_grad():
            t = torch.from_numpy(frame).to(device)
            if not tta and not avg_feats:
                logits, feats = _forward(t, want_features)
                if unet2 is not None:
                    logits = _blend(logits, t)
                logits = logits.cpu().numpy()
                return (logits, feats) if want_features else logits

            square = t.shape[-1] == t.shape[-2]
            acc = None
            facc = None
            feats = None
            # Rotations are a far stronger perturbation than flips for an
            # anisotropic microscope, and the reference only ever used flips:
            # PredictConfig.det_tta is flip-xy, four views. Our -0.015 for
            # "TTA" was measured on the full eight-view D4 group, which is not
            # the thing their +0.009 refers to. k in (0, 2) with the x-flip is
            # exactly the four-element flip subgroup {id, flip-x, flip-y,
            # flip-xy}, since flip-y = rot180 . flip-x.
            ks = ((0, 1, 2, 3) if tta_views >= 8 else (0, 2)) if square else (0,)
            for k in ks:
                rot = torch.rot90(t, k, dims=(-2, -1)) if k else t
                for flip in (False, True):
                    identity = k == 0 and not flip
                    view = torch.flip(rot, dims=(-1,)) if flip else rot
                    # Ask for features on every view only when averaging them.
                    out, f = _forward(view, want_features and (avg_feats or identity))
                    if f is not None:
                        if avg_feats:
                            # Same inverse as the logits, on the same two axes:
                            # a feature map is (C, Z, Y, X).
                            if flip:
                                f = torch.flip(f, dims=(-1,))
                            if k:
                                f = torch.rot90(f, -k, dims=(-2, -1))
                            facc = f if facc is None else facc + f
                        elif identity:
                            feats = f
                    if flip:
                        out = torch.flip(out, dims=(-1,))
                    if k:
                        out = torch.rot90(out, -k, dims=(-2, -1))
                    acc = out if acc is None else acc + out
                    if not tta and identity:
                        # Detection TTA is off: keep the sharp identity map and
                        # stop accumulating, so the peaks are unblurred.
                        sharp = acc
            n_views = (len(ks) * 2) if square else 2
            logits = ((acc / n_views) if tta else sharp).cpu().numpy()
            if avg_feats:
                feats = facc / n_views
            return (logits, feats) if want_features else logits

    def detect_pair(frame_a: np.ndarray, frame_b: np.ndarray):
        """Logits and trunk features for a **real** consecutive pair.

        This exists because ``detect_map`` duplicates one frame into a fake
        pair, and that is the single largest divergence between this file and
        the reference inference it reimplements. For *detection* the fake pair
        is free -- the temporal attention degenerates cleanly on two identical
        tokens, node recall 0.9973 against 0.9967 -- which is what was measured
        and what the fake pair was justified on. For the **edge head** it is
        not free at all: the head's whole job is to compare a cell's trunk
        representation at t against candidates at t+1, and a trunk told that
        nothing moved between the two frames has had exactly that signal
        removed.

        Measured on the four movies the public notebooks validate on, our
        stack against the reference's own path on the same weights and the same
        scorer: greedy edge head 0.8600 adjusted edge Jaccard here (node recall
        0.9428) against 0.8830 there, and with the ILP 0.8677 against 0.9299.
        The fake pair is the leading suspect for that gap, and this is how it
        gets tested rather than argued about.

        TTA is not applied here. It is off in every configuration that uses the
        edge head, and silently averaging the pair path over symmetries while
        the single-frame path did something else is how the last measurement
        error in this file happened.
        """
        a = torch.from_numpy(frame_a).to(device)
        b = torch.from_numpy(frame_b).to(device)
        with torch.no_grad():
            feats = unet(torch.stack([a, b], dim=0).unsqueeze(0).unsqueeze(2))
            la = head(feats[0, 0:1])[0, 0].float()
            lb = head(feats[0, 1:2])[0, 0].float()
            if unet2 is not None:
                la, lb = _blend(la, a), _blend(lb, b)
            return la.cpu().numpy(), lb.cpu().numpy(), feats[0, 0], feats[0, 1]

    detect_map.pair = detect_pair
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


def detect_and_candidates(
    volumes: np.ndarray,
    detect_map,
    edge_head,
    n_per_frame: int,
    separation_um: float,
    downsample: tuple[int, int, int],
    refine: bool = False,
    threshold: float | None = None,
    candidate_threshold: float = 0.48,
    want_probs: bool = False,
    bidirectional_weight: float = 0.0,
    temperature: float = 1.0,
    real_pairs: bool = False,
) -> tuple[list[np.ndarray], list[tuple[int, int, float, float]], list | None]:
    """Detect, then emit ILP *candidate* edges instead of resolved links.

    Same interleaved structure and the same memory argument as
    ``detect_and_link_learned`` — two frames of features live, 67 MB rather than
    3.4 GB — but it never resolves the probabilities into a topology. That is the
    whole point: resolving them greedily is what made the edge head score 0.7632
    against the geometric linker's 0.7830, because an abstention becomes a
    deleted node. Handed to the solver, the same abstention becomes a
    disappearance cost it can price against the alternatives.

    Candidates are accumulated sparsely per frame pair rather than by keeping the
    probability matrices: 770 x 770 float32 per pair is 2.4 MB, so a 100-frame
    video would hold 235 MB of matrices to extract a few thousand candidates
    from. The source-softmax means at most two sources clear 0.48 for any target,
    so the sparse form is 100x smaller with no loss.
    """

    from volt_biohub.edge_head import pair_probs
    from volt_biohub.ilp_link import candidates_from_probs

    window_shape = (2, *volumes.shape[1:])
    coords_per_t: list[np.ndarray] = []
    candidates: list[tuple[int, int, float, float]] = []
    kept_probs: list[np.ndarray] | None = [] if want_probs else None
    offset = 0

    def _peaks(logits):
        return detect_from_map(logits, n_per_frame, ISO_UM, separation_um,
                               refine=refine, threshold=threshold)

    def _pair(t, fa, fb):
        """Score pair *t*, accumulate its candidates, and drop the features."""
        nonlocal offset
        probs = pair_probs(
            edge_head, fa, fb, coords_per_t[t], coords_per_t[t + 1],
            t_src=0, window_shape=window_shape, downsample=downsample,
            bidirectional_weight=bidirectional_weight,
            temperature=temperature,
        )
        # candidates_from_probs indexes globally, so hand it just this pair and
        # shift by the running offset rather than re-deriving offsets over a
        # coords list that is still being built.
        local = candidates_from_probs(
            [probs], [coords_per_t[t], coords_per_t[t + 1]], ISO_UM,
            threshold=candidate_threshold,
        )
        candidates.extend((offset + a, offset + b, pr, d) for a, b, pr, d in local)
        offset += len(coords_per_t[t])
        if kept_probs is not None:
            # ~36 MB for a 100-frame video at 300 nodes/frame, 235 MB at 770.
            kept_probs.append(probs)

    # A real consecutive pair per window when the detector offers one. See
    # load_detector.detect_pair: the fake pair costs nothing for detection and
    # removes exactly the signal the edge head reads. Features are consumed and
    # dropped inside the loop -- 32 channels x 64^3 for every frame at once is
    # 3.4 GB, which is the whole reason this is interleaved.
    pair_fn = getattr(detect_map, "pair", None) if real_pairs else None

    if pair_fn is not None and len(volumes) > 1:
        for t in range(len(volumes) - 1):
            la, lb, fa, fb = pair_fn(volumes[t], volumes[t + 1])
            if t == 0:
                coords_per_t.append(_peaks(la))
            coords_per_t.append(_peaks(lb))
            _pair(t, fa, fb)
    else:
        prev_feat = None
        for t, frame in enumerate(volumes):
            logits, feat = detect_map(frame, want_features=True)
            coords_per_t.append(_peaks(logits))
            if t > 0:
                _pair(t - 1, prev_feat, feat)
            prev_feat = feat

    return coords_per_t, candidates, kept_probs


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
    tta_views: int = 8,
    div_kwargs: dict | None = None,
    ilp: bool = False,
    ilp_prune: bool = False,
    learned_bonus: float = 0.0,
    bidirectional_weight: float = 0.0,
    temperature: float = 1.0,
    feature_tta: bool = False,
    secondary_weights: Path | None = None,
    secondary_weight: float = 0.0,
    secondary_edge_weight: float = 0.0,
    candidate_threshold: float = 0.48,
    real_pairs: bool = False,
    reference: bool = False,
    ilp_kwargs: dict | None = None,
    gaps: bool = True,
    relink_kwargs: dict | None = None,
    gap_kwargs: dict | None = None,
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
        detect_map = load_detector(weights, device, tta=tta,
                                   tta_views=tta_views,
                                   feature_tta=feature_tta,
                                   secondary=secondary_weights,
                                   secondary_weight=secondary_weight,
                                   det_threshold_for_guard=det_threshold)
        if secondary_weights is not None and secondary_weight > 0:
            print(f"detector: BLEND w={secondary_weight} with {secondary_weights}",
                  flush=True)
        # The head is needed by three different consumers, not just --learned-edges:
        # the ILP needs its probabilities as candidate weights, and the relink
        # needs them as a cost bonus. Gating the load on `learned_edges` alone
        # silently disabled both -- six "ILP" arms ran with edge_head None and
        # measured --no-gaps instead, clustering by GPU card rather than by
        # parameter. Load it whenever a consumer asked for it.
        if learned_edges or ilp or learned_bonus:
            import json as _json

            from volt_biohub.edge_head import load_edge_head
            cfg = _json.loads((weights.parent / "config.json").read_text())
            edge_head = load_edge_head(weights, cfg, device)
            # None means the checkpoint has no transformer.* weights, so fall
            # back to the geometric linker rather than dying at hour ten.
            wanted = ("learned edge head" if learned_edges else
                      "ILP candidates" if ilp else "relink cost bonus")
            if edge_head is None:
                raise SystemExit(
                    f"asked for {wanted} but the checkpoint at {weights} has no "
                    "transformer.* weights. Failing here rather than silently "
                    "running the geometric linker and reporting it as this arm."
                )
            print(f"linker: {wanted}", flush=True)
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

    # The reference inference path, copied rather than reimplemented. On the
    # four movies the public notebooks validate on it reaches adjusted edge
    # Jaccard 0.9299 where this file's own detect-and-link reaches 0.8888, and
    # with our post-processing on top the pair scores 0.9505 -- above the 0.9417
    # those notebooks report for themselves. See volt_biohub.reference_predict.
    ref_model = ref_cfg = ref_secondary = None
    if reference:
        if weights is None:
            raise SystemExit("--reference needs --weights")
        from volt_biohub import reference_predict as rp
        ref_model, ref_W, ref_down = rp.load_model(weights, resolve_device())
        if tuple(ref_down) != DOWNSAMPLE:
            raise SystemExit(f"checkpoint downsample {ref_down} != {DOWNSAMPLE}")
        ref_cfg = rp.PredictConfig(det_threshold=det_threshold if det_threshold
                                   else 0.965, det_tta=tta)
        ref_secondary = rp.make_secondary(
            secondary_weights, resolve_device(), secondary_weight,
            secondary_edge_weight, window_size=ref_W,
            downsample=DOWNSAMPLE) if secondary_weights else None
        if secondary_edge_weight > 0.0:
            print("linker: SECONDARY EDGE HEAD ensembled at weight "
                  f"{secondary_edge_weight}", flush=True)
        if bidirectional_weight > 0.0:
            print("linker: BIDIRECTIONAL fusion, reverse weight "
                  f"{bidirectional_weight}",
                  flush=True)
        print(f"linker: REFERENCE path (det_tta={ref_cfg.det_tta}, "
              f"pool {ref_cfg.pool_kernel_um} um, edge threshold "
              f"{ref_cfg.threshold})", flush=True)

    rows: list[dict] = []
    t0 = time.monotonic()
    for i, name in enumerate(names, 1):
        t1 = time.monotonic()
        # The reference path reads its own frames, so loading them here as well
        # would decode 839 MB twice per video -- cheap on 256 cores, not on
        # Kaggle's handful of vCPUs. The division ranker's intensity lookup is
        # the one thing in that branch that needs the pixels.
        need_pixels = (not reference) or (div_kwargs or {}).get("rise_weight")
        if need_pixels:
            volumes, q90 = load_volumes(test_dir / f"{name}.zarr")
            n_frames, vol_shape = len(volumes), tuple(volumes.shape[1:])
        else:
            volumes = None
            n_frames, vol_shape, q90 = volume_meta(test_dir / f"{name}.zarr")
        n_true = predict_n_true(q90)
        n_per_frame = budget_for(n_true, n_frames, budget)

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
            crowded = (n_true / max(n_frames, 1)) >= switch_at
            use_map = detect_map if crowded else None
            picked = "unet" if crowded else "dog"
        reference_links = None
        learned_links = None
        candidates = None
        probs_per_pair = None
        # A video shorter than one window has no consecutive pair to slide
        # over, and reference_predict would index past the end building its
        # first one. Every sample measured is exactly 100 frames and the hidden
        # set is described as train-sized, so this is latent -- but a crash in
        # hour two of a graded rerun is unrecoverable and the fallback is one
        # branch we already maintain.
        use_reference = reference and n_frames >= 2
        if reference and not use_reference:
            print(f"    WARNING: {name} has {n_frames} frame(s); falling back "
                  f"to the single-frame detector", flush=True)
            volumes, _q = load_volumes(test_dir / f"{name}.zarr")
        if use_reference:
            from volt_biohub import reference_predict as rp
            from volt_biohub.ilp_link import prune_dp
            raw_coords, raw_edges = rp.predict_video(
                ref_model, test_dir / f"{name}.zarr", resolve_device(),
                cfg=ref_cfg, window_size=ref_W, downsample=DOWNSAMPLE,
                secondary=ref_secondary, bidir=bidirectional_weight)
            coords, reference_links, sparse = rp.to_per_frame(
                raw_coords, raw_edges, DOWNSAMPLE, n_frames, ISO_UM)
            probs_per_pair = rp.probs_as_matrices(coords, sparse)
            if ilp_prune:
                # Applied here rather than through run_all's prune_fn so the
                # solver sees the reference's own edges and nothing else.
                coords, reference_links = prune_dp(
                    coords, reference_links, probs_per_pair, ISO_UM,
                    **(ilp_kwargs or {}))
            picked = "ref"
            if learned_bonus:
                # Reference DETECTION, our linker. The decomposition that says
                # where the +0.036 on fold 1 comes from: their edge head
                # choosing the topology, or their peak picker and flip TTA
                # feeding a better node set into ours. It matters because the
                # edge head is the memorisation-exposed half -- it is part of a
                # checkpoint that trained on both our folds -- while the
                # detection half helps our own held-out detector too.
                reference_links = None

        want_pair_probs = (not use_reference) and (
            ilp or ilp_prune or learned_bonus or (learned_edges and postprocess))
        if want_pair_probs and edge_head is not None and use_map is not None:
            coords, candidates, probs_per_pair = detect_and_candidates(
                volumes, use_map, edge_head, n_per_frame, separation_um,
                DOWNSAMPLE, refine=refine, threshold=det_threshold,
                candidate_threshold=candidate_threshold,
                want_probs=bool(learned_bonus or ilp_prune
                                or (learned_edges and postprocess)),
                bidirectional_weight=bidirectional_weight,
                temperature=temperature,
                real_pairs=real_pairs,
            )
            if not ilp:
                candidates = None
        elif (not use_reference) and edge_head is not None and use_map is not None:
            coords, learned_links = detect_and_link_learned(
                volumes, use_map, edge_head, n_per_frame, separation_um,
                DOWNSAMPLE, refine=refine, threshold=det_threshold,
                edge_threshold=edge_threshold,
            )
        elif not use_reference:
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
            centre = np.array([[s // 2 for s in vol_shape]], dtype=np.int64)
            coords = [centre] + [np.empty((0, 3), dtype=np.int64)] * (n_frames - 1)

        def _make_prune(probs):
            # Solving the graph our relink already resolved, which is what the
            # reference does -- see volt_biohub.ilp_link.prune. Deliberately a
            # closure rather than a flag inside run_all: postprocess.py stays
            # free of tracksdata, so the DoG fallback still runs with it (and
            # torch) absent entirely.
            # The exact tree DP, not the solver: identical edge sets
            # (tests/test_ilp_dp.py) in milliseconds instead of ~15 s a video,
            # and it keeps tracksdata off the inference path entirely.
            from volt_biohub.ilp_link import prune_dp as ilp_prune_fn

            def _fn(pts, edges, probs=probs):
                return ilp_prune_fn(pts, edges, probs, ISO_UM, **(ilp_kwargs or {}))
            return _fn

        # The reference's linker: greedy resolution of the edge head's own
        # probabilities, in place of the geometric relink. Measured through its
        # code on the public notebooks' 4-movie proxy set, adjusted edge
        # Jaccard is 0.8830 this way and 0.9299 once the ILP prunes it, against
        # our relink's 0.8888 -- so the head and the solver are a pair, and
        # neither half reproduces the gain alone.
        if learned_edges and postprocess and probs_per_pair is not None:
            from volt_biohub.edge_head import resolve_greedy
            learned_links = [
                resolve_greedy(probs_per_pair[t], coords[t], coords[t + 1],
                               ISO_UM, threshold=edge_threshold)
                for t in range(len(coords) - 1)
            ]

        if candidates is not None:
            # The solver replaces the linker AND selects the node set. It runs
            # before post-processing so the remaining stages operate on its
            # topology -- and `relink=False` below is the whole point: the
            # reference's own ablation prices its two-pass Hungarian at -0.011
            # precisely because it overwrites these edges.
            from volt_biohub.ilp_link import solve as ilp_solve
            coords, links = ilp_solve(coords, candidates, ISO_UM,
                                      **(ilp_kwargs or {}))

        def _intensity(t, coord, volumes=volumes):
            """Normalised image intensity at a node, for the division ranker.

            Clipped rather than bounds-checked: line-fit smoothing can push a
            border node a fraction of a voxel outside the grid (which is what
            produced the -1 coordinates the validator rejected), and a fork
            candidate is not worth losing over a rounding artefact at the edge.
            """
            if volumes is None or t < 0 or t >= len(volumes):
                return None
            z, y, x = (int(round(float(v))) for v in coord)  # noqa: RUF046
            sz, sy, sx = volumes.shape[1:]
            return float(volumes[t,
                                 min(max(z, 0), sz - 1),
                                 min(max(y, 0), sy - 1),
                                 min(max(x, 0), sx - 1)])

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
                shape=vol_shape,
                div_kwargs={**(div_kwargs or {}), "intensity": _intensity}
                           if (div_kwargs or {}).get("rise_weight") else div_kwargs,
                links_per_t=(reference_links if reference_links is not None
                             else (links if candidates is not None
                                   else (learned_links
                                         if learned_edges and learned_links
                                         else None))),
                gaps=gaps,
                relink_kwargs={
                    **(relink_kwargs or {}),
                    **({"probs_per_pair": probs_per_pair,
                        "learned_bonus": learned_bonus}
                       if probs_per_pair is not None and not use_reference else {}),
                } or None,
                gap_kwargs=gap_kwargs,
                prune_fn=(_make_prune(probs_per_pair)
                          if ilp_prune and not use_reference else None),
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
