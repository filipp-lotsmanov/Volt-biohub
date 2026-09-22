#!/usr/bin/env python
"""Average the weights of several checkpoints FROM THE SAME RUN.

Why. The public pack's ``history.csv`` records ``validation_score`` oscillating
between 0.9696 and 0.9781 across epochs 370-400 with no trend, and their
selector is ``validation_acc x validation_recall`` with ``validation_acc``
pinned at 0.9999 for all 400 epochs -- i.e. effectively edge-matrix recall, the
quantity this project measured as worthless for selection (2026-09-16). So
``edge_predictor_best.pth``, the checkpoint the entire field forks, is the
**argmax of noise** over roughly thirty statistically indistinguishable
epochs. ``checkpoint_last.pth`` ships in the same directory, 19 epochs later,
and every one of its 136 tensors differs.

Averaging late checkpoints is the standard remedy, and here it is strictly
cheaper than the alternative: an activation ensemble needs a second forward
pass per frame, while a soup is one model of the same size and costs nothing at
inference -- which matters because the shipped kernel already runs ~5.7 h of a
12 h cap.

**Only valid within a run.** Two independently initialised networks occupy
different basins under permutation symmetry, so averaging them destroys both.
``pubfull`` (``unet_transformer_alltrain_seed314159_v1``) and ``pack50``
(``unet_transformer_5090_50ep_v1``, no seed recorded) are different runs and
must be combined at the activation level instead -- that is what the detector
blend in ``reference_predict.Secondary`` is for.

Two details that are easy to get wrong:

* ``checkpoint_last.pth`` is a *training* checkpoint (25 MB: optimizer state
  included), so the weights live under ``model_state_dict``, not at the top
  level. ``load_model`` wants a bare state dict.
* ``num_batches_tracked`` is an int64 counter. Averaging it is meaningless, and
  in these two checkpoints it differs by 8.9e4 -- large enough to look alarming
  in a naive per-tensor diff. Eval-mode BatchNorm never reads it, so it is
  taken from the last member. The float BatchNorm running statistics *are*
  averaged; that is an approximation (variance is not linear) but they differ
  by only 9-12 % relative here.

Usage
-----
    uv run python scripts/soup_checkpoints.py \
        --out weights/soup_pub/split_0 \
        --members <pack>/weights/unet_transformer/split_0/edge_predictor_best.pth \
                  <pack>/weights/unet_transformer/split_0/checkpoint_last.pth
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

_STATE_KEYS = ("model_state_dict", "model", "state_dict")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    """A bare state dict, whether *path* is one or wraps one."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and not any(
        isinstance(v, torch.Tensor) for v in obj.values()
    ):
        for k in _STATE_KEYS:
            if k in obj:
                return obj[k]
        raise ValueError(f"{path}: no state dict under any of {_STATE_KEYS}")
    return obj


def soup(members: list[Path]) -> dict[str, torch.Tensor]:
    states = [load_state(m) for m in members]
    keys = set(states[0])
    for s in states[1:]:
        if set(s) != keys:
            raise ValueError("checkpoints have different tensor names; "
                             "they are not from the same architecture")
    out: dict[str, torch.Tensor] = {}
    for k, ref in states[0].items():
        if ref.dtype.is_floating_point:
            out[k] = torch.stack([s[k].float() for s in states]).mean(0).to(ref.dtype)
        else:
            out[k] = states[-1][k].clone()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", type=Path, nargs="+", required=True,
                   help="Checkpoints from the SAME training run.")
    p.add_argument("--out", type=Path, required=True,
                   help="Directory to write edge_predictor_best.pth + config.json.")
    args = p.parse_args(argv)

    state = soup(args.members)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out / "edge_predictor_best.pth")

    # load_model reads config.json from the weights' own directory, so the soup
    # is only loadable if one travels with it.
    cfg = next((m.parent / "config.json" for m in args.members
                if (m.parent / "config.json").exists()), None)
    if cfg is None:
        raise SystemExit("no config.json beside any member; load_model needs one")
    shutil.copy(cfg, args.out / "config.json")

    n_float = sum(1 for v in state.values() if v.dtype.is_floating_point)
    print(f"souped {len(args.members)} checkpoints -> {args.out}")
    print(f"  {n_float} float tensors averaged, "
          f"{len(state) - n_float} integer tensors copied from the last member")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
