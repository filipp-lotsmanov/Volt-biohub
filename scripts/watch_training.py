#!/usr/bin/env python
"""Emit one line per genuine training anomaly. Silence means healthy.

Deliberately does NOT alert on the per-epoch `CV=` column or on "best epoch is
0". That print is `epoch_selection_score` over 6-8 videos -- an effective sample
size of about 2 -- and this project spent a week treating its noise as a
loss/metric anti-correlation. Scored on a full fold instead, the same checkpoints
improve monotonically with training. So a falling CV column is expected and is
not evidence of collapse.

What a real collapse looks like here, and all of it is watched:
  * NaN/inf in either loss -- the only unambiguous divergence signal
  * detection loss rising persistently above its own running minimum
  * detection loss flat to 4 decimals for many epochs (dead gradients)
  * a Traceback, CUDA OOM, or the OOM-killer -- the shared box has other tenants
    and one filled a card mid-run on 2026-09-12
  * the process disappearing, which is how an OOM-kill actually presents
Plus a heartbeat every 10th epoch so progress is visible without polling.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

EPOCH = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|\s*edge=([0-9.eE+-]+|nan|inf)\s*\|\s*"
    r"det=([0-9.eE+-]+|nan|inf)\s*\|\s*CV=([0-9.eE+-]+|nan)"
)
FATAL = re.compile(r"Traceback|OutOfMemoryError|CUDA error|Killed|MemoryError", re.I)

RISE_FACTOR = 1.5      # det loss this far above its running min, sustained
RISE_EPOCHS = 3
FLAT_EPOCHS = 12       # det loss unchanged to 4dp for this many epochs


def tail(path: Path, n: int = 400_000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-n:].decode("utf-8", "replace")


def alive(method: str) -> bool:
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    return any(f"--method {method}" in ln and "train_unet" in ln for ln in out.splitlines())


def main(argv: list[str]) -> int:
    # argv: method=logpath pairs
    runs = {}
    for spec in argv[1:]:
        method, _, log = spec.partition("=")
        runs[method] = {"log": Path(log), "seen": 0, "min_det": float("inf"),
                        "rising": 0, "flat": 0, "last_det": None, "dead": False}
    print(f"watching {len(runs)} run(s): {', '.join(runs)}", flush=True)

    while True:
        for method, st in runs.items():
            txt = tail(st["log"])

            for m in FATAL.finditer(txt[-20000:]):
                print(f"[{method}] FATAL in log: {m.group(0)}", flush=True)
                st["dead"] = True
                break

            rows = EPOCH.findall(txt)
            for ep, tot, edge, det, cv in rows[st["seen"]:]:
                st["seen"] += 1
                if any(v in ("nan", "inf") for v in (edge, det)):
                    print(f"[{method}] DIVERGED at epoch {ep}: edge={edge} det={det}",
                          flush=True)
                    continue
                d = float(det)

                if st["last_det"] is not None and abs(d - st["last_det"]) < 1e-4:
                    st["flat"] += 1
                    if st["flat"] == FLAT_EPOCHS:
                        print(f"[{method}] det loss FLAT at {d:.4f} for "
                              f"{FLAT_EPOCHS} epochs (epoch {ep}) -- dead gradients?",
                              flush=True)
                else:
                    st["flat"] = 0
                st["last_det"] = d

                if d > st["min_det"] * RISE_FACTOR:
                    st["rising"] += 1
                    if st["rising"] == RISE_EPOCHS:
                        print(f"[{method}] det loss RISING: {d:.4f} vs min "
                              f"{st['min_det']:.4f} for {RISE_EPOCHS} epochs "
                              f"(epoch {ep})", flush=True)
                else:
                    st["rising"] = 0
                st["min_det"] = min(st["min_det"], d)

                if int(ep) % 10 == 0:
                    print(f"[{method}] epoch {ep}/{tot} edge={edge} det={det} "
                          f"CV={cv} (min det {st['min_det']:.4f}) -- healthy",
                          flush=True)

            if not st["dead"] and not alive(method):
                done = st["seen"] > 0 and rows and rows[-1][0] == str(int(rows[-1][1]) - 1)
                print(f"[{method}] PROCESS GONE after {st['seen']} epochs "
                      f"({'finished' if done else 'UNEXPECTED -- likely OOM-killed'})",
                      flush=True)
                st["dead"] = True

        if all(s["dead"] for s in runs.values()):
            print("all watched runs ended", flush=True)
            return 0
        time.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
