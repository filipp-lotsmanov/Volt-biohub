"""GPU discovery and selection for a *shared* multi-GPU box.

The training server has 8 cards (4× RTX A6000 `sm_86`, 4× RTX 6000 Ada `sm_89`,
48 GB each) shared with other tenants, and which cards are busy changes from day
to day. Two consequences drive this module:

- **A hardcoded device index is wrong by tomorrow.** Pick at run time.
- **`nvidia-smi` inside the container sees other tenants' *memory* but not their
  *processes*.** So free memory is trustworthy and the process list is not; never
  infer "idle" from an empty process table.

Selection samples each card several times rather than trusting one instantaneous
reading, because utilisation swings second to second and a single sample happily
reports a busy card as idle. Memory is taken as the *worst* (least free) sample
and utilisation as the mean, so the estimate errs toward caution.

Usage
-----
    # shell: pin the best card for one run
    CUDA_VISIBLE_DEVICES=$(uv run python -m volt_biohub.gpu) uv run python scripts/train...

    # shell: the best two, or a human-readable table, or a prompt
    CUDA_VISIBLE_DEVICES=$(uv run python -m volt_biohub.gpu --count 2)
    uv run python -m volt_biohub.gpu --list
    uv run python -m volt_biohub.gpu --interactive

    # python: inside a script
    from volt_biohub.gpu import pick_gpus, resolve_device
    device = resolve_device()                 # honours CUDA_VISIBLE_DEVICES, cpu fallback
    ids = pick_gpus(count=1, min_free_gib=8)  # physical indices, for the env var
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

_QUERY = "index,name,memory.total,memory.free,utilization.gpu,compute_cap"


@dataclass(frozen=True)
class Gpu:
    """One card's observed state, aggregated over several samples."""

    index: int
    name: str
    total_mib: int
    free_mib: int          # the *minimum* free seen across samples
    util_pct: float        # mean utilisation across samples
    compute_cap: float     # e.g. 8.6 (Ampere A6000) or 8.9 (Ada)

    @property
    def free_gib(self) -> float:
        return self.free_mib / 1024

    @property
    def used_frac(self) -> float:
        return 1.0 - (self.free_mib / self.total_mib) if self.total_mib else 1.0


def _smi_once() -> list[tuple[int, str, int, int, float, float]]:
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
        text=True, timeout=30,
    )
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total, free, util, cap = parts[:6]
        # A card in a bad state reports "[N/A]" for utilisation; treat that as busy
        # rather than skipping the row, so it sorts last instead of looking idle.
        try:
            util_f = float(util)
        except ValueError:
            util_f = 100.0
        try:
            cap_f = float(cap)
        except ValueError:
            cap_f = 0.0
        rows.append((int(idx), name, int(total), int(free), util_f, cap_f))
    return rows


def scan(samples: int = 3, interval: float = 0.4) -> list[Gpu]:
    """Sample every visible GPU *samples* times and aggregate conservatively.

    Returns an empty list when no NVIDIA driver is present, so callers can fall
    back to CPU rather than crash — the box is not the only place this runs.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        batches = []
        for i in range(max(1, samples)):
            if i:
                time.sleep(interval)
            batches.append(_smi_once())
    except (subprocess.SubprocessError, OSError):
        return []
    if not batches or not batches[0]:
        return []

    by_index: dict[int, list[tuple]] = {}
    for batch in batches:
        for row in batch:
            by_index.setdefault(row[0], []).append(row)

    gpus = []
    for idx, rows in sorted(by_index.items()):
        gpus.append(Gpu(
            index=idx,
            name=rows[0][1],
            total_mib=rows[0][2],
            free_mib=min(r[3] for r in rows),          # worst case
            util_pct=sum(r[4] for r in rows) / len(rows),
            compute_cap=rows[0][5],
        ))
    return gpus


def rank(gpus: list[Gpu]) -> list[Gpu]:
    """Best first.

    Free memory is the dominant term because it is what actually kills a run, and
    it is the one number other tenants cannot hide from us. Utilisation is bucketed
    to 10 % rather than used raw: it is noisy enough that treating 3 % and 7 % as
    different would just add jitter. Compute capability breaks remaining ties so an
    Ada card (`sm_89`, ~1.7× the A6000's cores) wins over an equally free Ampere
    one, and the index breaks the rest so repeated calls are stable.
    """
    return sorted(
        gpus,
        key=lambda g: (-(g.free_mib // 1024), int(g.util_pct // 10), -g.compute_cap, g.index),
    )


def pick_gpus(count: int = 1, min_free_gib: float = 8.0, samples: int = 3) -> list[int]:
    """Physical indices of the *count* best cards with at least *min_free_gib* free.

    Physical, because that is what ``CUDA_VISIBLE_DEVICES`` takes. Inside the
    process torch will renumber them from 0.
    """
    eligible = [g for g in rank(scan(samples=samples)) if g.free_gib >= min_free_gib]
    return [g.index for g in eligible[:count]]


def resolve_device(prefer_cuda: bool = True):
    """The one device resolver: never assume a GPU exists.

    Respects an existing ``CUDA_VISIBLE_DEVICES`` — if the caller already pinned
    a card, this must not second-guess it.
    """
    import torch

    if prefer_cuda and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def format_table(gpus: list[Gpu], chosen: list[int] | None = None) -> str:
    chosen = chosen or []
    lines = [f"{'idx':>3}  {'name':<30} {'free GiB':>12}  {'util':>5}  {'cc':>4}  usage"]
    for g in rank(gpus):
        bar_len = 20
        filled = round(g.used_frac * bar_len)
        bar = "#" * filled + "." * (bar_len - filled)
        mark = "  <= pick" if g.index in chosen else ""
        lines.append(
            f"{g.index:>3}  {g.name:<30} {g.free_gib:>5.1f} /{g.total_mib / 1024:>4.0f}"
            f"  {g.util_pct:>4.0f}%  {g.compute_cap:>4.1f}  {bar}{mark}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scan the shared GPUs and print the best index/indices.",
        epilog="Default output is bare indices for: "
               "CUDA_VISIBLE_DEVICES=$(python -m volt_biohub.gpu)",
    )
    ap.add_argument("--count", type=int, default=1, help="How many cards to pick (default 1).")
    ap.add_argument("--min-free-gib", type=float, default=8.0,
                    help="Skip cards with less free memory than this (default 8).")
    ap.add_argument("--samples", type=int, default=3,
                    help="nvidia-smi samples to average (default 3); 1 is faster but noisy.")
    ap.add_argument("--list", action="store_true", help="Print the table, pick nothing.")
    ap.add_argument("--interactive", action="store_true",
                    help="Show the table and prompt, defaulting to the recommendation.")
    args = ap.parse_args(argv)

    gpus = scan(samples=args.samples)
    if not gpus:
        print("No NVIDIA GPU visible (no driver, or nvidia-smi failed).", file=sys.stderr)
        return 1

    ranked = rank(gpus)
    eligible = [g for g in ranked if g.free_gib >= args.min_free_gib]
    chosen = [g.index for g in eligible[: args.count]]

    if args.list:
        print(format_table(gpus, chosen))
        return 0

    if args.interactive:
        print(format_table(gpus, chosen), file=sys.stderr)
        if not chosen:
            print(f"\nNothing has {args.min_free_gib:.0f} GiB free.", file=sys.stderr)
            return 1
        default = ",".join(map(str, chosen))
        # Prompt on stderr and answer on stdout, so the chosen value can still be
        # captured with $(...) while the human-facing table goes to the terminal.
        try:
            reply = input(f"\nGPU(s) to use [{default}]: ").strip()
        except EOFError:
            reply = ""
        print(reply or default)
        return 0

    if not chosen:
        print(
            f"No GPU with >= {args.min_free_gib:.0f} GiB free. Best is "
            f"{ranked[0].index} with {ranked[0].free_gib:.1f} GiB.",
            file=sys.stderr,
        )
        return 1

    print(",".join(map(str, chosen)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
