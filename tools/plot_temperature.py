#!/usr/bin/env python3
"""
plot_temperature.py — Temperature vs. Time plot from LAMMPS log files
======================================================================
Parses LAMMPS thermo output from log.lammps and plots T(t).

Features
--------
  • Auto-detects timestep from log (falls back to --dt)
  • Skips SHAKE/non-numeric lines between thermo rows
  • Optional running average overlay (--avg-window)
  • Handles multiple thermo blocks in one log (e.g. heat + NVT runs)
  • QM/MM mode: sweeps mm_N directories and overlays or concatenates T(t)

Outputs (written next to the log file)
---------------------------------------
  temperature.png         — T vs. time plot
  temperature_rawdata.csv — time (ps), step, temperature (K)

Usage
-----
  # Single log file
  python tools/plot_temperature.py log.lammps

  # Directory (auto-finds log.lammps inside)
  python tools/plot_temperature.py run_dir/

  # Running average with 50-point window
  python tools/plot_temperature.py log.lammps --avg-window 50

  # Override timestep (fs) if not in log
  python tools/plot_temperature.py log.lammps --dt 0.5

  # Concatenate all mm_N/log.lammps under a QM/MM run directory
  python tools/plot_temperature.py --mm /path/to/qmmm_run_dir

  # Plot each mm_N separately (one line per QM/MM step)
  python tools/plot_temperature.py --mm /path/to/qmmm_run_dir --per-mm

  # Custom output prefix
  python tools/plot_temperature.py log.lammps -o my_temp

  # Plot extra thermo columns alongside T (e.g. TotEng)
  python tools/plot_temperature.py log.lammps --extra TotEng
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ── shared plot style ────────────────────────────────────────────────────────
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
try:
    import plot_setting as ps
    _C  = ps.colors
    _FS = ps.fontsize
    _LS = ps.labelsize
    _LW = ps.linewidth
    _FW, _FH = ps.figsize
except ImportError:
    ps = None
    _C  = ["#1A6FDF", "#F14040", "#37AD6B", "#B177DE", "#FEC211", "#515151"]
    _FS = 9; _LS = 9; _LW = 0.4; _FW, _FH = 3.5, 2.8


# ─────────────────────────── log parser ──────────────────────────────────────

def _parse_timestep(lines: List[str]) -> Optional[float]:
    """Extract 'timestep  VALUE' from log lines (in fs for 'units real')."""
    for line in lines:
        m = re.match(r"^\s*timestep\s+([\d.eE+\-]+)", line)
        if m:
            return float(m.group(1))
    return None


def _parse_thermo_blocks(
    lines: List[str],
    columns: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Parse all thermo data blocks from LAMMPS log lines.

    Returns a dict  {col_name: np.ndarray}  where the first column is always
    'Step'.  Rows from all blocks are concatenated in order.

    A thermo block starts with a header line whose *first* token is 'Step'
    and ends at a line starting with 'Loop time' or the next header.
    Lines that cannot be parsed as all-numeric are silently skipped (e.g.
    SHAKE stats).
    """
    header_re = re.compile(r"^Step\s")

    all_rows: List[List[float]] = []
    col_names: List[str] = []
    in_block = False

    for line in lines:
        stripped = line.strip()
        if header_re.match(stripped):
            # New thermo header — grab column names (or keep existing)
            new_cols = stripped.split()
            if not col_names:
                col_names = new_cols
            in_block = True
            continue

        if in_block:
            if stripped.startswith("Loop time") or stripped.startswith("Per MPI"):
                in_block = False
                continue
            # Try to parse a numeric data row
            tokens = stripped.split()
            if not tokens:
                continue
            try:
                row = [float(t) for t in tokens]
            except ValueError:
                continue  # SHAKE stats, WARNING lines, etc.
            if len(row) == len(col_names):
                all_rows.append(row)

    if not col_names or not all_rows:
        return {}

    arr = np.array(all_rows)
    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(col_names):
        if columns is None or name in columns or name == "Step":
            result[name] = arr[:, i]

    return result


def parse_log(
    log_path: Path,
    dt_override: Optional[float] = None,
    extra_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], float]:
    """
    Parse a LAMMPS log file.

    Returns
    -------
    steps    : 1-D array of step numbers
    time_ps  : 1-D array of simulation time in ps
    extra    : dict of extra thermo column arrays (empty if none requested)
    dt_fs    : timestep in fs (used for time conversion)
    """
    with open(log_path, "r", errors="replace") as fh:
        lines = fh.readlines()

    dt_fs = dt_override if dt_override is not None else _parse_timestep(lines)
    if dt_fs is None:
        print(
            f"WARNING: timestep not found in {log_path.name}; assuming 1.0 fs. "
            "Pass --dt to override.",
            file=sys.stderr,
        )
        dt_fs = 1.0

    want = {"Step", "Temp"}
    if extra_cols:
        want.update(extra_cols)

    data = _parse_thermo_blocks(lines, columns=list(want))
    if "Step" not in data or "Temp" not in data:
        raise ValueError(f"No thermo data (Step + Temp) found in {log_path}")

    steps   = data["Step"]
    time_ps = steps * dt_fs / 1000.0   # fs → ps

    extra: Dict[str, np.ndarray] = {
        k: v for k, v in data.items() if k not in ("Step", "Temp")
    }

    return steps, time_ps, data["Temp"], extra, dt_fs


# ─────────────────────────── running average ─────────────────────────────────

def running_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean; edges use a shrinking window to avoid boundary artifacts."""
    if window <= 1:
        return arr.copy()
    half = window // 2
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


# ─────────────────────────── mm_N helpers ────────────────────────────────────

def find_mm_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    """Find mm_N directories sorted by N."""
    mm_dirs = []
    for d in run_dir.iterdir():
        if d.is_dir() and re.match(r"^mm_(\d+)$", d.name):
            mm_dirs.append((int(d.name.split("_")[1]), d))
    return sorted(mm_dirs, key=lambda x: x[0])


def find_log_in_dir(directory: Path) -> Optional[Path]:
    for name in ("log.lammps", "log", "lammps.log"):
        p = directory / name
        if p.exists():
            return p
    candidates = sorted(directory.glob("*.log"))
    return candidates[0] if candidates else None


# ─────────────────────────── plotting ────────────────────────────────────────

def _setup_axes(n_extra: int = 0):
    """Return (fig, ax_temp, [ax_extra, ...])."""
    nrows = 1 + n_extra
    if ps is not None:
        fig, axes = plt.subplots(
            nrows, 1,
            figsize=(_FW, _FH * nrows),
            sharex=True,
            dpi=matplotlib.rcParams.get("figure.dpi", 300),
        )
    else:
        fig, axes = plt.subplots(nrows, 1, figsize=(3.5, 2.8 * nrows), sharex=True, dpi=300)

    if nrows == 1:
        axes = [axes]
    return fig, axes


def plot_single(
    time_ps: np.ndarray,
    temp_K: np.ndarray,
    extra: Dict[str, np.ndarray],
    avg_window: int,
    label: Optional[str],
    out_prefix: Path,
    target_T: Optional[float],
) -> None:
    """Plot T(t) for a single log file."""
    n_extra = len(extra)
    fig, axes = _setup_axes(n_extra)

    ax_T = axes[0]
    color = _C[0]
    lw = _LW * 2

    ax_T.plot(time_ps, temp_K, color=color, lw=lw * 0.6, alpha=0.4,
              label=label or "T(t)")
    if avg_window > 1:
        avg = running_average(temp_K, avg_window)
        ax_T.plot(time_ps, avg, color=color, lw=lw,
                  label=f"running avg ({avg_window})")

    if target_T is not None:
        ax_T.axhline(target_T, color=_C[5] if len(_C) > 5 else "grey",
                     lw=lw, ls="--", label=f"target {target_T:.0f} K")

    mean_T = np.mean(temp_K)
    ax_T.axhline(mean_T, color=_C[1] if len(_C) > 1 else "red",
                 lw=lw, ls=":", label=f"mean {mean_T:.1f} K")

    ax_T.set_ylabel("Temperature (K)", fontsize=_LS)
    ax_T.legend(fontsize=_FS - 1)
    if n_extra == 0:
        ax_T.set_xlabel("Time (ps)", fontsize=_LS)

    # Extra columns
    for i, (col, vals) in enumerate(extra.items()):
        ax_e = axes[i + 1]
        ax_e.plot(time_ps, vals, color=_C[(i + 2) % len(_C)], lw=lw * 0.8)
        ax_e.set_ylabel(col, fontsize=_LS)
        if i == n_extra - 1:
            ax_e.set_xlabel("Time (ps)", fontsize=_LS)

    for ax in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    fig.savefig(png_path)
    print(f"Saved: {png_path}")
    plt.close(fig)


def plot_multi_mm(
    records: List[Tuple[int, np.ndarray, np.ndarray]],   # (mm_idx, time_ps, temp_K)
    avg_window: int,
    out_prefix: Path,
    target_T: Optional[float],
    per_mm: bool,
) -> None:
    """Overlay or concatenate T(t) for QM/MM sweep."""
    fig, axes = _setup_axes(0)
    ax_T = axes[0]
    lw = _LW * 2

    if per_mm:
        # Each mm_N as a separate line; x-axis is time within that run
        for i, (mm_idx, time_ps, temp_K) in enumerate(records):
            c = _C[i % len(_C)]
            ax_T.plot(time_ps, temp_K, color=c, lw=lw * 0.6, alpha=0.5)
            if avg_window > 1:
                avg = running_average(temp_K, avg_window)
                ax_T.plot(time_ps, avg, color=c, lw=lw, label=f"mm_{mm_idx}")
            else:
                ax_T.lines[-1].set_label(f"mm_{mm_idx}")
    else:
        # Concatenate: shift time so each run continues from the last
        offset = 0.0
        all_t = np.array([])
        all_T = np.array([])
        for mm_idx, time_ps, temp_K in records:
            shifted = time_ps - time_ps[0] + offset
            all_t = np.concatenate([all_t, shifted])
            all_T = np.concatenate([all_T, temp_K])
            offset = shifted[-1]

        ax_T.plot(all_t, all_T, color=_C[0], lw=lw * 0.5, alpha=0.4)
        if avg_window > 1:
            avg = running_average(all_T, avg_window)
            ax_T.plot(all_t, avg, color=_C[0], lw=lw,
                      label=f"running avg ({avg_window})")

        mean_T = np.mean(all_T)
        ax_T.axhline(mean_T, color=_C[1], lw=lw, ls=":",
                     label=f"mean {mean_T:.1f} K")

    if target_T is not None:
        ax_T.axhline(target_T, color=_C[5] if len(_C) > 5 else "grey",
                     lw=lw, ls="--", label=f"target {target_T:.0f} K")

    ax_T.set_xlabel("Time (ps)", fontsize=_LS)
    ax_T.set_ylabel("Temperature (K)", fontsize=_LS)
    ax_T.legend(fontsize=_FS - 1)
    ax_T.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_T.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.tight_layout()
    png_path = out_prefix.with_suffix(".png")
    fig.savefig(png_path)
    print(f"Saved: {png_path}")
    plt.close(fig)


# ─────────────────────────── CLI ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot temperature vs. time from LAMMPS log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "log", nargs="?",
        help="Path to log.lammps or a directory containing one.",
    )
    p.add_argument(
        "--mm", metavar="RUN_DIR",
        help="QM/MM run directory; sweep all mm_N sub-directories.",
    )
    p.add_argument(
        "--per-mm", action="store_true",
        help="With --mm: plot each QM/MM step as a separate line.",
    )
    p.add_argument(
        "--dt", type=float, default=None, metavar="FS",
        help="Timestep in fs (overrides value parsed from log).",
    )
    p.add_argument(
        "--avg-window", type=int, default=1, metavar="N",
        help="Running-average window size in frames (default: 1 = off).",
    )
    p.add_argument(
        "--target-T", type=float, default=None, metavar="K",
        help="Draw a dashed horizontal line at this temperature (K).",
    )
    p.add_argument(
        "--extra", nargs="+", default=[], metavar="COL",
        help="Additional thermo columns to plot (e.g. TotEng Press).",
    )
    p.add_argument(
        "--skip", type=int, default=0, metavar="N",
        help="Skip first N data rows (equilibration).",
    )
    p.add_argument(
        "-o", "--output", default=None, metavar="PREFIX",
        help="Output filename prefix (default: temperature, placed next to log).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # ── QM/MM sweep ──────────────────────────────────────────────────────────
    if args.mm:
        run_dir = Path(args.mm)
        if not run_dir.is_dir():
            print(f"ERROR: {run_dir} is not a directory.", file=sys.stderr)
            sys.exit(1)

        mm_dirs = find_mm_dirs(run_dir)
        if not mm_dirs:
            print(f"ERROR: no mm_N directories found in {run_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"QM/MM sweep: {len(mm_dirs)} mm_N directories found")
        records = []
        for mm_idx, mm_dir in mm_dirs:
            log_path = find_log_in_dir(mm_dir)
            if log_path is None:
                print(f"  mm_{mm_idx}: no log file found, skipping")
                continue
            try:
                steps, time_ps, temp_K, _, dt_fs = parse_log(
                    log_path, dt_override=args.dt, extra_cols=args.extra or None
                )
            except Exception as e:
                print(f"  mm_{mm_idx}: {e}, skipping", file=sys.stderr)
                continue
            if args.skip:
                steps, time_ps, temp_K = steps[args.skip:], time_ps[args.skip:], temp_K[args.skip:]
            print(f"  mm_{mm_idx}: {len(steps)} rows, dt={dt_fs} fs, "
                  f"T_mean={np.mean(temp_K):.1f} K")
            records.append((mm_idx, time_ps, temp_K))

        if not records:
            print("ERROR: no valid data parsed.", file=sys.stderr)
            sys.exit(1)

        out_prefix = run_dir / (args.output or "temperature")
        plot_multi_mm(records, args.avg_window, out_prefix, args.target_T, args.per_mm)

        # CSV (concatenated)
        csv_path = out_prefix.with_name(out_prefix.name + "_rawdata.csv")
        with open(csv_path, "w") as fh:
            fh.write("mm_idx,step,time_ps,temperature_K\n")
            for mm_idx, time_ps, temp_K in records:
                for s, t, T in zip(
                    [mm_idx] * len(time_ps),
                    np.arange(len(time_ps)),
                    zip(time_ps, temp_K),
                ):
                    fh.write(f"{mm_idx},{int(s)},{T[0]:.6f},{T[1]:.4f}\n")
        print(f"Saved: {csv_path}")
        return

    # ── Single log file ───────────────────────────────────────────────────────
    if args.log is None:
        parser.print_help()
        sys.exit(1)

    log_input = Path(args.log)
    if log_input.is_dir():
        log_path = find_log_in_dir(log_input)
        if log_path is None:
            print(f"ERROR: no log file found in {log_input}", file=sys.stderr)
            sys.exit(1)
    else:
        log_path = log_input

    if not log_path.exists():
        print(f"ERROR: {log_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing: {log_path}")
    steps, time_ps, temp_K, extra, dt_fs = parse_log(
        log_path, dt_override=args.dt, extra_cols=args.extra or None
    )

    if args.skip:
        steps    = steps[args.skip:]
        time_ps  = time_ps[args.skip:]
        temp_K   = temp_K[args.skip:]
        extra    = {k: v[args.skip:] for k, v in extra.items()}

    print(
        f"  {len(steps)} data rows | dt={dt_fs} fs | "
        f"t=[{time_ps[0]:.2f}, {time_ps[-1]:.2f}] ps | "
        f"T_mean={np.mean(temp_K):.1f} ± {np.std(temp_K):.1f} K"
    )

    out_dir    = log_path.parent
    out_stem   = args.output or "temperature"
    out_prefix = out_dir / out_stem

    plot_single(time_ps, temp_K, extra, args.avg_window, None, out_prefix, args.target_T)

    # CSV
    csv_path = out_dir / f"{out_stem}_rawdata.csv"
    header   = "step,time_ps,temperature_K"
    if extra:
        header += "," + ",".join(extra.keys())
    with open(csv_path, "w") as fh:
        fh.write(header + "\n")
        for i, (s, t, T) in enumerate(zip(steps, time_ps, temp_K)):
            row = f"{int(s)},{t:.6f},{T:.4f}"
            if extra:
                row += "," + ",".join(f"{v[i]:.6f}" for v in extra.values())
            fh.write(row + "\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
