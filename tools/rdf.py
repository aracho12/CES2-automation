#!/usr/bin/env python3
"""
rdf.py — Radial Distribution Function calculator for LAMMPS trajectories
=========================================================================
Uses freud (C++ backend) for fast neighbor-list based RDF computation.

Benchmark (ces2.emd.lammpstrj, 10365 atoms):
  Ow-Ow  (~2850 atoms/type): ~6 ms/frame  → full 1274 frames in ~8 s
  Hw-Ow  (~5700+2850 atoms): ~12 ms/frame → full 1274 frames in ~15 s
  Speedup vs pure NumPy: ~22×

Usage
-----
  # All default pairs, all frames
  python tools/rdf.py test/ces2.emd.lammpstrj

  # Custom pairs (LAMMPS type IDs)
  python tools/rdf.py test/ces2.emd.lammpstrj --pairs "2-2 1-2 3-2 6-2"

  # Subset of frames & custom output
  python tools/rdf.py test/ces2.emd.lammpstrj --stride 5 --start 100 --end 500 -o output/rdf.png

  # Adjust r_max and bin count
  python tools/rdf.py test/ces2.emd.lammpstrj --r-max 12 --bins 300
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import freud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── local import ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from lammpstrj_to_traj import (
    iter_frames, frame_to_atoms,
    auto_detect_type_map, DEFAULT_TYPE_MAP,
)

# ── Default LAMMPS type labels for CES2 ──────────────────────────────────────
TYPE_LABELS = {
    1: "Hw",
    2: "Ow",
    3: "K",
    4: "H_oh",
    5: "O_oh",
    6: "Ir",
    7: "O_surf",
}

# Default pairs to compute (type1, type2, label)
DEFAULT_PAIRS: List[Tuple[int, int, str]] = [
    (2, 2, "Ow-Ow"),
    (1, 2, "Hw-Ow"),
    (3, 2, "K-Ow"),
    (6, 2, "Ir-Ow"),
    (6, 7, "Ir-O_surf"),
    (5, 2, "O_oh-Ow"),
]


def parse_pairs(pair_str: str) -> List[Tuple[int, int, str]]:
    """
    Parse a pair string like '2-2 1-2 3-2' into
    [(2,2,'Ow-Ow'), (1,2,'Hw-Ow'), (3,2,'K-Ow')].
    """
    result = []
    for token in pair_str.split():
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Bad pair spec '{token}': expected 'T1-T2'")
        t1, t2 = int(parts[0]), int(parts[1])
        label = f"{TYPE_LABELS.get(t1, str(t1))}-{TYPE_LABELS.get(t2, str(t2))}"
        result.append((t1, t2, label))
    return result


def compute_rdf(
    lammpstrj: Path,
    pairs: List[Tuple[int, int, str]],
    type_map: Dict[int, str],
    r_max: float = 10.0,
    bins: int = 200,
    stride: int = 1,
    start: int = 0,
    end: int = None,
    verbose: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute RDF for each pair using freud.

    Returns
    -------
    dict: label → (r_centers, g_r)
      r_centers : np.ndarray, shape (bins,)   — bin centres in Å
      g_r       : np.ndarray, shape (bins,)   — g(r) averaged over frames
    """
    # Initialise one freud RDF object per pair
    rdf_objs = {}
    for t1, t2, label in pairs:
        rdf_objs[label] = freud.density.RDF(bins=bins, r_max=r_max)

    frame_count = 0
    used_frames = 0
    t_start = time.perf_counter()

    for frame in iter_frames(lammpstrj):
        if frame_count < start:
            frame_count += 1
            continue
        if end is not None and frame_count >= end:
            break
        if (frame_count - start) % stride != 0:
            frame_count += 1
            continue

        atoms = frame_to_atoms(frame, type_map)
        box_arr = atoms.cell.diagonal()          # [Lx, Ly, Lz]
        pos     = atoms.positions.copy()
        types   = np.array(atoms.info["lammps_type"])

        # freud requires positions centred in [-L/2, L/2]
        pos_c = pos - box_arr / 2.0

        fbox = freud.box.Box(Lx=box_arr[0], Ly=box_arr[1], Lz=box_arr[2])

        for t1, t2, label in pairs:
            pts1 = pos_c[types == t1].astype(np.float32)
            pts2 = pos_c[types == t2].astype(np.float32)
            if len(pts1) == 0 or len(pts2) == 0:
                continue
            rdf_objs[label].compute(
                (fbox, pts2),
                query_points=pts1,
                reset=False,
            )

        used_frames += 1
        frame_count += 1

        if verbose and used_frames % 100 == 0:
            elapsed = time.perf_counter() - t_start
            rate = used_frames / elapsed
            print(f"  frame {frame_count} | {used_frames} used "
                  f"| {rate:.1f} frames/s "
                  f"| timestep {frame['timestep']}")

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"  Done: {used_frames} frames in {elapsed:.1f} s "
              f"({elapsed/used_frames*1000:.1f} ms/frame)")

    # Extract results
    results = {}
    for _, _, label in pairs:
        rdf = rdf_objs[label]
        results[label] = (rdf.bin_centers.copy(), rdf.rdf.copy())

    return results


def plot_rdf(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output: Path,
    title: str = "Radial Distribution Functions",
):
    """Plot all RDFs on a single figure and save to output."""
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    for idx, (label, (r, gr)) in enumerate(results.items()):
        ax = axes[idx // cols][idx % cols]
        ax.plot(r, gr, color=colors[idx % 10], linewidth=1.5, label=label)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("r (Å)", fontsize=11)
        ax.set_ylabel("g(r)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, r[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {output}")


def save_data(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output: Path,
):
    """Save RDF data as a single CSV: r, g(r)_pair1, g(r)_pair2, ..."""
    labels = list(results.keys())
    r = results[labels[0]][0]

    header = "r_Angstrom," + ",".join(labels)
    data = np.column_stack([r] + [gr for _, gr in results.values()])

    np.savetxt(output, data, delimiter=",", header=header, comments="")
    print(f"Saved data  → {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RDF from a LAMMPS .lammpstrj file using freud."
    )
    parser.add_argument("lammpstrj", type=Path, help="LAMMPS dump file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output plot path (default: <lammpstrj_stem>_rdf.png)",
    )
    parser.add_argument(
        "--lammps-input", type=Path, default=None,
        help="LAMMPS input file for atom type→element mapping",
    )
    parser.add_argument(
        "--pairs", type=str, default=None,
        help="Pairs to compute, e.g. '2-2 1-2 3-2 6-2'. "
             "Uses LAMMPS type IDs. Default: all CES2 pairs.",
    )
    parser.add_argument("--r-max",  type=float, default=10.0, help="Max r in Å (default: 10)")
    parser.add_argument("--bins",   type=int,   default=200,  help="Number of bins (default: 200)")
    parser.add_argument("--stride", type=int,   default=1,    help="Use every Nth frame")
    parser.add_argument("--start",  type=int,   default=0,    help="First frame index (0-based)")
    parser.add_argument("--end",    type=int,   default=None, help="Last frame index (exclusive)")
    parser.add_argument("--no-csv", action="store_true",      help="Skip saving CSV data file")
    args = parser.parse_args()

    if not args.lammpstrj.exists():
        print(f"ERROR: {args.lammpstrj} not found", file=sys.stderr)
        sys.exit(1)

    # --- Type map ---
    candidates = [
        args.lammps_input,
        args.lammpstrj.parent / "in.lammps",
        args.lammpstrj.parent.parent / "in.lammps",
    ]
    type_map = DEFAULT_TYPE_MAP
    for c in candidates:
        if c and c.exists():
            type_map = auto_detect_type_map(c)
            print(f"Type map from {c}: {type_map}")
            break

    # --- Pairs ---
    pairs = parse_pairs(args.pairs) if args.pairs else DEFAULT_PAIRS
    print(f"Pairs: {[p[2] for p in pairs]}")
    print(f"r_max={args.r_max} Å, bins={args.bins}, stride={args.stride}")
    if args.start or args.end:
        print(f"Frame range: [{args.start}, {args.end})")
    print()

    # --- Output paths ---
    stem = args.lammpstrj.stem
    base = args.output or args.lammpstrj.parent / f"{stem}_rdf.png"
    csv_out = base.with_suffix(".csv")

    # --- Compute ---
    results = compute_rdf(
        args.lammpstrj,
        pairs,
        type_map,
        r_max=args.r_max,
        bins=args.bins,
        stride=args.stride,
        start=args.start,
        end=args.end,
    )

    # --- Output ---
    n_frames_label = (
        f"frames {args.start}–{args.end or 'end'}, stride {args.stride}"
    )
    plot_rdf(results, base, title=f"RDF — {stem} ({n_frames_label})")
    if not args.no_csv:
        save_data(results, csv_out)


if __name__ == "__main__":
    main()
